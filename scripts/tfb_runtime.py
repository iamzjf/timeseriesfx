"""Instrumentation around unchanged TFB model, training, and rolling forecast code.

Only the broad time_series_library package initializer is skipped: it eagerly
imports unrelated baselines. Requested model modules and all their dependencies
are loaded from their original files in vendor/TFB.
"""
import csv
import importlib
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TFB_ROOT = ROOT / "vendor" / "TFB"
sys.path.insert(0, str(TFB_ROOT))


def _selective_tsl_import():
    name = "ts_benchmark.baselines.time_series_library"
    if name not in sys.modules:
        importlib.import_module("ts_benchmark.baselines")
        package = types.ModuleType(name)
        package.__path__ = [str(TFB_ROOT / "ts_benchmark/baselines/time_series_library")]
        package.__package__ = name
        package.__file__ = str(Path(package.__path__[0]) / "__init__.py")
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
        setattr(sys.modules["ts_benchmark.baselines"], "time_series_library", package)


_selective_tsl_import()

import numpy as np
import torch
from torch.utils.data import DataLoader
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.evaluation.metrics.regression_metrics import mae_norm, mse_norm
from ts_benchmark.evaluation.strategy.rolling_forecast import (
    RollingForecast,
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)
from ts_benchmark.utils.random_utils import fix_all_random_seed


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_reproducibility(seed, num_threads=4):
    # The launching process sets PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG
    # before this process imports torch. Keep TFB's full deterministic policy.
    fix_all_random_seed(seed)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_class(model_name):
    module = importlib.import_module(
        "ts_benchmark.baselines.time_series_library.models." + model_name
    )
    return getattr(module, model_name)


class AuditedTransformerAdapter(TransformerAdapter):
    """Retain upstream forecast_fit, Adam, early stopping, and batch_forecast.

    Changes: sample-weighted ordered validation, durable best checkpoint, epoch
    loss/time logging, and first-batch forward/backward checks. No model rewrite.
    """

    def __init__(self, model_name, run_dir, run_config, **kwargs):
        super().__init__(model_name, model_class(model_name), **kwargs)
        self.run_dir = Path(run_dir)
        self.run_config = run_config
        self.epoch_rows = []
        self.checks = {}
        self._train_squared_error = 0.0
        self._train_elements = 0
        self._train_batches = 0
        self._epoch_start = None
        self._optimizer = None
        self._ordered_validation_loader = None
        self._step_hook = None
        self.best_epoch = None
        self._forward_hook = None

    def _init_model(self):
        model = super()._init_model()

        def first_forward(module, inputs, output):
            expected = [inputs[0].shape[0], self.config.horizon, self.config.enc_in]
            if list(output.shape) != expected:
                raise RuntimeError("Unexpected output shape: {} != {}".format(list(output.shape), expected))
            if not torch.isfinite(output).all().item():
                raise RuntimeError("Non-finite first forward output")
            self.checks.update({
                "input_shape": list(inputs[0].shape),
                "input_mark_shape": list(inputs[1].shape),
                "decoder_shape": list(inputs[2].shape),
                "output_shape": list(output.shape),
                "forward_finite": True,
            })
            self._forward_hook.remove()
            self._forward_hook = None

        self._forward_hook = model.register_forward_hook(first_forward)
        return model

    def _init_criterion_and_optimizer(self):
        criterion, optimizer = super()._init_criterion_and_optimizer()
        self._optimizer = optimizer

        def before_first_step(opt, args, kwargs):
            if self.checks.get("backward_finite"):
                return
            square_norm = 0.0
            gradient_tensors = 0
            for parameter in self.model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError("Non-finite parameter gradients")
                square_norm += parameter.grad.detach().double().square().sum().item()
                gradient_tensors += 1
            if square_norm <= 0 or gradient_tensors == 0:
                raise RuntimeError("First backward pass produced no nonzero gradients")
            self.checks.update({"backward_finite": True, "gradient_l2": square_norm ** 0.5,
                                "gradient_tensors": gradient_tensors})
            print("FIRST_BATCH_CHECK " + json.dumps(self.checks), flush=True)
            write_json(self.run_dir / "smoke_checks.json", self.checks)

        self._step_hook = optimizer.register_step_pre_hook(before_first_step)
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        if self.model.training and self._epoch_start is None:
            synchronize()
            self._epoch_start = time.perf_counter()
        result = super()._process(input, target, input_mark, target_mark)
        if self.model.training and torch.is_grad_enabled():
            output = result["output"][:, -self.config.horizon:, :self.config.c_out]
            truth = target[:, -self.config.horizon:, :self.config.c_out]
            squared_error = (output.detach() - truth.detach()).square().sum().item()
            if not np.isfinite(squared_error):
                raise RuntimeError("Non-finite training loss")
            if "first_batch_loss" not in self.checks:
                self.checks["first_batch_loss"] = squared_error / truth.numel()
                self.checks["target_shape"] = list(truth.shape)
                self.checks["loss_finite"] = True
            self._train_squared_error += squared_error
            self._train_elements += truth.numel()
            self._train_batches += 1
        return result

    def validate(self, valid_data_loader, series_dim, criterion):
        # Upstream uses shuffle=True and an unweighted mean of batch losses.
        # Reuse the same upstream Dataset and retain every validation window.
        if self._ordered_validation_loader is None:
            self._ordered_validation_loader = DataLoader(
                valid_data_loader.dataset, batch_size=self.config.batch_size,
                shuffle=False, drop_last=False, num_workers=self.config.num_workers,
            )
        loader = self._ordered_validation_loader
        self.model.eval()
        device = next(self.model.parameters()).device
        weighted_loss, elements = 0.0, 0
        with torch.no_grad():
            for input, target, input_mark, target_mark in loader:
                input, target, input_mark, target_mark = (
                    value.to(device) for value in (input, target, input_mark, target_mark)
                )
                result = self._process(input, target, input_mark, target_mark)
                output = result["output"][:, -self.config.horizon:, :series_dim]
                truth = target[:, -self.config.horizon:, :series_dim]
                output, truth = self._post_process(output, truth)
                value = (criterion(output, truth) + result.get("additional_loss", 0)).item()
                if not np.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                weighted_loss += value * truth.numel()
                elements += truth.numel()
        synchronize()
        if not elements or not self._train_elements:
            raise RuntimeError("Empty training or validation data")
        validation_loss = weighted_loss / elements
        epoch = len(self.epoch_rows) + 1
        row = {
            "Epoch": epoch,
            "Train MSE": self._train_squared_error / self._train_elements,
            "Validation MSE": validation_loss,
            "Learning Rate": self._optimizer.param_groups[0]["lr"],
            "Epoch Time": time.perf_counter() - self._epoch_start,
            "Train Batches": self._train_batches,
            "Train Elements": self._train_elements,
            "Validation Windows": len(loader.dataset),
            "Validation Elements": elements,
        }
        self.epoch_rows.append(row)
        with (self.run_dir / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.epoch_rows)
        print("EPOCH " + json.dumps(row), flush=True)
        self._train_squared_error = 0.0
        self._train_elements = self._train_batches = 0
        self._epoch_start = None
        self.model.train()
        return validation_loss

    def save_checkpoint(self, model):
        state = super().save_checkpoint(model)
        self.best_epoch = len(self.epoch_rows)
        payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in state.items()},
            "model_name": self.model_name,
            "epoch": self.best_epoch,
            "validation_mse": self.epoch_rows[-1]["Validation MSE"],
            "config": dict(vars(self.config)),
            "run_config": self.run_config,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_var": self.scaler.var_.tolist(),
            "scaler_n_samples_seen": int(self.scaler.n_samples_seen_),
            "checkpoint_purpose": "best validation model for inference; reruns restart training from seed",
        }
        destination = self.run_dir / "best.pt"
        temporary = self.run_dir / "best.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination)
        return state


def evaluate_rolling(adapter, series, train_end, val_end, test_end, horizon, stride, run_dir):
    """Use upstream rolling indices, batch makers, model inference, and metrics."""
    from sklearn.preprocessing import StandardScaler

    run_dir = Path(run_dir)
    eval_scaler = StandardScaler().fit(series.iloc[:train_end].values)
    np.testing.assert_array_equal(adapter.scaler.mean_, eval_scaler.mean_)
    np.testing.assert_array_equal(adapter.scaler.scale_, eval_scaler.scale_)
    if int(adapter.scaler.n_samples_seen_) != train_end:
        raise AssertionError("Normalization fitted on unexpected data")

    index_list = RollingForecast._get_index(val_end, test_end - val_end, horizon, stride)
    if index_list[0] != val_end or index_list[-1] + horizon != test_end:
        raise AssertionError("Unexpected rolling forecast endpoints")
    batch_maker = RollingForecastEvalBatchMaker(series.iloc[:test_end], index_list)
    predict_batch_maker = RollingForecastPredictBatchMaker(batch_maker)
    predictions, batch_sizes = [], []
    synchronize()
    started = time.perf_counter()
    while predict_batch_maker.has_more_batches():
        batch = adapter.batch_forecast(horizon, predict_batch_maker)
        if batch.shape[1:] != (horizon, series.shape[1]) or not np.isfinite(batch).all():
            raise AssertionError("Invalid forecast output")
        predictions.append(batch)
        batch_sizes.append(len(batch))
    synchronize()
    inference_time = time.perf_counter() - started
    predictions = np.concatenate(predictions, axis=0)
    truth = batch_maker.make_batch_eval(horizon)["target"]
    if truth.shape != predictions.shape or len(truth) != len(index_list):
        raise AssertionError("Forecast/target count or shape mismatch; possible dropped tail batch")

    # Compute precisely the official average over complete rolling windows.
    official_mse = float(np.mean([mse_norm(t, p, eval_scaler) for t, p in zip(truth, predictions)]))
    official_mae = float(np.mean([mae_norm(t, p, eval_scaler) for t, p in zip(truth, predictions)]))
    normalized_truth = eval_scaler.transform(truth.reshape(-1, truth.shape[-1]))
    normalized_predictions = eval_scaler.transform(predictions.reshape(-1, predictions.shape[-1]))
    residual = normalized_truth - normalized_predictions
    vector_mse = float(np.mean(residual ** 2))
    vector_mae = float(np.mean(np.abs(residual)))
    np.testing.assert_allclose([official_mse, official_mae], [vector_mse, vector_mae], rtol=1e-10, atol=1e-12)
    if not np.isfinite([official_mse, official_mae]).all():
        raise AssertionError("Non-finite evaluation metric")

    np.save(run_dir / "predictions_raw.npy", predictions, allow_pickle=False)
    np.save(run_dir / "truth_raw.npy", truth, allow_pickle=False)
    np.save(run_dir / "forecast_origins.npy", np.asarray(index_list, dtype=np.int64), allow_pickle=False)
    origin = index_list[0]
    context = series.iloc[origin - adapter.config.seq_len:origin].values
    sample_truth = truth[0]
    sample_prediction = predictions[0]
    np.savez_compressed(
        run_dir / "sample.npz",
        columns=np.asarray(series.columns, dtype="U"),
        origin=np.asarray(origin),
        context_timestamps=series.index[origin - adapter.config.seq_len:origin].values,
        forecast_timestamps=series.index[origin:origin + horizon].values,
        context_raw=context,
        truth_raw=sample_truth,
        prediction_raw=sample_prediction,
        context_normalized=eval_scaler.transform(context),
        truth_normalized=eval_scaler.transform(sample_truth),
        prediction_normalized=eval_scaler.transform(sample_prediction),
    )
    adapter.checks.update({
        "test_prediction_shape": list(predictions.shape),
        "test_windows": len(index_list),
        "test_last_batch_size": batch_sizes[-1],
        "test_drop_last": False,
        "test_prediction_finite": True,
        "validation_shuffle": False,
        "validation_drop_last": False,
        "scaler_train_rows": train_end,
        "official_metrics_match_vectorized": True,
        "best_checkpoint_exists": (run_dir / "best.pt").is_file(),
        "first_test_origin": origin,
        "last_test_origin": index_list[-1],
        "status": "passed",
    })
    write_json(run_dir / "smoke_checks.json", adapter.checks)
    return {"MSE": official_mse, "MAE": official_mae,
            "Inference Time": inference_time, "Test Windows": len(index_list),
            "Last Test Batch": batch_sizes[-1]}
