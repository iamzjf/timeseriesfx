"""Execute smoke, main, and extended ETTh1 experiments using unchanged TFB code.

Examples (from the project root):
    python scripts/run_experiments.py --stage all --resume
    python scripts/run_experiments.py --stage smoke --models DLinear

Each run uses a fresh subprocess so RNG state, Python hash seed, and CUDA memory
statistics are independent. --resume skips only complete, verified run outputs
whose configuration fingerprints still match. Interrupted training restarts.
"""
import argparse
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["DLinear", "PatchTST", "iTransformer"]


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_commit():
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/TFB"), "rev-parse", "HEAD"], text=True
    ).strip()


def build_run_config(config, stage, model, horizon, seed):
    import platform
    import torch
    params = dict(config["models"][model][str(horizon)])
    params.update({
        "seq_len": config["input_length"], "horizon": horizon, "norm": True,
        "num_epochs": 1 if stage == "smoke" else config["training"]["num_epochs"],
        "patience": config["training"]["patience"], "num_workers": 0,
        "task_name": "long_term_forecast", "use_amp": 0,
    })
    params.pop("source", None)
    params.pop("notes", None)
    return {
        "stage": stage, "model": model, "horizon": horizon, "seed": seed,
        "dataset": config["dataset"], "model_parameters": params,
        "training": config["training"], "evaluation": config["evaluation"],
        "tfb_commit": git_commit(),
        "instrumentation_sha256": digest_file(ROOT / "scripts/tfb_runtime.py"),
        "runner_sha256": digest_file(Path(__file__)),
        "environment_signature": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def run_name(model, horizon, seed):
    return "{}_h{}_s{}".format(model, horizon, seed)


def run_directory(stage, model, horizon, seed):
    return ROOT / "results/runs" / stage / run_name(model, horizon, seed)


def is_complete(path, expected_fingerprint=None):
    try:
        result = read_config(path / "result.json")
        if result.get("Status") != "completed":
            return False
        if expected_fingerprint and result.get("Configuration SHA256") != expected_fingerprint:
            return False
        for filename in ("best.pt", "epochs.csv", "smoke_checks.json", "sample.npz",
                         "predictions_raw.npy", "truth_raw.npy", "forecast_origins.npy"):
            artifact = path / filename
            if not artifact.is_file() or artifact.stat().st_size == 0:
                return False
        return read_config(path / "smoke_checks.json").get("status") == "passed"
    except (OSError, ValueError):
        return False


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def execute_one(config, stage, model, horizon, seed):
    # Import torch only after environment flags were set by the parent launcher.
    from tfb_runtime import (
        AuditedTransformerAdapter, evaluate_rolling, set_reproducibility,
        synchronize, write_json,
    )
    import numpy as np
    import pandas as pd
    import torch

    run_config = build_run_config(config, stage, model, horizon, seed)
    run_dir = run_directory(stage, model, horizon, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = "{}_{}".format(stage, run_name(model, horizon, seed))
    log_path = ROOT / "results/logs" / (run_id + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "Stage": stage, "Status": "running", "Model": model,
        "Dataset": config["dataset"]["name"],
        "Input Length": config["input_length"], "Prediction Length": horizon,
        "Seed": seed, "Configuration SHA256": fingerprint(run_config),
        "TFB Commit": run_config["tfb_commit"],
        "Started UTC": datetime.now(timezone.utc).isoformat(),
        "Run Directory": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "Log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(run_dir / "result.json", record)
    write_json(ROOT / "results/configs" / (run_id + ".json"), run_config)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            handler = logging.StreamHandler(sys.stderr)
            logging.getLogger().addHandler(handler)
            try:
                print("RUN_CONFIG " + json.dumps(run_config, ensure_ascii=False), flush=True)
                set_reproducibility(seed, config["training"].get("num_threads", 4))
                dataset = config["dataset"]
                data_path = ROOT / dataset["path"]
                checksum = digest_file(data_path)
                if dataset.get("sha256") and checksum != dataset["sha256"]:
                    raise ValueError("ETTh1 SHA256 does not match pinned dataset")
                data = pd.read_csv(data_path, parse_dates=["date"]).set_index("date")
                columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
                if list(data.columns) != columns:
                    raise ValueError("Unexpected ETTh1 feature order: " + str(list(data.columns)))
                if not data.index.is_monotonic_increasing or data.index.has_duplicates:
                    raise ValueError("ETTh1 timestamps must be increasing and unique")
                if not (np.diff(data.index.values) == np.timedelta64(1, "h")).all():
                    raise ValueError("ETTh1 timestamps must be hourly")
                if not np.isfinite(data.values).all():
                    raise ValueError("Missing or non-finite ETTh1 data")
                train_end, val_end, test_end = (dataset[key] for key in ("train_end", "val_end", "test_end"))
                if not 0 < train_end < val_end < test_end <= len(data):
                    raise ValueError("Invalid split boundaries")
                data = data.iloc[:test_end]
                ratio = train_end / val_end
                if int(val_end * ratio) != train_end:
                    raise ValueError("Floating point split ratio does not reproduce train_end")
                adapter = AuditedTransformerAdapter(model, run_dir, run_config, **run_config["model_parameters"])
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    gpu_free, gpu_total = torch.cuda.mem_get_info()
                    record.update({"Device": torch.cuda.get_device_name(0),
                                   "GPU Total MiB": gpu_total / 1024 ** 2,
                                   "GPU Free Before MiB": gpu_free / 1024 ** 2})
                else:
                    record["Device"] = "CPU"
                synchronize()
                started = time.perf_counter()
                adapter.forecast_fit(data.iloc[:val_end].copy(), train_ratio_in_tv=ratio)
                synchronize()
                record["Training Time"] = time.perf_counter() - started
                if not adapter.checks.get("backward_finite") or not adapter.check_point:
                    raise AssertionError("Training did not create valid gradients and best checkpoint")
                record["Parameters"] = sum(p.numel() for p in adapter.model.parameters())
                record["Trainable Parameters"] = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
                record["Epochs"] = len(adapter.epoch_rows)
                record["Best Epoch"] = adapter.best_epoch
                record["Validation MSE"] = adapter.epoch_rows[adapter.best_epoch - 1]["Validation MSE"]
                record["GPU Peak Training Allocated MiB"] = (
                    torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
                )
                # Reload the physical checkpoint rather than relying only on RAM.
                # Contains trusted local tensor weights and primitive metadata.
                checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
                if checkpoint["epoch"] != adapter.best_epoch:
                    raise AssertionError("Persisted best checkpoint epoch mismatch")
                adapter.check_point = checkpoint["model_state_dict"]
                adapter.model.load_state_dict(adapter.check_point)
                adapter.checks["best_checkpoint_reloaded_from_disk"] = True
                metrics = evaluate_rolling(
                    adapter, data, train_end, val_end, test_end, horizon,
                    config["evaluation"]["stride"], run_dir,
                )
                record.update(metrics)
                record.update({
                    "Status": "completed", "Data SHA256": checksum,
                    "Completed UTC": datetime.now(timezone.utc).isoformat(),
                    "Metric Scale": "train-only StandardScaler; TFB mse_norm/mae_norm",
                    "GPU Peak Allocated MiB": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
                    "GPU Peak Reserved MiB": torch.cuda.max_memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
                    "Model Configuration": dict(vars(adapter.config)),
                })
                write_json(run_dir / "result.json", record)
                write_json(ROOT / "results/configs" / (run_id + ".json"), {
                    **run_config, "resolved_model_parameters": dict(vars(adapter.config))
                })
                print("RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
            except BaseException as error:
                record.update({"Status": "failed", "Error": str(error),
                               "Traceback": traceback.format_exc(),
                               "Failed UTC": datetime.now(timezone.utc).isoformat()})
                write_json(run_dir / "result.json", record)
                traceback.print_exc()
                raise
            finally:
                logging.getLogger().removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/benchmark.json"))
    parser.add_argument("--stage", choices=["smoke", "main", "extended", "all"], default="all")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+", type=int, choices=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    models = args.models or MODEL_NAMES
    seeds = args.seeds or config["seeds"]
    if args.worker:
        if args.stage == "all" or len(models) != 1 or not args.horizons or len(args.horizons) != 1 or len(seeds) != 1:
            parser.error("Internal worker expects one stage/model/horizon/seed")
        execute_one(config, args.stage, models[0], args.horizons[0], seeds[0])
        return
    stages = ["smoke", "main", "extended"] if args.stage == "all" else [args.stage]
    for stage in stages:
        horizons = [96] if stage in ("smoke", "main") else [192, 336, 720]
        if args.horizons:
            horizons = [h for h in horizons if h in args.horizons]
        stage_seeds = (args.seeds or [config["seeds"][0]]) if stage == "smoke" else seeds
        for model in models:
            if horizons and stage != "smoke":
                smoke_roots = list((ROOT / "results/runs/smoke").glob(model + "_h96_s*"))
                if not any(is_complete(path) for path in smoke_roots):
                    raise RuntimeError("Run a successful 1-epoch smoke test for {} before formal experiments".format(model))
            for horizon in horizons:
                for seed in stage_seeds:
                    run_config = build_run_config(config, stage, model, horizon, seed)
                    run_dir = run_directory(stage, model, horizon, seed)
                    if args.resume and is_complete(run_dir, fingerprint(run_config)):
                        print("SKIP COMPLETE " + str(run_dir), flush=True)
                        continue
                    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                               "--stage", stage, "--models", model, "--horizons", str(horizon),
                               "--seeds", str(seed), "--worker"]
                    environment = dict(os.environ)
                    environment.update({"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                                        "OMP_NUM_THREADS": str(config["training"].get("num_threads", 4))})
                    print("START {} {} horizon={} seed={}".format(stage, model, horizon, seed), flush=True)
                    subprocess.run(command, cwd=str(ROOT), env=environment, check=True)
                    if not is_complete(run_dir, fingerprint(run_config)):
                        raise RuntimeError("Missing or invalid output files after run: " + str(run_dir))
        print("STAGE FINISHED: " + stage, flush=True)


if __name__ == "__main__":
    main()
