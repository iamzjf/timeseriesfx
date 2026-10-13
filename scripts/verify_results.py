"""Independently verify saved targets, metrics, checkpoints and seed summaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = pd.read_csv(ROOT / 'data/ETTh1.csv').drop(columns='date').values
    mean, scale = raw[:8640].mean(axis=0), raw[:8640].std(axis=0)
    verified = []
    for path in sorted((ROOT / 'results/runs').glob('*/*/result.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['Status'] != 'completed':
            continue
        folder = path.parent
        horizon = row['Prediction Length']
        origins = np.load(folder / 'forecast_origins.npy')
        np.testing.assert_array_equal(origins, np.arange(11520, 14400 - horizon + 1))
        pred = np.load(folder / 'predictions_raw.npy', mmap_mode='r')
        truth = np.load(folder / 'truth_raw.npy', mmap_mode='r')
        assert pred.shape == truth.shape == (2880 - horizon + 1, horizon, 7)
        squared, absolute, count = 0., 0., 0
        for i, origin in enumerate(origins):
            np.testing.assert_array_equal(truth[i], raw[origin:origin + horizon])
            error = (truth[i].astype('float64') - pred[i].astype('float64')) / scale
            assert np.isfinite(error).all()
            squared += np.square(error).sum()
            absolute += np.abs(error).sum()
            count += error.size
        np.testing.assert_allclose([squared / count, absolute / count], [row['MSE'], row['MAE']], rtol=1e-6, atol=1e-8)
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert row['Input Length'] == 96 and row['Dataset'] == 'ETTh1'
        assert row['Seed'] in [2024, 2025, 2026]
        cfg = checkpoint['config']
        assert cfg['seq_len'] == 96 and cfg['pred_len'] == horizon and cfg['enc_in'] == cfg['c_out'] == 7
        assert cfg['norm'] is True and cfg['loss'] == 'MSE'
        assert row['Epochs'] <= (1 if row['Stage'] == 'smoke' else 10)
        np.testing.assert_allclose(checkpoint['scaler_mean'], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(checkpoint['scaler_scale'], scale, rtol=1e-12, atol=1e-12)
        assert checkpoint['scaler_n_samples_seen'] == 8640
        epoch = pd.read_csv(folder / 'epochs.csv')
        assert len(epoch) == row['Epochs']
        assert (epoch['Validation Windows'] == 2880 - horizon + 1).all()
        assert (epoch['Validation Elements'] == (2880 - horizon + 1) * horizon * 7).all()
        assert checkpoint['epoch'] == row['Best Epoch']
        np.testing.assert_allclose(checkpoint['validation_mse'], epoch['Validation MSE'].min(), rtol=1e-12)
        checks = json.loads((folder / 'smoke_checks.json').read_text(encoding='utf-8'))
        assert checks['status'] == 'passed' and checks['backward_finite']
        assert checks['test_drop_last'] is False and checks['best_checkpoint_reloaded_from_disk']
        verified.append({'stage': row['Stage'], 'model': row['Model'], 'horizon': horizon, 'seed': row['Seed'], 'verified': True})
    summary = pd.read_csv(ROOT / 'results/summary.csv')
    metrics = pd.read_csv(ROOT / 'results/metrics.csv')
    formal = metrics[(metrics['Status'] == 'completed') & metrics['Stage'].isin(['main', 'extended'])]
    if not formal.empty:
        assert formal['Device'].nunique() <= 1, 'Formal training devices differ; timings cannot be directly compared.'
    for _, group in summary.iterrows():
        rows = formal[(formal['Model'] == group['Model']) & (formal['Prediction Length'] == group['Prediction Length'])]
        assert len(rows) == group['N']
        for metric in ['MSE', 'MAE']:
            np.testing.assert_allclose(rows[metric].mean(), group[metric + ' Mean'], rtol=1e-12)
            np.testing.assert_allclose(rows[metric].std(ddof=1), group[metric + ' Std'], rtol=1e-9, atol=1e-12, equal_nan=True)
    output = {'verified_runs': len(verified), 'formal_runs': len(formal), 'checks': verified,
              'checks_description': 'Independent raw CSV targets, manual train-only scaling, full-window metrics, disk best validation checkpoints and ddof=1 summaries.'}
    (ROOT / 'results/verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('Independent verification passed for %d saved runs.' % len(verified))


if __name__ == '__main__':
    main()
