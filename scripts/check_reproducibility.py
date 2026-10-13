"""Repeat one complete run in a separate stage, without overwriting benchmark runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import run_experiments as runner
    if '--worker' in sys.argv:
        runner.execute_one(runner.read_config(ROOT / 'configs/benchmark.json'),
                           'reproducibility', 'DLinear', 96, 2024)
        return
    first = ROOT / 'results/runs/main/DLinear_h96_s2024'
    second = ROOT / 'results/runs/reproducibility/DLinear_h96_s2024'
    baseline = json.loads((first / 'result.json').read_text(encoding='utf-8'))
    assert baseline['Status'] == 'completed'
    first_hash = digest(first / 'predictions_raw.npy')
    environment = dict(os.environ)
    environment.update(PYTHONHASHSEED='2024', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1', OMP_NUM_THREADS='4')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                   cwd=str(ROOT), env=environment, check=True)
    repeat = json.loads((second / 'result.json').read_text(encoding='utf-8'))
    second_hash = digest(second / 'predictions_raw.npy')
    passed = (first_hash == second_hash and
              all(baseline[key] == repeat[key] for key in ['MSE', 'MAE', 'Best Epoch', 'Epochs', 'Model Configuration']))
    result = {'passed': passed, 'model': 'DLinear', 'horizon': 96, 'seed': 2024,
              'original_configuration_sha256': baseline['Configuration SHA256'],
              'original_predictions_sha256': first_hash, 'repeat_predictions_sha256': second_hash,
              'original_MSE': baseline['MSE'], 'repeat_MSE': repeat['MSE'],
              'original_MAE': baseline['MAE'], 'repeat_MAE': repeat['MAE'],
              'description': 'A separate full training, validation selection and test run; predictions must match bit for bit. Canonical benchmark files are preserved.'}
    (ROOT / 'results/reproducibility_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not passed:
        raise AssertionError('Reproducibility check failed; see results/reproducibility_check.json')
    print('DLinear full-run deterministic replay passed: identical predictions, metrics and best epoch.')


if __name__ == '__main__':
    main()
