param(
    [ValidateSet('smoke', 'main', 'extended', 'all')][string]$Stage = 'all',
    [string]$Python = 'E:\conda_envs\tsfx\python.exe',
    [switch]$Fresh
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run setup.ps1 first, or supply -Python.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONHASHSEED = '2024'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $Python scripts/prepare_data.py
if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
& $Python scripts/collect_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment capture failed.' }
if ($Fresh) {
    & $Python -u scripts/run_experiments.py --stage $Stage
} else {
    & $Python -u scripts/run_experiments.py --stage $Stage --resume
}
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; see results/logs.' }
& $Python scripts/build_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
& $Python scripts/verify_results.py
if ($LASTEXITCODE -ne 0) { throw 'Saved result verification failed.' }
