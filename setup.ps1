param([string]$Prefix = 'E:\conda_envs\tsfx')
$ErrorActionPreference = 'Stop'
$env:PYTHONNOUSERSITE = '1'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $Prefix 'python.exe'))) {
    conda create --prefix $Prefix python=3.8.20 pip=24.2 -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}
$taskPython = Join-Path $Prefix 'python.exe'
conda env config vars set --prefix $Prefix PYTHONNOUSERSITE=1
if ($LASTEXITCODE -ne 0) { throw 'Environment isolation configuration failed.' }
& $taskPython -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { throw 'Torch download failed. See README offline installation workaround.' }
& $taskPython -m pip install -r requirements.txt --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
if (-not (Test-Path -LiteralPath 'vendor/TFB/.git')) {
    git clone https://github.com/decisionintelligence/TFB.git vendor/TFB
    if ($LASTEXITCODE -ne 0) { throw 'TFB clone failed.' }
    git -C vendor/TFB checkout 889459188ec89c9cb3c7e5ec40f30b8c0c5c2529
    if ($LASTEXITCODE -ne 0) { throw 'TFB checkout failed.' }
}
& $taskPython scripts/prepare_data.py
if ($LASTEXITCODE -ne 0) { throw 'Data verification failed.' }
