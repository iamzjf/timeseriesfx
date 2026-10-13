# ETTh1：DLinear / PatchTST / iTransformer

使用固定版本TFB的已有模型、TransformerAdapter训练循环、滚动预测和指标，独立脚本完成实验和结果整理。模型源码不修改。主要结果见[REPORT.md](REPORT.md)。

## 环境

本机独立环境：`E:\conda_envs\tsfx`，名称`tsfx`，Python3.8.20。TFB官方README推荐Python3.8；相关依赖采用其`requirements-docker.txt`锁定版本。仅安装三模型所需依赖，未安装Darts/Merlion/Ray及网页报表等无关组件。外层选择性导入跳过TFB对所有模型的批量导入。

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
conda activate tsfx
```

环境激活后通过`PYTHONNOUSERSITE=1`排除用户目录依赖；`run.ps1`也始终设置此变量。直接调用解释器前应执行`$env:PYTHONNOUSERSITE='1'`，保证子进程使用同一套环境内依赖。模块实际路径在`results/environment.json`记录。

GPU采用PyTorch2.4.1 CUDA12.1 wheel。完整最终环境见`results/environment.txt`和`results/requirements-lock.txt`；后者用于相同Windows/Python/CUDA配置的精确依赖复现。

若Python3.8 pip出现TLS连接错误，可使用本机较新Python下载CPython3.8/Windows wheel，再离线安装。`scripts/download_torch.py`支持校验TLS的分块下载和官方SHA256验证。默认缓存位于`E:\conda_envs\wheelhouse_tsfx`，不会修改base或已有学习环境。

```powershell
python scripts/download_torch.py
python -m pip download -r requirements.txt --python-version 38 --platform win_amd64 --implementation cp --abi cp38 --only-binary=:all: --dest E:\conda_envs\wheelhouse_tsfx --index-url https://pypi.org/simple
& E:\conda_envs\tsfx\python.exe -m pip install --no-index --find-links E:\conda_envs\wheelhouse_tsfx -r requirements.txt
```

离线安装前必须完成CUDA wheel校验；不要把下载中的wheel作为安装源。最终`torch.__version__`应为`2.4.1+cu121`且`torch.cuda.is_available()`为True。网络失败日志保留在`results/logs/`。

## 一条命令运行

在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage all
```

顺序为A：三模型96→96、seed2024、1 epoch；B：96步正式三种子；C：192/336/720各三种子。正式训练最多10epochs、patience3。全部36项正式实验均为独立训练，另有3项Smoke Test。

默认`--resume`只跳过配置/代码指纹一致且结果文件完整的运行；中断的运行从该种子的初始状态重训，不接续优化器状态。重新训练全部实验：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage all -Fresh
```

单独运行与结果刷新：

```powershell
& E:\conda_envs\tsfx\python.exe scripts/run_experiments.py --stage smoke --resume
& E:\conda_envs\tsfx\python.exe scripts/run_experiments.py --stage main --resume
& E:\conda_envs\tsfx\python.exe scripts/run_experiments.py --stage extended --resume
& E:\conda_envs\tsfx\python.exe scripts/build_report.py
```

额外的独立核验（建议等GPU训练全部结束后执行）：

```powershell
& E:\conda_envs\tsfx\python.exe scripts/verify_results.py
& E:\conda_envs\tsfx\python.exe scripts/check_reproducibility.py
```

前者从原始CSV重新核算目标、训练归一化、指标和最佳epoch；后者将DLinear 96步seed2024完整重训到独立`reproducibility`阶段，比较预测文件哈希及指标。额外重训不覆盖正式实验，不进入正式均值，检查记录为`results/reproducibility_check.json`。

支持`--models DLinear PatchTST iTransformer`、`--horizons 96 192 336 720`、`--seeds 2024 2025 2026`、`--config 路径`。正式训练前要求相应模型已通过Smoke Test。

## 固定协议

- ETTh1官方原始数据17420行，SHA256固定；TFB Benchmark使用前14400行，训练8640、验证2880、测试2880，证据见[数据协议](docs/data_protocol.md)。
- 输入96，全7变量输入/预测；测试步长1并保留末批。验证窗口前置历史96行，测试预测只用已观测历史。
- StandardScaler仅拟合训练集；MSE/MAE使用TFB的`mse_norm/mae_norm`。模型内部历史窗口归一化作为模型设计保留。
- Adam、MSE、官方type1学习率衰减，验证最小MSE保存并重载最佳权重；无测试集调参。
- 每运行独立进程，设置Python/NumPy/PyTorch/CUDA种子、确定性算法和cuBLAS配置，禁用AMP与TF32。相同设备和版本可复现；不同设备/库版本不保证逐位相同。
- 主配置[configs/benchmark.json](configs/benchmark.json)记录12组超参数来源；完整解析配置在`results/configs/`。PatchTST官方`n_headers`无效拼写按实际默认`n_heads=8`保留。DLinear的`d_model/d_ff`等继承字段不参与线性网络。

## 输出

`results/metrics.csv`包含全部运行（含Stage/Status），`summary.csv`仅汇总完成的正式运行，标准差ddof=1。Smoke Test不会计入正式均值。`results/figures/`比较seed2024首测试窗口，先校验三模型时间戳、真实值、历史与变量顺序一致。

每次`results/runs/<stage>/<model>_h<horizon>_s<seed>/`保存：最佳权重`best.pt`（含训练scaler、配置、最佳epoch）、逐epoch损失/时间`epochs.csv`、前后向/尾批检查`smoke_checks.json`、真实结果`result.json`、全部测试预测/目标/起点`.npy`和同一样本`sample.npz`。`results/logs/`保留训练及安装日志。

源代码固定为TFB commit `889459188ec89c9cb3c7e5ec40f30b8c0c5c2529`，保存在`vendor/TFB/`。独立补充代码仅位于`scripts/`。审计及公平性修正见[docs/tfb_audit.md](docs/tfb_audit.md)，模型原理见[docs/model_explanations.md](docs/model_explanations.md)。
