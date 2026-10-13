# TFB 源码与评测协议审计

审计日期：2026-10-12。审计对象：官方仓库 `decisionintelligence/TFB`，commit `889459188ec89c9cb3c7e5ec40f30b8c0c5c2529`。下列行号均针对该 commit，路径均相对项目根目录。本文记录源码证据与执行建议；实际采用配置和结果以 `results/configs/`、`REPORT.md` 为准。

## 1. 三个模型已经受到完整支持

| 模型 | TFB 实现 | 已有适配器 | 官方 ETTh1 脚本 |
|---|---|---|---|
| DLinear | `vendor/TFB/ts_benchmark/baselines/time_series_library/models/DLinear.py:7` | `TransformerAdapter` | `vendor/TFB/scripts/multivariate_forecast/ETTh1_script/DLinear.sh` |
| PatchTST | `vendor/TFB/ts_benchmark/baselines/time_series_library/models/PatchTST.py:24` | 同上 | `vendor/TFB/scripts/multivariate_forecast/ETTh1_script/PatchTST.sh` |
| iTransformer | `vendor/TFB/ts_benchmark/baselines/time_series_library/models/iTransformer.py:10` | 同上 | `vendor/TFB/scripts/multivariate_forecast/ETTh1_script/iTransformer.sh` |

适配器源文件：`vendor/TFB/ts_benchmark/baselines/time_series_library/adapters_for_transformers.py:57`。模型工厂位于 94–133 行，实际模型初始化位于 78–79 行，统一前向入口位于 81–91 行。训练/预测主体继承自 `vendor/TFB/ts_benchmark/baselines/deep_forecasting_model_base.py`。不需要重新实现模型或为缺失模型补适配器。可以在仓库外继承该适配器，添加日志、磁盘 checkpoint、严格评测统计和资源记录。

## 2. 官方脚本实际配置

| 模型 | H | 官方 L | batch | lr | d_model | d_ff | encoder layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| DLinear | 96 | 336 | 32 | 0.005 | 不生效 | 不生效 | 不生效 |
| DLinear | 192 | 336 | 64 | 0.001 | 不生效 | 不生效 | 不生效 |
| DLinear | 336 | 336 | 32 | 0.005 | 不生效 | 不生效 | 不生效 |
| DLinear | 720 | 336 | 64 | 0.0001（默认） | 不生效 | 不生效 | 不生效 |
| PatchTST | 96 | 96 | 32（默认） | 0.0001（默认） | 512 | 2048 | 1 |
| PatchTST | 192 | 512 | 32（默认） | 0.0001（默认） | 128 | 512 | 2（默认） |
| PatchTST | 336 | 512 | 64 | 0.0001 | 128 | 512 | 2（默认） |
| PatchTST | 720 | 512 | 64 | 0.0001 | 128 | 512 | 2（默认） |
| iTransformer | 96 | 96 | 32（默认） | 0.0001（默认） | 256 | 256 | 2 |
| iTransformer | 192 | 512 | 32（默认） | 0.0001（默认） | 256 | 256 | 2 |
| iTransformer | 336 | 512 | 32（默认） | 0.0001（默认） | 128 | 128 | 2 |
| iTransformer | 720 | 96 | 32（默认） | 0.0001（默认） | 512 | 512 | 2 |

默认值来源：`adapters_for_transformers.py:10–54`。三者默认均为 Adam、MSE、最多 10 epochs、patience=3、`lradj=type1`、dropout=0.1、8 heads、无 AMP。优化器在 `deep_forecasting_model_base.py:117–137`，学习率调度在 `baselines/utils.py:15–31`。DLinear 移动平均窗口 25，季节项和趋势项分别使用共享线性层，`individual=False`；其 d_model/d_ff 不参与计算，不能将脚本里这些无效参数当成 DLinear 的网络宽度。

本任务要求所有 L=96，因此无法逐字复现原 ETTh1 脚本所有行；应该将 L=96 明确记录为用户指定的统一实验约束，再保留有依据的各模型特有配置。任何资源受限调整都应提前记录，不能根据测试指标挑配置。

**发现的脚本拼写问题：** PatchTST 96 的脚本写了 `n_headers=2`，但模型源码 `PatchTST.py:60` 读取 `config.n_heads`，所以原脚本实际生效的是默认 `n_heads=8`。复现实际 TFB 行为应明确写 `n_heads=8`，不要误报为 2 heads。

## 3. 已核验的数据长度与划分

`vendor/TFB/config/rolling_forecast_config.json:27–35` 给出 `tv_ratio=0.8`、ETTh1 的 `train_ratio_in_tv=0.75`、stride=1。`evaluation/strategy/rolling_forecast.py:216–218` 按 metadata.length（缺省为文件实际长度）计算边界。训练/验证再由 `baselines/utils.py:162–176` 切分，验证数据额外向前保留 L 个观测作为历史输入，验证目标始于真正的验证边界。

- TFB [正式论文](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) Table 5（PDF 第 6 页）明确列出 ETTh1 长度为 14,400、7 个变量、6:2:2 划分。结合官方 ETT loader 的固定边界，得到 train `[0,8640)`、val `[8640,11520)`、test `[11520,14400)`，等价于 ETT 常用 12/4/4 个 30 天月份划分。
- 如果直接将 17,420 行原始 ETT CSV 交给上述比例，则得到 train `[0,10452)`、val `[10452,13936)`、test `[13936,17420)`；这不是前一种协议。
- 本项目保留原始 17,420 行 CSV，不修改原始字节；Benchmark 只使用前 14,400 行。原始文件固定 commit、SHA256、列序和时间间隔的验证证据见 `docs/data_protocol.md` 与 `data/manifest.json`。协议依据来自论文及官方代码；没有声称将 TFB 云盘包与原始 CSV 做过逐字节比较。

上述证据已解决数据长度和边界问题。不可直接把原始全文件乘比例后仍称为论文固定月份设置。三个模型必须引用同一个协议对象。

在标准 14,400 行协议与 L=96 下，预测长度 H 对应训练窗口数 `8640-96-H+1`，验证和测试窗口数均为 `2880-H+1`。测试每小时滑动一次，第一测试预测起点固定为行 11,520，最后一个预测起点为 `14400-H`。随着滚动时点推进，使用该时点已经观测的历史测试值是合法滚动预测，不能用未来目标值。

## 4. 归一化、目标与时间信息

- `deep_forecasting_model_base.py:396–400` 先分 train/val，再只用 train 拟合 StandardScaler，未用验证/测试值拟合。列级均值与方差用于 train/val/test，同一模型的预测再逆变换回原单位。
- `evaluation/strategy/forecasting.py:83–102` 的评价 scaler 同样只在训练段拟合。
- TFB 主报告指标 `mse_norm`/`mae_norm` 见 `evaluation/metrics/regression_metrics.py:142–155`，为训练 scaler 的标准化尺度。MSE/MAE 应按所有窗口、预测步长、7 个变量的全部标量等权汇总。若同时提供原始尺度指标，必须另名记录，不能混入主表。
- `target_channel=null` 表示全部列预测（`utils/data_processing.py:61–117`），等同本任务 Features=M。原 CSV 的 date 只用作索引，不属于目标。7 个变量列序应固定并校验。
- PatchTST（`PatchTST.py:95–123`）及 iTransformer（`iTransformer.py:69–85`）内部还做每个输入窗口的均值/方差归一化，并把输出还原。这是其架构的一部分，仅依赖历史输入，不是测试集泄漏；统一外部 StandardScaler 不要求删除这些机制。
- iTransformer 接收 input_mark。`layers/Embed.py:143–155` 会把已知日历特征（小时数据通常有小时、星期、日、年内日 4 个特征）作为额外 tokens，输出仍截取前 7 个数值变量（`iTransformer.py:82`）。DLinear/PatchTST 不使用这些 marks。保留官方行为时，应说明该架构差异；日历信息本身没有读取未来数值。
- 适配器 `adapters_for_transformers.py:83–88` 的 decoder future 数值明确置零，历史标签只取输入末端 label_len。三个模型本身也不依赖未来 decoder 真值。

## 5. 验证、checkpoint 与尾批

已有优点：

- 验证 loader `drop_last=False`（`deep_forecasting_model_base.py:416–423`）。
- 测试 `forecast` loader `shuffle=False, drop_last=False`（同文件 549–552）；批量预测由 `rolling_forecast.py:39–55,111–119` 保留最后不足 batch_size 的窗口，没有丢弃。
- 最优验证权重已有 `check_point` 内存副本（同文件 490–495）；预测前加载（529–530；批量预测入口亦有）。

建议外层修正：

1. 原 `validate` 把每个 batch 的平均 loss 再取无权平均（同文件 306–346），并且验证 shuffle=True（421 行）。最后不足 batch_size 的 batch 因此被过度加权，且组成随 shuffle 变化。重写为累计 squared-error sum / 标量总数，最好验证固定顺序。这修正统计权重，不改变 train/val/test 边界或 MSE 定义。
2. 原 checkpoint 只在内存，需要外层把最佳权重、训练 scaler、配置、随机种子、最佳 epoch/validation MSE 写盘；在测试前明确恢复最佳权重。
3. 原训练日志缺少完整 epoch train/val loss、训练时间和资源用量，需要外层补充并与 smoke 结果分开保存。
4. 训练多变量默认 drop_last=True（383、431 行），允许保留官方设置；公平性要求强调测试不丢尾批。记录该训练设置。验证与测试所有样本都必须纳入。

## 6. 随机种子、依赖与兼容性

`utils/random_utils.py` 提供 Python/NumPy/PyTorch 种子函数，`ForecastingStrategy.execute` 根据 deterministic 配置调用。独立入口应对 2024、2025、2026 分别在模型初始化和 DataLoader 创建前设置种子，并记录 CUDA/cuDNN/CUBLAS 确定性选项；Python hash seed 应在进程启动前设置。每个模型/预测长度/种子均独立初始化训练。

README 推荐 Python 3.8；复现实验脚本应优先 `requirements-docker.txt`。该文件锁定 PyTorch 2.4.1、NumPy 1.24.4、pandas 1.5.3、scikit-learn 1.3.2 等。默认 `requirements.txt` 使用较宽版本范围，安装到 NumPy 2 会遇到 `baselines/utils.py:40` 的 `np.Inf` 已删除问题。优先兼容环境；若环境必须较新，可在仓库外继承 EarlyStopping 修复为 `np.inf`，明确记录。

`time_series_library/__init__.py` 会一次导入很多无关模型，可能要求非本实验必需的 reformer-pytorch 等依赖。若精简依赖，应通过明确的外部导入层导入原始模块，并保留包路径供相对 import 使用；不能把模型替换成自行简化的实现。

## 7. 实验验收建议

Smoke 应在三模型各 1 个完整 epoch 后检查 `[B,96,7] -> [B,H,7]`、finite loss、finite gradients、优化器更新、最优 checkpoint 和完整测试窗口数量。正式结果主表不纳入 smoke。

同一样本可视化建议固定 seed=2024、测试首个预测起点、目标变量 OT、同一 H，将三个模型预测放在同一时间轴，分别保存训练标准化尺度或清楚标注的原始单位。不得分别从各模型挑最好看的样本。跨 H 比较须提醒窗口总数不同，且每个 H 为独立训练。参数量用真实可训练参数统计，训练时间包含说明清晰的训练/验证边界；CUDA 峰值显存与 CPU-only 情况分别记录，不能填零冒充 GPU 测量。
