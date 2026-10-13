# -*- coding: utf-8 -*-
"""Build tables, aligned prediction plots and a Chinese report from actual runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODELS = ['DLinear', 'PatchTST', 'iTransformer']


def save_figure(fig, path, dpi):
    # Publish a complete image atomically; do not truncate a file open in a viewer.
    temporary = path.with_name(path.stem + '.tmp.png')
    fig.savefig(temporary, dpi=dpi, format='png')
    temporary.replace(path)


def table(frame):
    if frame.empty:
        return '尚无已完成的结果。'
    def fmt(value):
        if isinstance(value, float):
            return '—' if np.isnan(value) else '%.6f' % value
        return str(value).replace('|', '/')
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def figures(formal):
    paths = []
    folder = RESULTS / 'figures'
    folder.mkdir(exist_ok=True)
    if formal.empty:
        return paths
    for horizon in sorted(formal['Prediction Length'].unique()):
        selected = formal[(formal['Prediction Length'] == horizon) & (formal['Seed'] == 2024)]
        if set(selected['Model']) != set(MODELS):
            continue
        samples = {row['Model']: np.load(ROOT / row['Run Directory'] / 'sample.npz') for _, row in selected.iterrows()}
        base = samples[MODELS[0]]
        for item in samples.values():
            for key in ['columns', 'origin', 'forecast_timestamps', 'context_timestamps', 'truth_raw', 'context_raw']:
                np.testing.assert_array_equal(base[key], item[key])
        cols = list(base['columns'])
        fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(base['context_timestamps'], base['context_raw'][:, j], color='gray', label='History')
            ax.plot(base['forecast_timestamps'], base['truth_raw'][:, j], color='black', label='Ground truth', linewidth=1.6)
            for model in MODELS:
                ax.plot(base['forecast_timestamps'], samples[model]['prediction_raw'][:, j], label=model, alpha=.85)
            ax.axvline(base['forecast_timestamps'][0], color='gray', linestyle=':')
            ax.set_ylabel(cols[j] + ' (raw)')
            ax.grid(alpha=.2)
        axes[0].legend(ncol=5)
        axes[-1].set_xlabel('Timestamp (hourly); first test origin; seed 2024')
        fig.suptitle('ETTh1: 96 -> %d; aligned multivariate forecasts' % horizon)
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = 'all_variables_h%d.png' % horizon
        save_figure(fig, folder / filename, dpi=140)
        plt.close(fig)
        paths.append(filename)
        j = cols.index('OT')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(base['context_timestamps'], base['context_raw'][:, j], color='gray', label='History')
        ax.plot(base['forecast_timestamps'], base['truth_raw'][:, j], color='black', label='Ground truth')
        for model in MODELS:
            ax.plot(base['forecast_timestamps'], samples[model]['prediction_raw'][:, j], label=model)
        ax.axvline(base['forecast_timestamps'][0], color='gray', linestyle=':')
        ax.set(title='ETTh1 OT: 96 -> %d; seed 2024; first test origin' % horizon, ylabel='OT (raw scale)', xlabel='Timestamp')
        ax.legend(ncol=5)
        ax.grid(alpha=.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = 'prediction_OT_h%d.png' % horizon
        save_figure(fig, folder / filename, dpi=160)
        plt.close(fig)
        paths.append(filename)
        for item in samples.values():
            item.close()
    return paths


def main():
    RESULTS.mkdir(exist_ok=True)
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((RESULTS / 'runs').glob('*/*/result.json'))]
    frame = pd.DataFrame(records)
    columns = ['Stage', 'Status', 'Model', 'Dataset', 'Input Length', 'Prediction Length', 'Seed', 'MSE', 'MAE', 'Parameters', 'Training Time']
    for col in columns:
        if col not in frame:
            frame[col] = pd.Series(dtype='object')
    for col in frame.columns:
        if frame[col].map(lambda x: isinstance(x, (dict, list))).any():
            frame[col] = frame[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    frame = frame.sort_values(['Stage', 'Model', 'Prediction Length', 'Seed'])
    frame.to_csv(RESULTS / 'metrics.csv', index=False, encoding='utf-8-sig')
    formal = frame[(frame['Status'] == 'completed') & frame['Stage'].isin(['main', 'extended'])].copy()
    summary_cols = ['Model', 'Dataset', 'Input Length', 'Prediction Length', 'N', 'MSE Mean', 'MSE Std', 'MAE Mean', 'MAE Std', 'Parameters', 'Training Time Mean']
    rows = []
    for key, group in formal.groupby(summary_cols[:4]):
        rows.append(dict(zip(summary_cols[:4], key), **{
            'N': len(group), 'MSE Mean': group['MSE'].mean(), 'MSE Std': group['MSE'].std(ddof=1),
            'MAE Mean': group['MAE'].mean(), 'MAE Std': group['MAE'].std(ddof=1),
            'Parameters': group['Parameters'].iloc[0], 'Training Time Mean': group['Training Time'].mean()}))
    summary = pd.DataFrame(rows, columns=summary_cols)
    summary.to_csv(RESULTS / 'summary.csv', index=False, encoding='utf-8-sig')
    paths = figures(formal)
    visual_observation = ''
    if 'prediction_OT_h192.png' in paths:
        visual_observation = ('\n\n**固定样本的视觉观察：** 192步OT图中，约10月29日至30日真实值明显下行，三个模型仍保持较平滑的日内起伏，未充分跟上该转折。'
                              '这只描述seed2024首个测试窗口，不能代表全部窗口或全7变量的平均表现。短历史及未提供未来外部驱动可能与此有关，但本实验没有相关消融，原因尚不能确定。')
    if not summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for model in MODELS:
            group = summary[summary['Model'] == model].sort_values('Prediction Length')
            for ax, metric in zip(axes, ['MSE', 'MAE']):
                ax.errorbar(group['Prediction Length'], group[metric + ' Mean'], yerr=group[metric + ' Std'].fillna(0), marker='o', label=model, capsize=3)
                ax.set(xlabel='Prediction length', ylabel=metric + ' (normalized)', title=metric + ': mean +/- sample std')
                ax.grid(alpha=.2)
        axes[0].legend()
        fig.tight_layout()
        save_figure(fig, RESULTS / 'figures/metrics_by_horizon.png', dpi=160)
        plt.close(fig)
        paths.append('metrics_by_horizon.png')
    smoke = frame[(frame['Stage'] == 'smoke') & (frame['Status'] == 'completed')]
    missing = []
    for model in MODELS:
        for h in [96, 192, 336, 720]:
            for seed in [2024, 2025, 2026]:
                if formal[(formal['Model'] == model) & (formal['Prediction Length'] == h) & (formal['Seed'] == seed)].empty:
                    missing.append('%s H=%d seed=%d' % (model, h, seed))
    environment = RESULTS / 'environment.json'
    env = json.loads(environment.read_text(encoding='utf-8')) if environment.exists() else {}
    analysis = []
    for h, group in summary.groupby('Prediction Length'):
        if len(group) == 3 and (group['N'] == 3).all():
            best = group.loc[group['MSE Mean'].idxmin()]
            analysis.append('本次 H=%d 的三个种子中，%s 的平均 MSE 最低（%.6f）；这是当前配置和数据集内的观察，不能据此宣称普遍优势。' % (h, best['Model'], best['MSE Mean']))
    for model in MODELS:
        group = summary[(summary['Model'] == model) & (summary['N'] == 3)].sort_values('Prediction Length')
        if len(group) == 4:
            first, last = group.iloc[0], group.iloc[-1]
            trajectory = '、'.join('H%d: %.4f' % (r['Prediction Length'], r['MSE Mean']) for _, r in group.iterrows())
            analysis.append('**实验观察：%s。** MSE轨迹为%s。H720相对H96变化%+.4f；长预测任务的平均误差还受到不同窗口集合、直接预测头及推荐容量/学习率变化影响，不能由这四个点单独证明长期建模能力。' % (model, trajectory, last['MSE Mean'] - first['MSE Mean']))
    baseline = summary[summary['Prediction Length'] == 96]
    long_i = formal[(formal['Model'] == 'iTransformer') & (formal['Prediction Length'] == 720)].sort_values('Seed')
    if len(long_i) == 3:
        best_epochs = '/'.join(str(int(x)) for x in long_i['Best Epoch'])
        actual_epochs = '/'.join(str(int(x)) for x in long_i['Epochs'])
        analysis.append('**训练过程观察。** iTransformer在H720、种子2024/2025/2026的最佳验证epoch分别为%s，实际训练epoch为%s。最终指标对应验证集选择的解；若早停选中了早期权重，不能据此单独断言网络结构的表达能力不足。额外预算和早停策略应另设实验，在训练/验证集合内预先定义，不回头按本次测试指标调整当前配置。' % (best_epochs, actual_epochs))
    if len(baseline) == 3 and (baseline['N'] == 3).all():
        detail = []
        for _, row in baseline.iterrows():
            detail.append('%s：%d个参数，平均训练%.1f秒' % (row['Model'], row['Parameters'], row['Training Time Mean']))
        analysis.append('**实验观察：效率。** ' + '；'.join(detail) + '。训练时间包含验证与checkpoint保存；各运行可能提前停止，耗时既反映网络成本，也反映实际训练轮数，不能直接等同于每轮吞吐量。')
        analysis.append('**解释与待验证假设。** DLinear利用固定平滑分解和共享线性映射，以很少参数表达稳定的滞后关系；这使它有可能在较短历史和有限预算下保持竞争力。PatchTST压缩时间token并共享跨变量权重，iTransformer通过变量注意力引入联合信息；这些结构可能更适合特定非线性或跨变量模式，也可能需要不同训练预算。当前实验没有单独消融容量、窗口归一化或日历tokens，因此不能把分数差异归因于某一模块。相关论文提出的设计动机已在第2节与实际观察区分。')
    explanation = (ROOT / 'docs/model_explanations.md').read_text(encoding='utf-8')
    explanation = '\n'.join('##' + line if line.startswith('#') else line for line in explanation.splitlines())
    report = '''# ETTh1 三模型可复现实验报告

## 1. 实验目的

在固定输入96小时、多变量输入/输出、统一数据划分与指标的条件下，比较DLinear、PatchTST、iTransformer。正式结果与1 epoch Smoke Test严格分开。当前完成 Smoke Test %d/3，正式实验 %d/36。

## 2. 核心架构区别

%s

## 3. 数据集介绍

ETTh1为小时级电力变压器数据，7变量：HUFL/HULL/MUFL/MULL/LUFL/LULL/OT。原始CSV17420行，本Benchmark依照TFB论文Table5使用前14400行；其余3020行不参与。来源与SHA256见[data/manifest.json](data/manifest.json)，划分证据见[数据协议](docs/data_protocol.md)。

## 4. 训练与评测协议

训练[0,8640)、验证[8640,11520)、测试[11520,14400)。输入96，预测96/192/336/720，Features=M，全7变量预测。验证和测试窗口可使用边界以前的已观测历史，目标完全位于对应集合。测试滑动步长1，窗口数依次2785/2689/2545/2161，保留末批。

StandardScaler仅拟合训练8640行。指标为TFB mse_norm/mae_norm，按所有窗口、预测步、变量等权平均，且与独立向量计算相互验证。测试指标在训练标准化尺度计算；预测曲线显示原始单位OT，不能用单条曲线代表全7变量误差。

每个模型/长度/种子独立训练。正式采用TFB默认最多10 epochs、patience=3、Adam、MSE损失与type1学习率衰减；最小验证MSE选择checkpoint，重新从磁盘加载后才测试。不做测试集选参。三个种子2024/2025/2026，标准差ddof=1。训练按官方多变量loader使用drop_last=True；验证和测试均False。确定性采用TFB full模式（含禁用cuDNN），另强制确定性算法，禁用AMP/TF32；这些设置会影响运行速度，耗时仅用于本机相同协议内比较。完整超参数见[benchmark.json](configs/benchmark.json)和results/configs/。

外层子类修正验证末批权重并固定验证顺序，添加日志、检查和磁盘checkpoint；复用官方模型、训练循环、早停和滚动推理。vendor保持原样。跳过TFB初始化时无关模型的批量导入；未安装未用的Darts/Merlion/Ray等依赖。详见[审计](docs/tfb_audit.md)。

## 5. 实验环境

```json
%s
```

独立conda环境E:\\conda_envs\\tsfx；完整版本、GPU信息与pip检查见[environment.txt](results/environment.txt)，精确依赖见[requirements-lock.txt](results/requirements-lock.txt)。固定TFB commit记录在环境及每个运行结果中。

交付环境通过PYTHONNOUSERSITE=1完全排除用户site。收尾时将原先从用户site读取的同版本NumPy/绘图库装入tsfx；训练用NumPy的502个源文件/二进制逐个哈希完全一致，记录见[隔离核验](results/archive/numpy_isolation_check.json)。未改变模型、数据或超参数。

## 6. 实验结果

### 正式实验：均值与样本标准差

%s

### Smoke Test：仅检查流程，不纳入正式汇总

%s

逐次实验、参数量和耗时见[metrics.csv](results/metrics.csv)，汇总见[summary.csv](results/summary.csv)。每次运行包含best.pt、epochs.csv、检查记录、全部测试预测和同一起点sample.npz。GPU分配/缓存峰值为PyTorch进程统计，不等于整机nvidia-smi显存占用。

## 7. 预测曲线

固定seed2024、首个测试起点2017-10-24 00:00，三个模型在绘图前逐项核验时间戳、历史、真实值和变量顺序完全一致。图例中的预测来自各长度独立训练的checkpoint，未按测试表现挑选样本。

%s

## 8. 性能差异分析

%s

不同预测长度下模型容量与官方推荐超参数也可能变化，而且合法测试窗口数不同；不能把曲线变化完全归因于预测长度。三个种子只能给出初步波动估计，不能证明统计显著性。10 epoch预算保留TFB默认，可用于规范流程学习，但不代表每个模型经过充分调参或重现原论文最佳数值。

## 9. 复现过程中遇到的问题

默认Python为CPU版PyTorch；另建E盘tsfx。E:\\miniconda\\envs无写权限，因此环境放E:\\conda_envs。Python3.8 pip网络TLS连接失败，采用正常校验TLS的下载及离线安装，CUDA wheel还校验官方SHA256。GPU Smoke Test首次重载发现TorchVersion元数据类型不兼容安全权重加载，已转成普通字符串并重新运行；失败记录保留，不计入完成实验。TFB的PatchTST脚本n_headers拼写未被模型读取，保留实际n_heads=8并明确记录。原始数据17420行不能直接按比例划分，已按TFB论文14400行解决。官方验证简单平均批次loss会使末批权重偏大，已在外层修正。网络与安装原始日志保存在results/logs，环境处理详见[环境记录](docs/environment_notes.md)。

尚未完成的正式实验：%s

未完成条目表示当前没有可报告的正式指标；原因应结合对应日志和运行状态判断，不能把缺项或Smoke Test冒充正式结果。

## 10. 后续改进

先完成所有长度和种子，再扩展训练预算、在训练/验证集合内进行预先定义的超参数搜索；增加数据集、更多种子及误差分解。若研究输入长度、内部归一化、日历token或容量的影响，另建独立消融实验，保留本次Benchmark协议与结果。

一条命令继续或复现：`powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Stage all`。默认跳过配置和代码指纹相同且文件完整的已完成运行；重新训练全部运行使用README中不带--resume的命令。
''' % (len(smoke), len(formal), explanation, json.dumps(env, ensure_ascii=False, indent=2), table(summary), table(smoke[columns]), ('\n\n'.join('![%s](results/figures/%s)' % (p, p) for p in paths) or '尚无三模型同一样本的正式预测图。') + visual_observation, '\n\n'.join(analysis) or '正式三种子对比尚未齐全，暂不作模型优劣判断。', '无（全部36项完成）。' if not missing else '\n\n' + '\n'.join('- ' + item for item in missing))
    config = json.loads((ROOT / 'configs/benchmark.json').read_text(encoding='utf-8'))
    config_rows = []
    for model in MODELS:
        for h in [96, 192, 336, 720]:
            cfg = config['models'][model][str(h)]
            config_rows.append({'Model': model, 'H': h, 'Batch': cfg['batch_size'], 'LR': cfg['lr'],
                                'd_model': cfg['d_model'] if model != 'DLinear' else '—',
                                'd_ff': cfg['d_ff'] if model != 'DLinear' else '—',
                                'Layers': cfg.get('e_layers', '—'), 'Heads': cfg.get('n_heads', '—'),
                                'Patch/Stride': '16/8' if model == 'PatchTST' else '—'})
    report = report.replace('## 5. 实验环境', '### 采用的主要超参数\n\n' + table(pd.DataFrame(config_rows)) +
                            '\n\nDLinear共享季节/趋势线性层，moving_avg=25；三模型dropout=0.1（DLinear预测分支未使用dropout）。\n\n## 5. 实验环境')
    replay_path = RESULTS / 'reproducibility_check.json'
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        source = formal[(formal['Model'] == 'DLinear') & (formal['Prediction Length'] == 96) & (formal['Seed'] == 2024)]
        if len(source) == 1 and replay.get('original_configuration_sha256') == source.iloc[0]['Configuration SHA256']:
            status = '通过：全部测试预测逐位一致，MSE/MAE及最佳epoch相同' if replay['passed'] else '未通过，详见检查记录'
            report = report.replace('## 10. 后续改进', '### 独立重训复现核验\n\nDLinear 96步、seed2024完整重训核验' + status +
                                    '。重训结果另存reproducibility阶段，不计入正式三种子均值。见[检查记录](results/reproducibility_check.json)。\n\n## 10. 后续改进')
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
    print('Saved metrics.csv, summary.csv, %d figures, REPORT.md; formal=%d smoke=%d' % (len(paths), len(formal), len(smoke)))


if __name__ == '__main__':
    main()
