# 环境与执行记录

- 原始默认环境：E:\miniconda\python.exe，Python3.12.8，torch2.11.0+cpu，无法使用CUDA。
- 硬件：AMD Ryzen 5 5600H，6核12线程，约16GiB内存；RTX3050 Laptop，4GiB显存，驱动546.30。
- 按用户要求，新环境位于E盘，名称tsfx：E:\conda_envs\tsfx。原计划E:\miniconda\envs\tsfx因目录只允许管理员写入而改为上述可写目录。通过conda envs_dirs注册后可conda activate tsfx。
- 采用TFB README推荐的Python3.8，本机安装3.8.20。TFB requirements-docker.txt中与本实验相关的numpy/pandas/scipy/scikit-learn/matplotlib/statsmodels/torch/reformer-pytorch/tqdm版本保留；其余用于无关模型或网页/调度框架的依赖不安装。所有实际递归依赖在results/requirements-lock.txt记录。
- Python3.8 pip直接访问PyTorch站点和已有镜像出现TLS EOF；改由base较新Python正常验证TLS下载CPython3.8/Windows wheel，再离线安装。官方CUDA wheel的SHA256为a48b991cd861266523cbed4705f89bef09669d5d2bbfa2524486156f74a222a8。下载源download.pytorch.org；索引返回的download-r2端点遇403，已回退原官方域名。分块下载验证HTTP范围和最终哈希，不关闭TLS校验。
- 下载期间临时安装torch2.4.1 CPU wheel，完成三模型1epoch Smoke Test及独立结果核验。该CPU检查副本保存在results/archive/cpu_smoke/，只用于诊断安装与代码；正式结果目录中的Smoke Test会使用最终GPU环境重新运行，不混入正式三种子均值。
- 线程数4，num_workers=0适配Windows；每个运行独立进程；确定性算法开启，AMP/TF32关闭。最终运行环境以results/environment.txt和各运行Device字段为准。
- 随机性采用TFB的fix_all_random_seed（full模式），该官方函数还会禁用cuDNN；本实验保留此行为。严格确定性、逐批损失记录和Windows GPU调度均可能增加耗时，因此这里的耗时只用于本机相同协议内比较，不等同于论文或其他平台的速度。
- 工作目录早期创建过程被中断，可能留有隐藏的.venv-tfb临时文件；它不注册为有效环境，也不用于任何实验。环境安装过程中的失败日志保留在results/logs，未将失败当作成功。
- 上述临时目录的递归清理被自动审批拦截（工具返回blocked by policy，未提供进一步理由）；较安全的conda环境卸载也因它不是有效环境而不能执行，因此保留残留文件。此限制未阻塞E盘正式环境或实验。
- GPU Smoke Test首次磁盘重载发现torch.__version__实际为TorchVersion子类，放入checkpoint元数据后无法以weights_only=True加载；已将运行配置中的版本明确转换为普通str，保留安全权重加载。失败日志为results/logs/failed_checkpoint_version_object.log。模型训练本身正常，该失败不计作完成实验。
- 最终入口验证发现Python3.8默认允许读取用户site，NumPy1.24.4、Matplotlib3.7.5、Pillow10.4.0实际来自用户目录。它们版本符合锁定，但不满足完全隔离；交付前将相同版本完整装入tsfx，并通过conda环境变量及run.ps1设置PYTHONNOUSERSITE=1。训练用的NumPy共502个.py/.pyd/.dll文件迁移前后全部哈希一致（results/archive/numpy_isolation_check.json）；Pandas/SciPy/sklearn/torch本来就在tsfx。隔离后再次执行全指标核验和DLinear完整重训检查，不修改既有模型或数据。
- 重复输出图片时，Windows对已有all_variables_h336.png的w+b打开出现OSError22；图表现改为先写临时文件，再原子替换，重复生成全部9图已通过。该问题发生于报告绘图，不影响训练、checkpoint或保存的预测数组。

本次是固定输入96、固定种子及统一协议的复现实验；部分官方脚本原先使用更长输入，已按用户要求覆盖。保留官方推荐其余超参数和10epoch早停预算；不宣称逐项复刻论文最佳表。
