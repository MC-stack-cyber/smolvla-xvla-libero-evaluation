# 结果复核与仿真复跑

## 复核公开结果

[236 轮总表](../experiments/phase2/all_mac_episodes.csv)的 `path` 列相对于 `experiments/`。每个路径指向一轮实验的 `result.json`。第二阶段各轮还附有 `stage_metrics.json`。置黑组的目录保存了策略实际接收的首帧。

在仓库根目录运行以下脚本可重建总表与图表。总表只需 Python 标准库，绘图需要 NumPy 和 Matplotlib。

```bash
python3 experiments/phase2/scripts/export_all.py
python3 experiments/scripts/plot_results.py
python3 experiments/phase2/scripts/plots.py
```

原始审计记录覆盖完整实验交付中的视频、轨迹和指标文件，公开版保留其中的逐轮结果、阶段指标及代表视频。审计中的 SHA-256 可用于核对随仓库提供的文件。

- [90 轮基线审计](../experiments/audit.json)
- [116 轮主方案审计](../experiments/phase2/audit.json)
- [10 轮补充实验审计](../experiments/phase2/followup/audit.json)
- [20 轮配对校准审计](../experiments/phase2/clearance_calibration/audit.json)

## 复跑仿真

原始环境为 Apple Silicon macOS、Python 3.12、PyTorch MPS、LeRobot 0.6.1，以及 LIBERO 提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。[原始环境](../experiments/environment.json) · [依赖版本快照](../experiments/requirements.lock.txt)

完整复跑需要原课程的 SmolVLA 检查点、LIBERO 仿真资产和兼容的 macOS 运行环境。依赖清单记录原始版本，macOS 上的 LIBERO 渲染还依赖本机兼容配置。准备环境和资源后，将它们放到下列位置，路径以 `experiments/` 为根。

| 资源 | 路径 |
| --- | --- |
| [LIBERO 源码](https://github.com/Lifelong-Robot-Learning/LIBERO) | `vendor/LIBERO/` |
| 课程离线 LIBERO 资产 | `offline/vla_course_offline/libero_assets/` |
| 课程离线 SmolVLA 检查点 | `offline/vla_course_offline/models/smolvla_libero/` |
| [X-VLA LIBERO 检查点](https://huggingface.co/lerobot/xvla-libero) | `models/xvla-libero/` |

仿真队列会跳过已有的已完成记录。复跑时，使用单独的工作副本并先将该副本里的 `experiments/results/`、`experiments/phase2/results/`、`experiments/phase2/followup/results/`、`experiments/phase2/clearance_calibration/results/` 移至该副本外。然后使用准备好的 Python 环境运行：

```bash
cd experiments
python3 scripts/prepare_tasks.py
python3 scripts/run_batch.py --pilot
python3 scripts/run_batch.py
python3 phase2/scripts/run_batch.py
python3 phase2/scripts/followup_batch.py
python3 phase2/scripts/calibration_batch.py
```

批处理脚本会调用 macOS 温控监测。各阶段队列与参数见 [实验配置](../experiments/experiment_config.json)、[主方案](../experiments/phase2/protocol.json)、[40 步补充协议](../experiments/phase2/followup/protocol.json)和 [80 步配对协议](../experiments/phase2/clearance_calibration/protocol.json)。
