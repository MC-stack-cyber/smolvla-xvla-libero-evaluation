"""Generate a factual progress/final Markdown report from completed raw results."""
import collections
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'experiment_config.json').read_text())
names = {'wine_bottle': '酒瓶放柜顶', 'black_bowl': '黑碗放炉子', 'milk': '牛奶放篮子'}
groups = collections.defaultdict(list)
incomplete = []
for path in (ROOT / 'results').rglob('result.json'):
    d = json.loads(path.read_text())
    if d['status'] == 'completed':
        groups[(d['model'], d['task'], d['n_action_steps'])].append((path, d))
    else:
        incomplete.append((path, d))
expected = len(config['tasks']) * len(config['seeds']) * (2 + len(config['ablation']['n_action_steps']))
total = sum(len(v) for v in groups.values())
complete = total == expected and not incomplete
lines = ['# Mac 三任务扩展实验', '',
    f"生成时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')}", '',
    f"状态：{'正式评测已完成' if complete else '进行中，以下为当前数据，不能视为最终结论'}。已完成 {total}/{expected} 个正式 episode。", '',
    '[完整实验报告](../docs/report.md)', '',
    '## 实验问题', '',
    '1. 在相同初始状态与测试种子下，X-VLA 与 SmolVLA 的任务完成表现有何差异？',
    '2. 将 SmolVLA 每次执行动作数从默认 50 缩短到 10，是否改善完成率，代价是多少？', '',
    '## 环境与设计', '',
    '- Apple M5 MacBook Air，24GB 统一内存，原生 macOS，PyTorch MPS。',
    '- LeRobot 0.6.1；完整依赖版本见 requirements.lock.txt；LIBERO 上游源码提交 8f1084e3132a39270c3a13ebe37270a43ece2a01。',
    '- 检查点实际张量规模约为 SmolVLA 0.45B、X-VLA 0.88B。保存精度见 checkpoint_info.json；运行时参数类型在后续轮次 result.json 的 parameter_dtypes 字段记录。SmolVLA 使用 BF16/F32 混合参数，精度差异也可能影响性能对比。',
    '- 三个任务均由课程原始生成脚本创建，保留原场景，只改变语言与目标。',
    '- 每个设置、每个任务测试 10 次，种子固定为 0–9，初始状态 Index 0，上限 800 步。',
    '- 10 个种子是同一初始状态下的重复运行，主要反映策略采样等随机性，不等于 10 个不同物体布局或相机条件。',
    '- SmolVLA 使用相对动作及其官方处理器，X-VLA 使用绝对动作及其官方处理器。',
    '- 两个模型均使用 LIBERO 适配权重；不把这些任务称为严格未见过的零样本测试。',
    '- 正式结果以 LIBERO check_success 为准。环境错误、温控中断与短 pilot 不计入任务失败。',
    '- 执行块长度 10 是在五步速度 pilot 后、正式对照结果出来前固定的；没有按成功率挑选设置。',
    '- 历史 Linux 结果仅作背景，不与本轮 Mac 的结果混合计算。', '',
    '本轮酒瓶指令统一使用课程示例 “Put the wine bottle on top of the cabinet.”。历史输出使用 “Put the bottle on the top of the cabinet.”；历史酒瓶上限为 300 步，本轮统一为 800 步。目标相同，但措辞、步数上限和计算平台均有变化，因此不能将历史与本轮成功率差异单独归因于硬件或提示词。不同设备上的同一随机种子也不保证产生完全相同的随机张量。', '',
    '## 温控与内存', '',
    '每两秒读取 macOS 系统热状态；仅在 nominal 时启动，升至 fair/serious/critical 或监控读取失败即终止本轮。每十秒检查系统内存压力，报告余量低于 15% 时停止。每轮前要求连续 30 秒 nominal，CPU 数值库限制为两线程。GPU 未设置硬功耗上限。', '',
    '这里读取的是系统热状态，不是 CPU 摄氏温度，也不保证机器完全不发热。', '']
thermal = []
log = ROOT / 'logs/thermal.jsonl'
if log.exists():
    for line in log.read_text().splitlines():
        try:
            thermal.append(json.loads(line))
        except json.JSONDecodeError:
            pass
samples = [r for r in thermal if r.get('event') == 'sample']
if samples:
    counts = collections.Counter(r['state'] for r in samples)
    lines += ['本项目监控日志中的采样计数（含兼容性短测与下载测试）：' + '，'.join(f'{k}={v}' for k, v in sorted(counts.items())) + '。', '']
memory_samples = [r['free_percentage'] for r in thermal if r.get('event') == 'memory']
if memory_samples:
    lines += [f'系统 memory_pressure 报告的最低余量为 {min(memory_samples)}%；该指标不等于简单的空闲物理内存百分比。', '']
lines += ['## 正式结果' if complete else '## 当前结果', '',
    '| 模型 | 任务 | 每次执行步数 | 成功/次数 | 成功率 | Wilson 95% 区间 | 平均耗时/秒 |',
    '|---|---|---:|---:|---:|---:|---:|']
for (model, task, chunk), items in sorted(groups.items()):
    runs = [r for _, r in items]
    n = len(runs)
    s = sum(r['success'] for r in runs)
    p = s/n
    z = 1.96
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    mean_time = sum(r['elapsed_seconds'] for r in runs)/n
    lines.append(f'| {model} | {names[task]} | {chunk} | {s}/{n} | {p:.0%} | {max(0,center-half):.1%}–{min(1,center+half):.1%} | {mean_time:.1f} |')
if (ROOT / 'figures/success_rates.png').exists():
    lines += ['', '![成功率与区间](figures/success_rates.png)', '', '![每轮耗时](figures/episode_times.png)']
lines += ['', '耗时从开始加载策略到保存本轮结果前计时，包含模型加载与仿真，不包含 Python 初次导入和轮次间冷却。首次图形/模型编译会影响速度，不能据此宣称硬件加速倍数。', '',
    '## 成对比较', '',
    '同一任务按种子编号配对；仅对双方都完成的编号比较。不同模型的相同种子不保证相同采样噪声，且这些运行共用一个初始状态，因此配对仅作描述。使用 exact McNemar 双侧检验列出成败不一致的数量，不将小样本检验外推为跨场景泛化证据；保留多重比较限制。', '',
    '| 任务 | 比较 | 配对数 | A成功B失败 | A失败B成功 | 双侧p值 |',
    '|---|---|---:|---:|---:|---:|']
for task in config['tasks']:
    for akey, bkey, label in [(('smolvla',task,50),('xvla',task,30),'A=SmolVLA，B=X-VLA'),
                              (('smolvla',task,50),('smolvla',task,10),'A=50步，B=10步')]:
        a = {r['seed']: r for _, r in groups.get(akey, [])}
        b = {r['seed']: r for _, r in groups.get(bkey, [])}
        seeds = sorted(a.keys() & b.keys())
        if not seeds:
            continue
        ab = sum(a[s]['success'] and not b[s]['success'] for s in seeds)
        ba = sum(not a[s]['success'] and b[s]['success'] for s in seeds)
        discord = ab+ba
        pvalue = min(1, 2*sum(math.comb(discord,i) for i in range(min(ab,ba)+1))/2**discord) if discord else 1
        lines.append(f'| {names[task]} | {label} | {len(seeds)} | {ab} | {ba} | {pvalue:.4f} |')
lines += ['', '## 代表性视频与原始证据', '', '[公开版代表视频](../demos/README.md)', '']
for key, items in sorted(groups.items()):
    model, task, chunk = key
    for success, label in [(True, '成功'), (False, '失败')]:
        candidate = sorted([(p,d) for p,d in items if d['success'] == success], key=lambda x:x[1]['seed'])
        if candidate:
            path, d = candidate[0]
            folder = path.parent.relative_to(ROOT).as_posix()
            lines.append(f"- {model} / {names[task]} / 执行 {chunk} 步 / seed={d['seed']} / {label}：[视频]({folder}/rollout.mp4)，[原始记录]({folder}/result.json)。")
lines += ['', '## 结论边界', '',
    '- 只有三个任务、一个初始状态和有限种子，不能外推为全部场景的能力。',
    '- 不同模型权重的训练数据、适配方式和结构均可能不同；本实验比较具体策略检查点，不能单独隔离架构的因果作用。',
    '- 改变执行长度会改变利用新观测重新预测动作的频率，环境仍逐步提供观测；成功率变化不能单独证明某种内部认知机制。',
    '- 本轮为推理策略与模型对照，未进行 LoRA、深度/力觉接入或新增训练。',
    '- 各任务的具体失败阶段需结合视频复核，不能从失败标签推断原因。', '',
    '成功按课程 Goal 判定后立即结束，不额外强制松爪或保持若干秒。因此成功率表示达到指定 Goal 的比例，不能直接解释为全部完成松爪后稳定放置。', '',
    '## 复现', '',
    '在本目录执行：', '', '```bash',
    '.venv/bin/python scripts/prepare_tasks.py',
    '.venv/bin/python scripts/run_batch.py --model smolvla',
    '.venv/bin/python scripts/run_batch.py --model xvla',
    '.venv/bin/python scripts/summarize.py',
    '.venv/bin/python scripts/make_report.py', '```', '',
    '在本目录创建 STOP 文件会停止批量队列。队列自动跳过已有完成结果；失败或中断会停止队列等待检查。', '',
    '## 论文与实现', '',
    '- [X-VLA 论文](https://arxiv.org/abs/2510.10274)',
    '- [X-VLA LeRobot 官方接入](https://huggingface.co/docs/lerobot/xvla)',
    '- [SmolVLA 论文](https://arxiv.org/abs/2506.01844)',
    '- [LIBERO 论文](https://arxiv.org/abs/2306.03310)',
    '- [Apple 系统热状态定义](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum)', '']
(ROOT / '实验报告.md').write_text('\n'.join(lines), encoding='utf-8')
print('Report updated:', total, '/', expected)
