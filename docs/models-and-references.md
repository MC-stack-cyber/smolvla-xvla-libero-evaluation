# 参考论文与 X-VLA 模型说明

## 文献及引用用途

1. Jinliang Zheng 等，2025，**X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model**。[论文](https://arxiv.org/abs/2510.10274)。用于解释领域soft prompt和模型背景，不用于代替本项目测量。
2. Mustafa Shukor 等，2025，**SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics**。[论文](https://arxiv.org/abs/2506.01844)。用于说明轻量VLA研究背景。
3. Bo Liu 等，2023，**LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning**。[论文](https://arxiv.org/abs/2306.03310)。用于说明仿真基准。本项目调用其环境基础，不是完成了该论文全部终身学习实验。

## 从输入到执行：以本地实现为准

策略先接收外部相机图像、腕部相机图像、语言指令和机器人自身状态。图像不包含深度通道，状态不是力觉。主视角图像特征与文本进入Florence视觉语言编码部分，辅助视角保留视觉特征供后续动作网络使用。

机器人状态让模型知道机械臂当前在哪里、夹爪状态如何。动作Transformer还接收时间条件和噪声，以及学习得到的领域soft prompt。这里的prompt是权重中的向量，不是我们临时写的“先抬高再移动”这类文字。它提供与数据领域有关的条件，但本项目没有独立移除或替换它，因此不能证明它是成功的决定因素。

模型从噪声出发，在时间条件下多次迭代预测连续动作序列。本地检查点一次预测30步，使用10次迭代。不要把10次生成迭代理解为执行10步：前者在网络内部，后者决定多久重新拍照和预测。默认执行30步，对照实验只执行10步后重新预测，预测块本身仍为30步。

动作并非直接原样送入仿真。官方处理器执行归一化逆变换及动作表示转换。X-VLA本地接口使用绝对末端位姿和6D旋转表示，再转为LIBERO可接受的控制动作；SmolVLA接口使用相对动作。这种差异可能影响表现，但没有在本轮被单独控制。

## 本地事实和核对入口

本轮检查点为lerobot/xvla-libero，约0.88B参数，Float32、MPS。保存的处理器domain_id为3。soft prompt长度、领域槽数等是配置，不能由槽数推断实际训练了多少种机器人。没有运行7B模型，也没有重训模型、训练LoRA、增加深度/力觉或文字规划阶段。

完整证据中phase2/analysis/checkpoint_configs保存配置快照；checkpoint_info.json记录检查点信息。本地环境源码位于LeRobot的policies/xvla/modeling_xvla.py、soft_transformer.py、processor_xvla.py。交付保留我们自己的复现代码及配置，但未打包整个第三方安装或权重。恢复环境还需对应依赖和模型资源，可参考[LeRobot官方X-VLA文档](https://huggingface.co/docs/lerobot/xvla)。

## 如何正确讲“X-VLA为什么好”

论文告诉我们模型设计有哪些选择；本地配置告诉我们实际使用哪些选择；干预实验告诉我们在特定条件下行为如何变化。这三类证据不能相互替代。可以说同10步执行时优势保留，或X-VLA更常完成目标对齐。不能说本实验已经证明soft prompt、Florence或flow matching分别贡献了多少。尤其flow matching在两个模型中都有，不能列作X-VLA独有的解释。
