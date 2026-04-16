## Leworldmodel

实现的数据集的技术细节

1 使用tworoom 数据集为例 dataloader的时候 使用frameskip的方法

 例如 假设 frameskip的值为5， 数据会如下加载

  o1 o6 o11 o16

而这个常数num_steps 对应的是 4

所以它仍然是滑窗，只不过是在“原始时间轴上滑动 1 步”，而观测之间间隔 5。

2 在target 数据集上的构建

假设一条样本为

[o1, o2, o3, o4]

pixels[:, :-1]  表示取除最后一帧之外的所有帧，所以得到：

[o1, o2, o3]

pixels[:, 1:]

表示取除第一帧之外的所有帧，所以得到：

[o2, o3, o4]

于是就自动对齐成了三组训练对：

"pixels_t":   pixels[:, :-1]
"pixels_tp1": pixels[:, 1:]

“拿前一帧当输入，后一帧当监督信号”。

3.训练与预测

预测 latent representation。输入当前图像，编码成一个 state representation再结合 action预测下一个 state representation。 latent dynamics model

再在这个 learned representation 空间里做规划 / eval

给定起点 observation，先编码成起点 representation。

然后会采样很多 action 序列，把它们“喂给” dynamics model 往前 rollout，得到未来的一串预测 representation / reward / value。
最后从这些候选序列里挑一个最好的。