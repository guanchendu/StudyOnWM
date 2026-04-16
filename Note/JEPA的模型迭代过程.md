## JEPA的模型迭代过程

1. Image-jepa 

   I-JEPA 通过从同一图像的上下文块(context block)去预测多个目标块(target block)的表征来学习,context encoder 用 ViT 处理可见块,predictor 是一个窄 ViT,target encoder 通过 EMA(指数移动平均)更新

2. Video-jepa

   把 I-JEPA 思路扩展到视频,在表征空间预测未来帧,开始学习"直觉物理"。

3. Video-jepa 2

   目前最重要的一个 1.2B 参数,在超过 100 万小时互联网视频上预训练,在 Something-Something v2 动作理解上取得 77.3% top-1,在 Epic-Kitchens-100 人类动作预测上达到 39.7 recall-at-5 的 SOTA这是第一个真正用于**机器人零样本规划**的 JEPA 世界模型。

4. Video-jepa2.1

   2026 年 3 月刚发布的新一代模型,引入了 Dense Predictive Loss(所有 token,包括可见和被 mask 的,都参与自监督损失)、Deep Self-Supervision(在 encoder 多个中间层都施加自监督损失)、以及多模态 tokenizer,目的是学到高质量且时间一致的 dense features

5. Vl-jepa

   VL-JEPA 预测目标文本的连续 embedding。在相同视觉编码器和训练数据下,性能超越 token-space VLM,而可训练参数减少 50%。推理时仅在需要时调用轻量文本解码器,支持 selective decoding,解码操作减少 2.85 倍

   X-Encoder 用的是 V-JEPA 2,Predictor 由 Llama 3 的 Transformer 层初始化

6. Lejepa

   Balestriero 和 LeCun 的理论升级版,用 SIGReg(Gaussian 分布正则化)来防止表征塌缩.这是 JEPA 系列**第一个有严格理论保证**的版本,解决了之前 JEPA 依赖 EMA、stop-gradient 等启发式 trick 的问题。

7. Leworldmodel 

   最新研究 第一个能从原始像素端到端稳定训练的 JEPA,只用两个损失项(下一嵌入预测损失 + Gaussian 正则)。约 15M 参数,单 GPU 几小时可训,规划速度比基于基础模型的世界模型快 48 倍 [Le-wm](https://le-wm.github.io/)。规划时用 Cross-Entropy Method 优化动作序列。LeWM 的作者包括 LeCun 和 Balestriero,是 LeJEPA 理论的直接产物。