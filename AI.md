# AI引论

## Chap01 概论

## Chap02 数学基础

- **样本空间&样品点**
  - **样本空间**：所有可能的结果
  - **样本点**：样本空间元素
- **随机事件**
  - 满足某些条件的样本点组成的集合。它是样本空间 Ω 的子集，记为 𝐴,𝐵, ...
  - 关系
    - 包含，并，交，对立（是和不是），互斥（红灯和绿灯（还有黄灯））
- **古典概型**
  - **有限性**：只有有限个试验结果（样本）
  - **等可能性**：每个试验结果（样本）在一次试验中出现的可能性相等；
- **条件概率**
  - 在事件𝐵已经出现的条件下，事件𝐴发生的概率，记作 P(A | B).
  - **P(A | B) = P(A ∩ B) / P(B)** => **P(A ∩ B)=P(A | B) · P(B)**
- **独立事件**
  - 若 P(AB) = P(A) · P(B),则**相互独立**。
  - 若 P(ABCDEFG) = P(A) P(B) …… P(G)，则**相互独立**。
  - 若 P(AB) = P(A) · P(B)；P(CB) = P(C) · P(B)；P(AC) = P(A) · P(C)，则**两两独立**。
  - 两两独立的事件组**不一定**相互独立。
  - 相互独立的事件组**一定**两两独立。
- **全概率公式**
  - ![1](https://pic.imgdb.cn/item/63f71c5ff144a010073f3d26.jpg)
  - ![2](https://pic.imgdb.cn/item/63f71c82f144a010073f73f4.jpg)
  - ![3](https://pic.imgdb.cn/item/63f71cd1f144a010073ff0c5.jpg)
- **贝叶斯公式**
  - ![4](https://pic.imgdb.cn/item/63f72553f144a010074e3223.jpg)
- **随机变量**
  - ![5](https://pic.imgdb.cn/item/63f751c8f144a0100797e05b.jpg)
  - ![6](https://pic.imgdb.cn/item/63f751f2f144a01007981ade.jpg)
  - ![7](https://pic.imgdb.cn/item/63f7520af144a01007983a5b.jpg)
- **数学期望**
  - ![8](https://pic.imgdb.cn/item/63f75234f144a0100798720b.jpg)
  - ![9](https://pic.imgdb.cn/item/63f7524bf144a01007989334.jpg)
- **方差与标准差**
  - ![10](https://pic.imgdb.cn/item/63f7525ff144a0100798b02d.jpg)
  - 方差的算术平方根 $\sqrt{𝐷(X)}$ 称为 X 的标准差或均方差，记为σ (X)
- **协方差**
  - 若随机变量 𝑋 的期望 𝐸(𝑋) 和 𝑌 的期望 𝐸(𝑌) 存在，则称
$$Cov(X,Y)=E{[X-E(x)][Y-E(Y)]}$$为 𝑋 与 𝑌 的协方差。
  - 若随机变量 𝑋，𝑌的方差和协方差均存在，且 𝐷(𝑋) > 0，𝐷(𝑌) > 0，则
  - $$ρ(X,Y)=\frac{Cov(X,Y)}{\sqrt{D(X)·D(Y)}}$$称为 𝑋，𝑌 的相关系数。

## Chap03 python

## Chap04 搜索

- 目标（goal）：即我们要去“找”什么；什么时候可以结束搜索
- 状态（state）：这里面其实包括三个主要的部分，开始状态（initial states），目标状态（goal states），当前状态（current state）
- 动作（actions）：智能体可以采取的行动/决策
- （状态）转移模型（transition model）：当前状态随着动作会怎么变化
- 动作代价函数（action cost function）：每个动作要消耗多大的成本
