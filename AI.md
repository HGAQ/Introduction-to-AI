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
  - $$Cov(X,Y)=E{[X-E(x)\][Y-E(Y)]}$$
    为 𝑋 与 𝑌 的协方差。
  - 若随机变量 𝑋，𝑌的方差和协方差均存在，且 𝐷(𝑋) > 0，𝐷(𝑌) > 0，则
  - $$ρ(X,Y)=\frac{Cov(X,Y)}{\sqrt{D(X)·D(Y)}}$$称为 𝑋，𝑌 的相关系数。

## Chap03 python

- intended left blank

## Chap04-05 搜索

- 目标（goal）：即我们要去“找”什么；什么时候可以结束搜索
- 状态（state）：这里面其实包括三个主要的部分，开始状态（initial states），目标状态（goal states），当前状态（current state）
- 动作（actions）：智能体可以采取的行动/决策
- （状态）转移模型（transition model）：当前状态随着动作会怎么变化
- 动作代价函数（action cost function）：每个动作要消耗多大的成本
- 深度优先搜索：DFS
  - 一条路走到黑，不撞南墙不回头
  - stack！！！（后进先出）
  - 不是代价最小的，也不是最短的
- 广度优先搜索：BFS
  - 搜索一步的所有情况
  - queue！！！（先进先出）
  - 最短但是不是代价最小的
- 一致代价搜索：UFS
  - 我们不应该只关注单边的成本，而是要考虑总成本（从开始节点累积到当前节点），广搜是UFS的一种
  - 代价->步骤
  - 代价轮廊
  - priority queue！！（优先队列）
    - 保存所有节点与优先值
    - 优先队列通常依靠 堆（heap）来实现
    - 依靠树的层级来维持顺序，父母节点一定小于等于孩子节点（树是完整的，即只有最底层的右边有空缺）
  - 假设最佳答案是有限代价，并且边的最低代价是正的，算法完备且最优。
  - **没有考虑关于目标的信息**
- 启发算法
  - 启发(heuristic)是一个估计当前状态离目标状态还有多“远”的方程
- 贪心搜索
  - 永远扩展看起来最近的（启发告诉我们离目标最近的节点）
  - 不是最优
  - 最差情况: 像一个被错误引导的深搜，最后导致要扩展完整个搜索空间才能到目标
- A*
  - UCS 向后看，根据至今积累的代价排序 g(n)
  - 贪心 向前看, 根据离目标还有多远的估计排序 h(n)
  - A* 根据两者的和决定顺序 : f(n) = g(n) + h(n)
  - 一个启发 ℎ 是 可接受的 (乐观的)，需满足:
    - 启发小于代价，单边的启发值小于实际动作代价
    - 从而，在一条路径下f永远不会减小
  - $$ 0≤h(n)≤h^*(n)$$

$$ g(E) = \frac{8\pi}{3h^3}(2m)^{3/2}E^{1/2} $$

## Chap06 逻辑与CSP

- 变量、域、约束、目标
- 回溯搜索：
  - 先考虑一个变量，然后每一步判断若不符合就回退。
  - 25皇后
- 改进：
  - 顺序：
    - 最少剩余值启发：选择可能性最少的变量赋值
    - 当一个变量被赋值的时候，选择给剩下的变量留下最多选择的值。
  - 筛选：
    - 约束传递: 从约束推理到约束
    - 一条边 X → Y 是一致的：对于每一个X的剩余值，Y都有某个赋值方式，使其不会违反约束
- SAT问题：是否存在一种布尔赋值组合，使所有的逻辑约束都能被满足
  - 将问题变为布尔运算
  - DPLL算法：
    - 合取范式（CNF）： (p1∨¬p3∨p4 ) ∧ (¬p1∨p2∨¬p3 ) ∧ ...
      - 就是每一个都要真
    - 单字符传递（unit propagation）：当某个子句只剩下一个字符，对这个变量赋值使子句为真
    - 布尔约束传递(BCP):重复使用单字符传递，直到无法使用为止
    - （就是深搜……
  - 矛盾指引的子句学习算法:
    - 隐含图（implication graph）：对于SAT的搜索树，我们有两种主要赋值方式，手动赋值，和由BCP产生的赋值。
    - 通过已有的条件推出越来越多的矛盾语句
    - 对于完全可观察的、确定性的情况:
      - 规划问题是可解的 当且仅当 存在可满足的赋值
      - 解法从动作变量的赋值获得