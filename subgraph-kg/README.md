# 人工智能实训 II 项目

下面是一些课程介绍和本次项目开展内容

## 选择任务

基于知识图谱的社交网络用户行为分析

- 掌握知识图谱的构建方法
- 理解实体、关系三元组的意义
- 掌握社交网络数据中数据处理方法；实现数据中的实体提取
- 完成模型的构建对比至少5种传统方法；实现用户数据可视化，形成社交关系拓扑图；分析用户行为规律。
- 选题不需要完全一致，任务数据不需要过于拘泥于要求中列出的。
- 每个人至少独立撰写小组期末报告3页及其以上。最长不超过5页。

### 方向

基于 Subgraph 做 link prediction 的 KG 补全, 同时训练策略模型(policy learning)来选择 subgraph

### 创新点

- Subgraph GNN on KG completion
- Policy learning of subgraph selection on Knowledge Graphs

### 实验

分开做 benchmark

- 不同 GNN 模型 (GCN, GAT, GraphSAGE, VGAE)，语义模型对比（TransE）
- 子图有效性验证（全图/3-hop邻居/子图）, 效率对比（子图规模/推理时间/内存占用）

启发式方法作为 baseline，和 Policy learning 的差异

可开展下面的消融实验

- 不同奖励机制的影响分析
- 子图编码器的层数敏感性测试

选取最佳方案后

- 其它 sota 方案的对比: SEAL

## Main Reference

- Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs (ICLR 2024)
- Efficient Subgraph GNNs by learning effective selection policies (ICLR 2024)
