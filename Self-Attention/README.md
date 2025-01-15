# 实训项目

基于自注意力机制的情感分析模型

数据集 [Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data)

## 目标

- 掌握注意力机制、自注意力机制、多头注意力机制
- 掌握RNN(LSTM、GRU)、GLU结构
- 完成基于自注意力机制的情感分析模型

## 环境搭建

需要一个 torch+cuda 的运行环境. 推荐使用 conda 来管理环境.

## 项目结构

- `RNN` 下包含一个双向 GRU 结构的情感分析模型
- `Transformer` 下包含一个基于 Transformer 的情感分析模型

`index_view.py` 可以查看模型在该任务上的得分.
