# ANN(Artificial Neural Network)

人工神经网络

实训任务 I

- 掌握感知机、BP神经网络
- 掌握均方误差损失函数
- 完成基于logistic/softmax regression的分类模型

## 使用

1. 环境配置: 一个 torch + cuda 环境
2. sklearn 库来运行感知机的代码
3. 准备好 MNIST 数据集 `data/raw/MNIST`. 如果你没有, 也会自动下载.
4. 运行

```py
python test_FNN.py
```

## 从感知机出发

### Preceptron

感知机是一种经典的**线性分类模型**:

<div style="text-align: center;"><img src="images/linear-model.png" alt="线性二分模型" style = "width: 60%;"></div>

$$f(\mathbb{x}) = h(\mathbb{w}^T \mathbb{x}+b)$$

- $\mathbb{w}$: 权重向量
- $\mathbb{x}$: 输入向量
- $b$: 偏置项
- $h(\cdot)$: 激活函数

对于感知机, 它采用的激活函数是符号函数:

$$y= \mathrm{sgn} (\mathbb{w}^T \mathbb{x}+b)$$

- 即, 当 $\mathbb{w}^T \mathbb{x}+b$ 是正数时, 输出 $y=1$; 当为 0 时, 输出 $y=0$; 如果是负数的话则输出 $y=-1$.
- 在一些文献里, 截距项可能不作为学习参数

它对应特征空间中的分离超平面:

$$\mathbb{w}^T \mathbb{x}+b=0$$

### 学习策略

感知器的学习算法是一种错误驱动的在线学习算法. 损失函数定义为

$$
\min _{w, b} L(w, b)=-\sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)
$$

即, 最小化误分类点到超平面的总距离.

算法采用**随机梯度下降**(Stochastic Gradient Descent, SGD): 沿着损失函数梯度下降的方向更新参数, 随机抽取一个误分类点使其梯度下降.

$$w \leftarrow w + \eta y_{i}x_{i} \\ b \leftarrow b + \eta y_{i}$$

- $\eta$ 是学习率, 控制步长.

当实例点被误分类, 即位于分离超平面的错误侧, 则调整 $w$, $b$ 的值, 使分离超平面向该无分类点的一侧移动, 直至误分类点被正确分类.

## 前馈神经网络

最简单的 ANN 就是感知机, MP 神经元.

多添加几个神经元, 构成一个网络, 根据连接方式的不同区分为三类网络:

- 如果整个网络中的信息是朝一个方向传播, 构成一个有向无环图, 那么就是**前馈神经网络**(Feedforward Neural Network, FNN)
- 此外我们还有记忆网络(RNN), 以及图网络(GNN).

FNN 也被人称为**多层感知器**(MLP)[^1].

<div style="text-align: center;"><img src="images/network-structure.png" alt="网络结构" style = "width: 60%;"></div>

- 在前馈神经网络中, 各神经元分别属于不同的层.
- 每一层的神经元可以接收前一层神经元的信号, 并产生信号输出到下一层.
- 第 0 层称为输入层, 最后一层称为输出层, 其他中间层称为隐藏层. 整个网络中无反馈, 信号从输入层向输出层单向传播, 可用一个有向无环图表示.

<div style="text-align: center;"><img src="images/FNN.png" alt="FNN" style = "width: 60%;"></div>

- 每个神经层看作一个仿射变换 + 非线性变换

$$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$$

$$a^{(l)}=f_l(z^{(l)})$$

前馈神经网络的结构很简单, 但非常实用. 在 Transformer 结构里也能看到它的身影.

### 参数学习

对于回归任务: 均方误差损失函数(Mean Squared Error, MSE)

$$\mathcal{L}(w, b) = \frac{1}{N}\sum_{i=1}^{N}\left(y_{i}-\hat{y}_i\right)^{2}$$

也有 MAE(Mean Absolute Error):

$$\mathcal{L}(w, b) = \frac{1}{N}\sum_{i=1}^{N}\left|y_{i}-\hat{y}_i\right|$$

均方误差损失函数在分类问题中, 对模型输出概率的微小变化相对不敏感.

对于分类任务: 交叉熵损失函数(Cross-Entropy)

$$\mathcal{L}(w, b)=-\sum_{i=1}^{C}y_i \log p_i$$

- $C$: 类别数
- $y_i$: 第 $i$ 个样本的真实类别
- $p_i$: 第 $i$ 个样本的预测概率

优化目标: 最小化训练集 $\mathcal{D}$ 的结构化风险

$$\mathcal{R}(W, b) = \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(y^{(i)}, \hat{y}^{(i)}) + \frac{1}{2}\lambda \Vert W \Vert_F^2$$

- 这里引入了正则化项, 使用 Frobenius 范数.

参数更新, 使用梯度下降法:

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{R} }{\partial W^{(l)} } \\ b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{R} }{\partial b^{(l)} }$$

## 反向传播算法

我们常说的 BP 神经网络实际上是使用 BP 算法训练的 FNN. BP 算法是一种高效地计算梯度的方法.

在优化的过程中, 我们希望令目标函数逐步变小. 如果采用梯度下降的方法, 我们会计算

$$ \frac{\partial \mathcal{L}}{\partial W}, \frac{\partial \mathcal{L}}{\partial b} $$

要逐一计算 $\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}$, 会涉及到多层的复合函数求导, 复杂度很高.

BP 算法利用了链式法则, 简化了运算过程:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}=\frac{\partial z^{(l)}}{\partial w_{ij}^{(l)}}\frac{\partial \mathcal{L}}{\partial z^{(l)}} \\
\frac{\partial \mathcal{L}}{\partial b^{(l)}}=\frac{\partial z^{(l)}}{\partial b^{(l)}}\frac{\partial \mathcal{L}}{\partial z^{(l)}}
$$

只需计算三个偏导数, 其中的 $\frac{\partial \mathcal{L}}{\partial z^{(l)}}$ 记为 $\delta^{(l)}$, 有迭代公式

$$\delta^{(l)}=f_l'(z^{(l)})\odot (W^{(l+1)})^T \delta^{(l+1)}$$

- $f_l'$ 是 $l$ 层的激活函数的导数
- $\odot$ 是向量的 Hadamard 积, 表示每个元素相乘 $(s \odot t)_j = s_j t_j$

于是梯度更新公式为

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}=\delta^{(l)}(a^{(l-1)})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}}=\delta^{(l)}$$

torch API 中提供了 BP 算法的实现. 自动求导(AD)引擎 Autograd 记录了所有的前向操作, 并在反向传播时自动计算梯度(反向累积). 具体接口的使用可参考 `utils.py`.

# References

- http://neuralnetworksanddeeplearning.com/chap2.html

[^1]: 但注意 Preceptron 的激活函数和 FNN 的激活函数不同, 这个说法并不是十分准确.
