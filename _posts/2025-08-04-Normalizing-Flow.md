---
title: "Normalizing Flow"
mathjax: true
layout: post
categories: Generative Modeling
---
# Normalizing Flow

***

这篇ICLR 2015 workshop论文提出了\*\*NICE (Non-linear Independent Components Estimation)\*\*框架，旨在解决无监督学习中的一个核心问题：**如何捕捉具有未知结构的复杂高维数据分布**。

论文基于一个关键观点：**好的数据表示是使数据分布易于建模的表示**。具体来说，作者寻求找到一个非线性变换，将原始数据映射到潜在空间，使得变换后的数据分布是**可分解的**（即潜在变量相互独立），从而简化概率建模。

## 引入

传统的方法中，往往是对数据求极大似然估计。然而，数据的概率分布往往是很复杂的。比如在VAE中我们不得不求其对数似然的变分下界。然而，变分下界只能让最低点逐渐增加，并不是真正的似然。有没有办法直接求出似然呢？

训练数据的概率过于复杂，这是无法直接求解的原因，因此我们可以想到：是否可以把概率分布转化为一个比较简单的分布（例如高斯分布），如果我们可以找到这个简单分布和数据分布之间的关系，就可以进行概率分布之间的转换。

## 变量替换定理

构造一个函数 $z=f(x)$ ，并且该函数存在反函数 $x=f^{-1}(z)$ 。也就是 $x$ 可以通过某个函数 $f$ 转换成 $z$ （并且维度必须保持不变），则有：

$$
P_{x}(x) = P_{z}(z)\left|\det{\frac{\partial{z}}{\partial{x}}}\right| = P_{z}(f(x))\left|\det{\frac{\partial{f(x)}}{\partial{x}}}\right| = P_{z}(f(x))\left| \det{J_{f}(x)} \right|
$$

其中 $\det$ 表示求矩阵的行列式， $||$ 表示求绝对值， $J_{f}(x)$ 表示雅可比矩阵。

## 归一化流模型

令 $x_{1} = f_{0}(x_{0})$ ，其中 $x_{0}$ 表示原图像，则：

$$
P_{x_{0}}(x_{0}) = P_{x_{1}}(x_{1})\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right|
$$

现在我们对其似然函数运用变量替换定理，可以得到：

$$
\log{P_{x_{0}}(x_{0})} = \log\left( P_{x_{1}}(x_{1})\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right| \right) = \log{P_{x_{1}}(x_{1})}+\log{\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right|}
$$

实际上，我们显然不能一步就从原始图像变成简单的先验分布，这样神经网络很难拟合，因此我们可以让概率分布逐渐简单，例如， $x_{1}$ 只是一个比 $x_{0}$ 简单一点点的概率分布， $x_{2}$ 是一个比 $x_{1}$ 简单一点点的概率分布，我们可以依次进行计算，得到：

$$
\begin{aligned}
\log{P_{x_{0}}(x_{0})} &= \log{P_{x_{1}}(x_{1})}+\log{\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right|}
\\[10pt]
&= \log{\left( P_{x_{2}}(x_{2})\left| \det{\frac{\partial{x_{2}}}{\partial{x_{1}}}} \right| \right)} + \log{\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right|}
\\[10pt]
&= \log{P_{x_{2}}(x_{2})}+\log{\left| \det{\frac{\partial{x_{2}}}{\partial{x_{1}}}} \right|}+\log{\left| \det{\frac{\partial{x_{1}}}{\partial{x_{0}}}} \right|}
\\[10pt]
&= \log{P_{x_{2}}(x_{2})} + \sum_{i=1}^{2}{\log{\left| \det{\frac{\partial{x_{i}}}{\partial{x_{i-1}}}} \right|}}
\\[10pt]
&= \log{P_{x_{M}}(x_{M})} + \sum_{i=1}^{M}{\log{\left| \det{\frac{\partial{x_{i}}}{\partial{x_{i-1}}}} \right|}}
\end{aligned}
$$

因此，我们就得到了优化目标，也就是最大化 $\log{P_{x_{0}}(x_{0})}$ ，而 $P_{x_{M}}(x_{M})$ 是一个已经事先规定好的简单先验分布，因此目标转化为最大化第二项，即

$$
\arg{\max{\sum_{i=1}^{M}{\log{\left| \det{\frac{\partial{x_{i}}}{\partial{x_{i-1}}}} \right|}}}}
$$

现在只有两个问题：

1.  选择一个简单的先验分布。
2.  简化雅可比矩阵行列式的计算。

### 分块耦合层

假设我们需要拟合一个函数 $z=f(x)$ ，其思想是将 $x$ 的维度按照某种比例分成两部分维度。假设 $x$ 的维度是 $D$ 维，那么就变成两部分 $x_{1}\in(1:d),\;x_{2}\in(d+1:D)$ 。现在对其进行函数变换得到 $z_{1},z_{2}$

$$
z_{1} = x_{1}
\\
z_{2} = x_{2}+m(x_{1})
$$

其中， $m(x)$ 是一个神经网络。最后将 $z_{1},z_{2}$ 堆叠起来，形成 $z$ 。我们可以分析其雅可比矩阵：

$$
\begin{bmatrix}
\frac{\partial{z_{1}}}{\partial{x_{1}}}&\frac{\partial{z_{1}}}{\partial{x_{2}}}\\
\frac{\partial{z_{2}}}{\partial{x_{1}}}&\frac{\partial{z_{2}}}{\partial{x_{2}}}
\end{bmatrix}
=
\begin{bmatrix}
1&0\\
\frac{\partial{z_{2}}}{\partial{x_{1}}}&1
\end{bmatrix}
$$

其行列式为1，则 $\log{\left| \frac{\partial{z}}{\partial{x}} \right|} = \log{1} = 0$ 。这大大简化了雅可比行列式的计算。

后面生成数据的时候需要反函数，所以我们需要得到 $x=f^{-1}(z)$ 这个反函数：

$$
x_{1}=z_{1}
\\
x_{2} = z_{2} - m(x_{1}) = z_{2} - m(z_{1})
$$

### 交替耦合

容易看到，我们如果对其进行分块耦合，一部分经过了变化，一部分不经过变化。我们进行了 $M$ 次的变化，如果每次都是 $x_{1}$ 不经过变化，只变化 $x_{2}$ ，这样是不合理的，因此我们会交替进行。假设上面的 $z$ 为第一次变化，则在第二次变化时，我们就用：

$$
z_{2}^{(2)} = z_{2}^{(1)}
\\
z_{1}^{(2)} = z_{1}^{(1)}+m(z_{2})
$$

如此进行交替耦合，就可以更好地拟合原本的函数，确保所有维度相互影响。

### 缩放

经过交替耦合的输出有一个致命的缺陷：其行列式必定为1，这很显然不应该，因此我们需要在经过 $M$ 次的交替分块耦合之后，对最终的输出 $z^{(M)}$ 做一次缩放。即引入一个与 $z$ 相同维度的向量 $s$ 。记最终结果为 $h$ ，则

$$
h=s\cdot z^{(M)}
$$

即对应元素相乘。其反函数为

$$
z^{(M)} = s^{-1}h
$$

同样的，求出其雅可比矩阵（仍然以二维为例）

$$
\begin{bmatrix}
\frac{\partial{h_{1}}}{\partial{z_{1}^{(M)}}}&\frac{\partial{h_{1}}}{\partial{z_{2}^{(M)}}}
\\
\frac{\partial{h_{2}}}{\partial{z_{1}^{(M)}}}&\frac{\partial{h_{2}}}{\partial{z_{2}^{(M)}}}
\end{bmatrix}
=
\begin{bmatrix}
s_{1}&0\\
0&s_{2}
\end{bmatrix}
$$

所以行列式为

$$
\sum_{i=1}^{M}{\log{\left| \det{\frac{\partial{x_{i}}}{\partial{x_{i-1}}}} \right|}} = \sum_{i=1}^{D}{\log{s_{i}}}
$$

### 目标函数——雅可比行列式

假设先验分布的各个维度都相互独立，即

$$
P(h) = \prod_{i=1}^{D}{P(h_{i})}
$$

如果该分布是标准高斯分布，则

$$
\log{P(h)} = \log{\prod_{i=1}^{D}{P(h_{i})}} = \sum_{i=1}^{D}\log{P(h_{i})} = \sum_{i=1}^{D}{\left( \log{\frac{1}{\sqrt{2\pi}}}-\frac{h_{i}^{2}}{2} \right)}
$$

前面那一项 $\log$ 显然不在优化的参数范围之内，因此我们最终的目标函数可以写成：

$$
\log{P_{x_{0}}(x_{0})} = \log{P_{h}(h)}+\sum_{i=1}^{D}{\log{s_{i}}} = \sum_{i=1}^{D}\left( \log{s_{i}}-\frac{h_{i}^{2}}{2} \right)
$$

## 主要贡献

1.  创新的可逆变换架构：

    *   设计了特殊的耦合层结构，使得Jacobian行列式和逆变换易于计算。
    *   通过组合简单构建快实现复杂的非线性变换能力。

2.  精确的似然计算：

    *   与VAE不同，无需变分下界，直接优化精确的对数似然。
    *   避免了GAN中的对抗训练过程

3.  高效的采样与推断：

    *   无偏采样： $h\sim p_{h}(h),x=f^{-1}(h)$

    *   可用于图像修复等任务（通过梯度上升最大化缺失部分的似然）

NICE是\*\*标准化流(normalizing flow)\*\*领域的开创性工作，为后续的RealNVP、Glow等模型奠定了基础。它证明了通过精心设计的可逆变换结构，可以直接优化精确的对数似然，同时保持高效的采样和推断能力，为深度生成模型提供了一条不同于GAN和VAE的重要路径。论文的关键洞见是：**通过结构化设计，可以同时实现"易于计算Jacobian行列式"和"能够学习复杂变换"这两个看似矛盾的目标**。
