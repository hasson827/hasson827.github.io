---
title: "Neural Ordinary Differential Equations"
mathjax: true
layout: post
categories: Generative Modeling
---

# Neural Ordinary Differential Equations

***

## 问题背景

这篇论文主要解决传统深度神经网络架构的局限性问题。传统模型(如残差网络、循环神经网络等)通过离散的隐藏层序列构建复杂变换，存在以下问题：

*   内存消耗随网络深度线性增长，成为训练深层模型的主要瓶颈
*   计算量固定，无法根据输入复杂度自适应调整
*   在归一化流等生成模型中存在维度限制

### 创新思路

论文提出了"神经微分方程"(Neural ODEs)这一全新深度学习模型家族，其核心思想是：

1.  **从离散到连续的范式转变**：

    *   传统网络： $h_{t+1} = h_{t}+f(h_{t},\theta_{t})$  (离散序列)

    *   Neural ODEs： $\frac{dh(t)}{dt} = f(h(t),t,\theta)$ (连续动态系统)

    作者将传统神经网络视为微分方程的欧拉离散化，当层数趋于无限、步长趋于零时，网络行为可以用常微分方程(ODE)来描述。

2.  **黑盒微分方程求解器**：

    *   不再定义离散的隐藏层，而是用神经网络参数化隐藏状态的导数
    *   输出通过黑盒ODE求解器计算，无需存储中间层状态

### 主要优势

1.  **恒定内存成本**：通过伴随灵敏度方法(adjoint sensitivity method)实现反向传播，内存消耗与网络"深度"无关
2.  **自适应计算**：利用现代ODE求解器的自适应特性，可以根据问题复杂度动态调整计算量，精度和速度可权衡
3.  **连续归一化流**：提出连续归一化流(CNF)，避免了传统归一化流的维度限制，可以直接通过最大似然训练
4.  **自然处理连续时间数据**：可以无缝处理不规则时间点到达的数据，特别适合时间序列建模

### 应用价值

这种方法为深度学习提供了新的建模范式，特别适用于：

*   需要高精度但内存受限的场景
*   不规则采样的时间序列数据(如医疗记录)
*   需要可逆变换的生成模型
*   需要明确控制计算精度与速度权衡的应用

论文的核心贡献是将ODE求解器作为深度学习模型的可微分组件，并提出了一种高效计算梯度的方法，使得连续深度模型成为可能，为深度学习架构设计开辟了新方向。

## ResNet和欧拉法之间的关系

**欧拉法**：假设有一个未知曲线 $h(t)$ ，和一个给定的微分方程，且能通过这个微分方程算出曲线任意一点的导数 $f(t,h(t))$ 。那么给定 $t_{0}$ 和 $h(t_{0})$ ，我们就可以用欧拉法求解 $t_{1}$ 和 $h(t_{1})$ 的值。

$$
h(t_{1}) = h(t_{0})+f(t_{0},h(t_{0}))\cdot(t_{1}-t_{0})
$$

如果 $t_{1}\to t_{0}$ ，那么有：

$$
h(t_{1}) = h(t_{0})+f(t_{0},h(t_{0}))dt
$$

对于一个残差神经网络，我们有：

$$
\begin{aligned}
h_{t+1} &= \text{ReLU}(W_{t},h_{t}+b_{t})+h_{t}
\\[10pt]
&=f(h_{t},\theta_{t})+h_{t}
\end{aligned}
$$

ResNet当中的一个残差快可以看成欧拉法中 $dt=1$ 的特殊情况。因此我们只需要对这个“导数”进行建模就可以了。

## 用欧拉法建模的Neural ODE的形式

建模公式如下：

$$
\frac{dh(t)}{dt} = f(h(t),t,\theta)
$$

我们可以写成：

$$
dh(t) = f(h(t),t,\theta)dt
$$

两边取积分：

$$
\int_{t_{0}}^{t_{1}}{dh(t)} = \int_{t_{0}}^{t_{1}}{f(h(t),t,\theta)dt}
$$

这就是欧拉法要求解的微分方程的形式， $t_{1}$ 可以取任何值，从 $t_{0}$ 到 $t_{1}$ 的过程是连续的。

由于Neural ODE的根本目的就是求解微分方程，因此我们可以给出其损失函数：

$$
\begin{aligned}
\mathcal{L}(h(t_{1})) &= \mathcal{L}(\text{ODESolve}(h(t_{0}),f,t_{0},t_{1},\theta))
\\[10pt]
&= \mathcal{L}\left( h(t_{0})+\int_{t_{0}}^{t_{1}}{f(h(t),t,\theta)dt} \right)
\end{aligned}
$$

## 伴随灵敏度法(Adjoint Sensitivity Method)

理论上，我们可以根据前向传播的方向，直接微分反向传播就可以更新参数。然而，这种方式会导致高内存成本和额外的数值误差，因此作者采用**伴随灵敏度法**来计算梯度，具体如下：

假设通过前向传播得到了一个ODE的解为 $h(t)$ ，我们定义一个伴随状态(Adjoint States)

$$
\begin{equation}
a(t) = \frac{d\mathcal{L}}{dh(t)}\label{eq1}
\end{equation}
$$

那么通过链式法则我们可以得到：

$$
a(t) = \frac{d\mathcal{L}}{dh(t)} = \frac{d\mathcal{L}}{dh(t+\delta)}\frac{dh(t+\delta)}{dh(t)} = a(t+\delta)\frac{\partial h(t+\delta)}{\partial h(t)}
$$

对于ODE的前馈，我们有：

$$
h(t+\delta) = h(t)+\int_{t}^{t+\delta}{f(h(t'),t',\theta)dt'}
$$

因此我们将 $h(t+\delta)$ 带入 $a(t)$ 的表达式中，可以得到：

$$
a(t) = a(t+\delta)\frac{\partial h(t+\delta)}{\partial h(t)} = a(t+\delta)\frac{\partial}{\partial h(t)}\left( h(t)+ \int_{t}^{t+\delta}{f(h(t'),t',\theta)dt'} \right)
$$

将导数项中的 $h(t)$ 提取出来，可以得到：

$$
a(t) = a(t+\delta)+a(t+\delta)\frac{\partial}{\partial h(t)}\left( \int_{t}^{t+\delta}{f(h(t'),t',\theta)dt'} \right)
$$

因此我们可以求得 $a(t)$ 的导数：

$$
\begin{equation}
\begin{aligned}
\frac{da(t)}{dt} &= \lim_{\delta\to0^{+}}\frac{a(t+\delta)-a(t)}{\delta}
\\[10pt]
&= \lim_{\delta\to0^{+}}\frac{-a(t+\delta)\frac{\partial}{\partial h(t)}\left( \int_{t}^{t+\delta}{f(h(t'),t',\theta)dt'} \right)}{\delta}
\\[10pt]
&= -a(t)\frac{\partial{f(h(t),t,\theta)}}{\partial{h(t)}}
\end{aligned}\label{eq2}
\end{equation}
$$

现在总结一下，我们已经有了伴随状态的微分方程，即式 $\eqref{eq2}$ 。如果我们已知 $t_{1}$ 时刻的损失函数，由式 $\eqref{eq1}$ 可以得到一个伴随状态。我们也可以再次使用欧拉法推导出 $h$ 在 $t_{0}$ 时刻的伴随状态，即：

$$
\begin{aligned}
a(t_{0}) &= a(t_{1})+\int_{t_{1}}^{t_{0}}{\frac{da(t)}{dt}dt}
\\[10pt]
&= a(t_{1}) - \int_{t_{1}}^{t_{0}}{a(t)\frac{\partial{f(h(t),t,\theta)}}{\partial{h(t)}}dt}
\end{aligned}
$$

我们不仅可以推导出损失函数关于 $h(t)$ 的梯度，也可以推导出损失函数关于 $t$ 和 $\theta(t)$ 的梯度。

假设 $\theta,t$ 都是状态，并且拥有恒定的微分方程表达式

$$
\frac{\partial{\theta(t)}}{\partial{t}} = 0\qquad \frac{\partial{t(t)}}{\partial{t}} = 1
$$

把 $h,\theta,t$ 结合起来，形成一个增广的ODE，即

$$
\frac{d}{dt}\begin{bmatrix}
h\\\theta\\t
\end{bmatrix}(t)
 = f_{aug}(\left[ h,\theta,t \right]):=
 \begin{bmatrix}
 f[h(t),\theta,t]\\0\\1
 \end{bmatrix}
 \\[10pt]
 a_{aug}:=\begin{bmatrix}
 a\\a_{\theta}\\a_{t}
 \end{bmatrix},\;
 a_{\theta}(t) = \frac{d\mathcal{L}}{d\theta(t)};\;a_{t}(t)=\frac{d\mathcal{L}}{dt(t)}
$$

那么，如果对于这个增广的ODE关于 $h,\theta,t$ 分别求偏导，则

$$
\frac{\partial{f_{aug}}}{\partial[h,\theta,t]} = \begin{bmatrix}
\frac{\partial{f_{aug}^{1}}}{\partial{h}}&\frac{\partial{f_{aug}^{1}}}{\partial{\theta}}&\frac{\partial{f_{aug}^{1}}}{\partial{t}}\\
\frac{\partial{f_{aug}^{2}}}{\partial{h}}&\frac{\partial{f_{aug}^{2}}}{\partial{\theta}}&\frac{\partial{f_{aug}^{2}}}{\partial{t}}\\
\frac{\partial{f_{aug}^{3}}}{\partial{h}}&\frac{\partial{f_{aug}^{3}}}{\partial{\theta}}&\frac{\partial{f_{aug}^{3}}}{\partial{t}}
\end{bmatrix}
=\begin{bmatrix}
\frac{\partial{f}}{\partial{h}}&\frac{\partial{f}}{\partial{\theta}}&\frac{\partial{f}}{\partial{t}}\\
0&0&0\\0&0&0
\end{bmatrix}
$$

其中，0是一个具有适当尺寸的零矩阵。

所以依据 $\frac{da(t)}{dt}$ 的表达式，我们可以得到其增广形式的表达式：

$$
\frac{da_{aug}(t)}{dt} = -\begin{bmatrix}
a(t)&a_{\theta}(t)&a_{t}(t)
\end{bmatrix}\frac{\partial{f_{aug}}}{\partial{\left[ h, \theta, t \right]}}(t) = -\begin{bmatrix}
a(t)\frac{\partial{f}}{\partial{h}}&a(t)\frac{\partial{f}}{\partial{\theta}}&a(t)\frac{\partial{f}}{\partial{t}}
\end{bmatrix}(t)
$$

所以有：

$$
\frac{da_{\theta}(t)}{dt} = -a(t)\frac{\partial{f}}{\partial{\theta}}
\\[10pt]
\frac{da_{t}(t)}{dt} = -a(t)\frac{\partial{f}}{\partial{t}}
$$

设定 $a_{\theta}(t_{N}) = 0$ ，则有：

$$
\frac{d\mathcal{L}}{d\theta} = a_{\theta}(t_{0}) = a_{\theta}(t_{N})-\int_{t_{N}}^{t_{0}}{a(t)\frac{\partial{f(h(t),\theta,t)}}{\partial{\theta}}dt} = -\int_{t_{N}}^{t_{0}}{a(t)\frac{\partial{f(h(t),\theta,t)}}{\partial{\theta}}dt}
$$

实际应用的时候，我们通过ODE数值求解，求解出 $a_{\theta}(t_{0})$

$$
a_{\theta}(t_{N-1}) = a_{\theta}(t_{N})+\epsilon\left( -a(t_{N})\frac{\partial{f}}{\partial{\theta}} \right)
\\[10pt]
a_{\theta}(t_{N-2}) = a_{\theta}(t_{N-1})+\epsilon\left( -a(t_{N-1})\frac{\partial{f}}{\partial{\theta}} \right)
\\\vdots\\
a_{\theta}(t_{0}) = a_{\theta}(t_{1})+\epsilon\left( -a(t_{1})\frac{\partial{f}}{\partial{\theta}} \right)
$$

同样可以得到

$$
a_{t}(t) = \frac{\partial{\mathcal{L}}}{\partial{t}} = \frac{\partial{\mathcal{L}}}{\partial{h(t)}}\frac{\partial{h(t)}}{\partial{t}} = a(t) \frac{\partial{h(t)}}{\partial{t}} = a(t)\cdot f(h(t),t,\theta)
\\[10pt]
\frac{d\mathcal{L}}{dt_{0}} = a_{t}(t_{0}) = a_{t}(t_{N}) - \int_{t_{N}}^{t_{0}}{a(t)\frac{\partial{f(h(t),t,\theta)}}{\partial{t}}dt}
$$

实际应用的时候，依然是通过数值求解器求解

$$
a_{t}(t_{N-1}) = a_{t}(t_{N})+\epsilon\left( -a(t_{N})\frac{\partial{f}}{\partial{t}} \right)
\\[10pt]
a_{t}(t_{N-2}) = a_{t}(t_{N-1})+\epsilon\left( -a(t_{N-1})\frac{\partial{f}}{\partial{t}} \right)
\\\vdots\\
a_{t}(t_{0}) = a_{t}(t_{1})+\epsilon\left( -a(t_{1})\frac{\partial{f}}{\partial{t}} \right)
$$

**这个方法为什么重要**：

*   传统方法反向传播通过ODE求解器需要存储所有中间状态（内存 $\mathcal{O}(L)$ ）

*   伴随方法允许通过求解一个增广ODE来计算梯度（内存 $\mathcal{O}(1)$ ）

*   这是实现"恒定内存成本"的关键技术

*   使深层连续模型的训练在内存上变得可行

## 连续归一化流

提出了"连续归一化流"(Continuous Normalizing Flows)的概念，声称它能避免传统归一化流的维度限制，并可直接通过最大似然训练。它和Normalizing Flow之间有很密切的关系，具体如下：

1.  **共同的理论基础**：两者都基于**变量变换定理**(change of variables theorem)来建模概率分布：

    $$
    z_{1} = f(z_{0})\Rightarrow\log{p(z_{1})} = \log{p(z_{0})}-\log{\left| \text{det}\frac{\partial{f}}{\partial{z_{0}}} \right|}
    $$

    这是归一化流方法的核心数学原理。

2.  **相同的建模目标**：两者都旨在通过可逆变换将简单分布(如高斯分布)转换为复杂的目标分布。

3.  **离散到连续的演变**：传统归一化流的离散方程也出现在归一化流框架中。CNF实际上是将离散变换扩展为连续变换。

### Normalizing Flow To Continuous

在之前讲过的归一化流中，曾经提到一个**变量替换定理**，假设存在 $z_{1}=f(z_{0})$ ，函数 $f$ 存在反函数，则

$$
\log{p(z_{1})} = \log{p(z_{0})} - \log\left| \text{det}{\frac{\partial{f}}{\partial{z_{0}}}} \right|
$$

如果我们把这个式子加入一个步长 $h=1$ ，那么就和前面的欧拉法基本一样了：

$$
\begin{aligned}
\log{p(z_{1})} - \log{p(z_{0})} &= - \log\left| \text{det}{\frac{\partial{f}}{\partial{z_{0}}}} \right|
\\[10pt]
\frac{\log{p(z_{1})} - \log{p(z_{0})}}{\Delta t}\Bigg|_{\Delta t=1} &= - \log\left| \text{det}{\frac{\partial{f}}{\partial{z_{0}}}} \right| 
\end{aligned}
$$

为了表达的一致性，我们大可以将 $\log{p(z_{1})} - \log{p(z_{0})}$ 也称为一般化的形式，即 $\log{p(z_{t+1})}-\log{p(z_{t})}$ ，所以

$$
\frac{\partial{\log{p(z(t))}}}{\partial{t}} = - \log\left| \text{det}{\frac{\partial{f}}{\partial{z(t)}}} \right|
$$

变换函数 $f$ 也表示为一个微分方程

$$
\frac{dz}{dt}=f(z(t),t)
$$

这样的话，我们就得到了连续归一化流的表达形式。然而，这种形式是存在一些困难的，最明显的，就是里面的雅可比行列式的计算，它必须要是相对简单的，否则计算量将非常之大。

### Continuous Normalizing Flow

然而，当归一化流从离散层面过渡到连续层面之后，竟然简化了运算，我们可以得到以下的式子：

$$
\frac{\partial{\log{p(z(t))}}}{\partial{t}} = -\tr{\left( \frac{df}{dz(t)} \right)}
$$

此时，我们不难发现，这个公式，竟然不再需要求行列式，而是求雅可比矩阵的迹，我们知道，矩阵的迹就是对角线上的值求和，这是非常容易计算的。除此之外，微分方程 $f$ 不再需要是双射的，因为一旦满足唯一性条件，那么整个变换自动成为双射（双射意味着存在反函数）。

**这个公式为什么重要**：

*   传统归一化流需要计算雅可比矩阵的行列式（计算复杂度 $\mathcal{O}(D^{3})$ ）

*   这个公式表明在连续变换中，只需计算迹(trace)操作（计算复杂度 $\mathcal{O}(D)$ ）

*   使得"连续归一化流"(CNF)成为可能，解决了传统归一化流的维度限制问题

*   可以使得CNF轻松使用宽流模型，一次使用多个隐藏单元

简单说，这个公式展示了为什么连续变换能简化概率密度变换。

#### 推导Continuous Normalizing Flow公式

首先，令

$$
z(t+\epsilon) = z(t)+\epsilon\cdot f(z(t),t,\theta)+\dots=T_{\epsilon}(z(t))
$$

我们假设 $f$ 在 $z(t)$ 上是 $Lipschitz$ 连续的，并且在 $t$ 上是连续的，因此根据Picard存在定理，每个初值问题都有唯一解。我们还假设 $z(t)$ 是有界的。这些条件意味着 $f,T_{\epsilon},\frac{\partial{T_{\epsilon}}}{\partial{z}}$ 都是有界的。那么我们可以得到

$$
\begin{aligned}
\frac{\partial{\log{p(z(t))}}}{\partial{t}} &= \lim_{\epsilon\to0^{+}}{\frac{\log{p(z(t+\epsilon))} - \log{p(z(t))}}{\epsilon}}
\\[10pt]
&= \lim_{\epsilon\to0^{+}}{\frac{\log{p(z(t))} - \log\left| \text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right| - \log{p(z(t))}}{\epsilon}}
\\[10pt]
&= -\lim_{\epsilon\to0^{+}}\frac{\log\left| \text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right|}{\epsilon}
\\[10pt]
&= -\lim_{\epsilon\to0^{+}}\frac{\frac{\partial}{\partial{\epsilon}} \log\left|\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right|}{\frac{\partial}{\partial{\epsilon}}\epsilon}\qquad\text{(By L'Hopitals Rule)}
\\[10pt]
&= -\lim_{\epsilon\to0^{+}}\frac{\frac{\partial}{\partial{\epsilon}} \left|\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right|}{\left|\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right|}\qquad\left( \frac{\partial{\log{(z)}}}{\partial{z}}\Bigg|_{z=1}=1 \right)
\\[10pt]
&= -\lim_{\epsilon\to0^{+}}\frac{\partial}{\partial{\epsilon}} \left|\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right|
\end{aligned}
$$

我们可以利用一个公式去进行转换：

$$
\frac{d\text{det}{A}}{dt} = \text{det}{A}\tr\left( A^{-1}\frac{dA}{dt} \right)
$$

又因为对于伴随矩阵，有

$$
A^{-1} = \frac{1}{\text{det}{A}}\text{adj}(A)
$$

所以

$$
\frac{d\text{det}{A}}{dt} = \text{det}{A}\tr\left( \frac{1}{\text{det}{A}}\text{adj}(A) \frac{dA}{dt} \right)
$$

我们将其带入回原式：

$$
\frac{\partial}{\partial{\epsilon}} \left(\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))} \right) = \text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))}\tr{\left( \frac{1}{\text{det}{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))}} \text{adj}\left( \frac{\partial}{\partial{z}}T_{\epsilon}(z(t)) \right) \frac{\partial}{\partial{\epsilon}}\frac{\partial}{\partial{z}}{T_{\epsilon}(z(t))} \right)}
$$

当 $\epsilon\to0^{+}$ 时， $\det{\frac{\partial}{\partial{z}}T_{\epsilon}(z(t))}=1$ ，所以

$$
\begin{aligned}
\frac{\partial{\log{p(z(t))}}}{\partial{t}} &= -\lim_{\epsilon\to0^{+}}\tr\left( \text{adj}\left( \frac{\partial}{\partial{z}}T_{\epsilon}(z(t)) \right) \frac{\partial}{\partial{\epsilon}}\frac{\partial}{\partial{z}}{T_{\epsilon}(z(t))} \right)
\\[10pt]
&= -\tr{\left( \underbrace{\left( \lim_{\epsilon\to0^{+}}{\text{adj}\left( \frac{\partial}{\partial{z}}T_{\epsilon}(z(t)) \right)} \right)}_{=I}\left( \lim_{\epsilon\to0^{+}} \frac{\partial}{\partial{\epsilon}}\frac{\partial}{\partial{z}}{T_{\epsilon}(z(t))} \right) \right)}
\\[10pt]
&= -\tr{\left( \lim_{\epsilon\to0^{+}} \frac{\partial}{\partial{\epsilon}}\frac{\partial}{\partial{z}}{T_{\epsilon}(z(t))} \right)}
\end{aligned}
$$

将 $T_{\epsilon}$ 进行泰勒展开即可得到结果

$$
\begin{aligned}
\frac{\partial{\log{p(z(t))}}}{\partial{t}} &= -\tr{\left( \lim_{\epsilon\to0^{+}} \frac{\partial}{\partial{\epsilon}}\frac{\partial}{\partial{z}}{\left( z+\epsilon f(z(t),t)+\mathcal{O}(\epsilon^{2})+\mathcal{O}(\epsilon^{3})+\cdots \right)} \right)}
\\[10pt]
&= -\tr{\left( \lim_{\epsilon\to0^{+}} \frac{\partial}{\partial{\epsilon}}\left( I+\frac{\partial}{\partial{z}}\epsilon f(z(t),t)+\mathcal{O}(\epsilon^{2})+\mathcal{O}(\epsilon^{3})+\cdots \right) \right)}
\\[10pt]
&= -\tr{\left( \lim_{\epsilon\to0^{+}}{\left( \frac{\partial}{\partial{z}} f(z(t),t)+\mathcal{O}(\epsilon)+\mathcal{O}(\epsilon^{2})+\cdots \right)} \right)}
\\[10pt]
&= -\tr{\left( \frac{\partial{f(z(t),t)}}{\partial{z}} \right)}
\end{aligned}
$$

## 具体实现步骤

### 定义连续变换的ODE

首先定义一个由神经网络参数化的常微分方程：

$$
\frac{dz(t)}{dt} = f(z(t),t,\theta)
$$

其中：

*   $z(t)$ 是在时间 $t$ 的隐藏状态

*   $f$ 是一个神经网络（通常为多层感知机）

*   $\theta$ 是神经网络的参数

**实现细节**：

*   $f$ 的输入是 $[z(t),t]$ 的拼接向量

*   网络结构通常为2～3层MLP，每层50～100个隐藏单元

*   激活函数常用tanh或ReLU

### 密度变换的计算

给定初始分布 $p(z(0))$ ，目标分布 $p(z(T))$ 的对数密度为：

$$
\log{p(z(T))} = \log{p(z(0))}-\int_{0}^{T}{\tr\left( \frac{df(\tau)}{dz} \right)d\tau}
$$

**实现关键**：将密度问题转化为一个增广ODE系统：

$$
\frac{dz(t)}{dt} = f(z(t),t,\theta)
\\[10pt]
\frac{d}{dt}\log{p(z(t))} = -\tr{\left( \frac{df}{dz(t)} \right)}
$$

这两个方程可以同时求解，通过ODE求解器一步完成。

### 多隐藏单元的实现（宽流模型）

可以使用宽流模型提高表达能力：

$$
\frac{dz(t)}{dt} = \sum_{n=1}^{M}{f_{n}(z(t))}
$$

对应的密度变换：

$$
\frac{d}{dt}\log{p(z(t))} = -\sum_{n=1}^{M}{\tr{\left( \frac{\partial{f_{n}}}{\partial{z}} \right)}}
$$

**优势**：

*   传统Normalizing Flow：计算 $\text{det}\sum{J_{n}}$ ，复杂度为 $\mathcal{O}(M^{3})$

*   Continuous Normalizing Flow：计算 $\sum{\tr(J_{n})}$ ，复杂度为 $\mathcal{O}(M)$

*   这使得CNF可以轻松使用64个隐藏单元，而传统NF受限于单隐藏单元层

### 时变动力学

论文引入了两种时间依赖机制：

1.  **参数作为时间函数**：

    $$
    f(z(t),t,\theta(t))
    $$

    其中 $\theta(t)$ 是 $t$ 的函数

2.  **门控机制**：

    $$
    \frac{dz(t)}{dt} = \sum_{n=1}^{M}{\sigma_{n}(t)\cdot f_{n}(z)}
    $$



$\sigma_{n}(t)\in(0,1)$ 是学习得到的门控信号，控制时间 $t$ 对总体的贡献程度。
