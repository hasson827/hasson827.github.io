---
title: "Score-Based Generative Modeling throught SDE"
mathjax: true
layout: post
categories: Generative Modeling
---
# Score-Based Generative Modeling throught SDE

***

从数据中创建噪声很容易；从噪声中创建数据是生成式建模。我们提出一个随机微分方程（SDE），通过缓慢注入噪声，将复杂的数据分布平滑地转换为已知的先验分布，以及一个相应的反向时间 SDE，通过缓慢去除噪声将先验分布转换回数据分布。关键在于，反向时间 SDE 仅依赖于扰动数据分布的时间依赖梯度场（简称分数）。通过利用基于分数的生成式建模的进展，我们可以使用神经网络准确地估计这些分数，并使用数值 SDE 求解器生成样本。

## Introduction

两种成功的概率生成模型包括依次用逐渐增加的噪声干扰训练数据，然后学习逆转这种干扰以形成数据生成模型。基于朗之万动力学的分数匹配(SMLD)在每个噪声尺度上估计分数（即对数概率密度关于数据的梯度），然后在生成过程中使用朗之万动力学从一系列递减的噪声尺度中进行采样。去噪扩散概率建模(DDPM)训练一系列概率模型以逆转噪声干扰的每一步，利用逆转分布的函数形式知识使训练变得可行。对于连续状态空间，DDPM 训练目标隐式地计算每个噪声尺度的分数。因此，我们将这两种模型类别统称为基于分数的生成模型。为了实现新的采样方法并进一步扩展基于分数的生成模型的能力，我们提出了一种统一框架，该框架通过随机微分方程（SDEs）的视角推广了先前的方法。

具体来说，我们不是用有限数量的噪声分布扰动数据，而是考虑一个随时间根据扩散过程演变的分布连续体。该过程逐步将数据点扩散成随机噪声，并由一个不依赖于数据且没有可训练参数的指定随机微分方程（SDE）给出。通过逆转这一过程，我们可以将随机噪声平滑地塑造成数据以进行样本生成。关键的是，这一逆转过程满足一个逆时间 SDE，该逆时间 SDE 可以根据边际概率密度的分数作为时间的函数从正向 SDE 推导出来。因此，我们可以通过训练一个时间依赖的神经网络来估计分数，从而近似逆时间 SDE，然后使用数值 SDE 求解器生成样本。我们的核心思想如下图所示：

![\<img alt="Refer to caption" data-attachment-key="35ZR27NA" width="389" height="175" src="attachments/35ZR27NA.png" ztype="zimage"> | 389](attachments/35ZR27NA.png)

我们提出的框架有几个理论和实践贡献：

**灵活的采样和似然计算**：我们可以使用任何通用随机微分方程（SDE）求解器来积分反向时间 SDE 进行采样。此外，我们提出了两种对一般 SDE 不适用的高级方法：

1.  预测-校正（PC）采样器，将数值 SDE 求解器与基于分数的 MCMC 方法相结合
2.  基于概率流常微分方程（ODE）的确定性采样器

前者统一并改进了现有基于分数模型的采样方法。后者允许通过黑盒 ODE 求解器进行快速自适应采样，通过潜在代码进行灵活的数据操作，具有唯一可识别的编码，并且值得注意的是，可以进行精确的似然计算。

**可控生成**：我们可以通过在训练期间不可用的信息对生成过程进行调节，因为条件反向时间 SDE 可以从无条件分数中高效估计。这使应用成为可能，例如类条件生成、图像修复、上色和其他逆问题，所有这些都可以使用单个无条件基于分数的模型实现，而无需重新训练。

**统一框架**：我们的框架提供了一种统一的方法来探索和调整各种 SDEs，以改进基于分数的生成模型。SMLD 和 DDPM 的方法可以整合到我们的框架中，作为两个独立 SDEs 的离散化。

## Background

### Denoising Score Matching with Langevin Dynamics (SMLD)

令$p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}):=\mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x},\sigma^{2}I)$为扰动核，$p_{\sigma}(\tilde{\mathbf{x}}) = \int{p_{\text{data}}(\mathbf{x})p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})d\mathbf{x}}$，其中$p_{\text{data}}(\mathbf{x})$表示数据分布。考虑一系列正噪声尺度$\sigma_{\min} = \sigma_{1}\lt\sigma_{2}\lt\dots\lt\sigma_{N}=\sigma_{\max}$。通常情况下，$\sigma_{\min}$足够小，使得$p_{\sigma_{\min}}(\mathbf{x})\approx p_{\text{data}}(\mathbf{x})$，而$\sigma_{\max}$足够大，使得$p_{\sigma_{\max}}(\mathbf{x})\approx\mathcal{N}(\mathbf{x};0,\sigma_{\max}^{2}I)$。去噪分数匹配的目标函数的加权组合，就是噪声条件分数网络，即Noise Conditional Score Network(NCSN)，记为$\boldsymbol{s_{\theta}}(\mathbf{x},\sigma)$：

$$
\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}}{\sum_{i=1}^{N}{ \sigma_{i}^{2}\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\mathbb{E}_{p_{\sigma_{i}}(\tilde{\mathbf{x}}|\mathbf{x})}\left[ \left\| \boldsymbol{s_{\theta}}(\tilde{\mathbf{x}},\sigma_{i})-\nabla_{\tilde{\mathbf{x}}}\log{p_{\sigma_{i}}(\tilde{\mathbf{x}}|\mathbf{x})} \right\|_{2}^{2} \right]}}
$$

给定足够的数据和模型参数量，最优的基于分数的模型$\boldsymbol{s_{\theta^{*}}}(\mathbf{x},\sigma)$在$\sigma\in\left\{\sigma_{i}\right\}_{i=1}^{N}$上几乎处处等于$\nabla_{\mathbf{x}}\log{p_{\sigma}(\mathbf{x})}$。对于采样，可以运行$M$步的朗之万MCMC来为每个$p_{\sigma_{i}}(\mathbf{x})$依次得到一个样本：

$$
\mathbf{x}_{i}^{m} = \mathbf{x}_{i}^{m-1}+\epsilon_{i}\boldsymbol{s_{\theta^{*}}}(\mathbf{x}_{i}^{m-1},\sigma_{i})+\sqrt{2\epsilon_{i}}\mathbf{z}_{i}^{m},\quad m=1,2,\dots,M
$$

其中，$\epsilon_{i}\gt0$是步长，$\mathbf{z}_{i}^{m}$是标准正态分布。当$i\lt N$的时候，上述过程依次对$i=N,N-1,\dots,1$、$\mathbf{x}_{N}^{0}\sim\mathcal{N}(\mathbf{x}|0,\sigma_{\max}^{2}I)$和$\mathbf{x}_{i}^{0} = \mathbf{x}_{i+1}^{M}$进行重复。对于所有$i$，当$M\to\infin,\epsilon_{i}\to0$的时候，$\mathbf{x}_{1}^{M}$成为$p_{\sigma_{\min}}(\mathbf{x})\approx p_{\text{data}}(\mathbf{x})$的精确样本。

### 去噪扩散概率模型 (DDPM)

考虑一系列正噪声尺度$0\lt\beta_{1},\beta_{2},\dots,\beta_{N}\lt1$。对于每个训练数据点$\mathbf{x}_{0}\sim p_{\text{data}}(\mathbf{x})$，构建一个离散马尔可夫链$\left\{ \mathbf{x}_{0},\mathbf{x}_{1},\dots,\mathbf{x}_{N} \right\}$，使得$p(\mathbf{x}_{i}|\mathbf{x}_{i-1})=\mathcal{N}(\mathbf{x}_{i};\sqrt{1-\beta_{i}}\mathbf{x}_{i-1},\beta_{i})$，因此

$$
p_{\alpha_{i}}(\mathbf{x}_{i}|\mathbf{x}_{0}) = \mathcal{N}(\mathbf{x}_{i};\sqrt{\alpha_{i}}\mathbf{x}_{0},(1-\alpha_{i})\mathbf{I})
$$

其中$\alpha_{i}:=\prod_{j=1}^{i}{(1-\beta_{j})}$。类似于SMLD，我们可以将扰动数据分布表示为$p_{\alpha_{i}}(\tilde{\mathbf{x}}) := \int{p_{\text{data}}(\mathbf{x})p_{\alpha_{i}}(\tilde{\mathbf{x}}|\mathbf{x})d\mathbf{x}}$。噪声尺度被规定为$\mathbf{x}_{N}$大致按照$\mathcal{N}(0,I)$分布。反向方向的变分马尔可夫链使用

$$
p_{\boldsymbol{\theta}}(\mathbf{x}_{i-1}|\mathbf{x}_{i}) = \mathcal{N}(\mathbf{x}_{i-1};\frac{1}{\sqrt{1-\beta_{i}}}(\mathbf{x}_{i}+\beta_{i}\boldsymbol{s_{\theta}}(\mathbf{x}_{i},i)),\beta_{i}\mathbf{I})
$$

参数化，并且使用证据下界(ELBO)的重加权变体进行训练：

$$
\boldsymbol{\theta}^{*}=\arg\min_{\boldsymbol{\theta}}\sum_{i=1}^{N}(1-\alpha_{i})\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\mathbb{E}_{p_{\alpha_{i}}(\tilde{\mathbf{x}}|\mathbf{x})}\left[\left\|\boldsymbol{s_{\theta}}(\tilde{\mathbf{x}},\sigma_{i})-\nabla_{\tilde{\mathbf{x}}}\log{p_{\alpha_{i}}(\tilde{\mathbf{x}}|\mathbf{x})}\right\|_{2}^{2}\right]
$$

在求解上述方程得到最优模型$\boldsymbol{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x},i)$之后，可以从$\mathbf{x}_{N}\sim{\mathcal{N}(0,I)}$开始，沿着估计的逆马尔可夫链生成样本，如下所示：

$$
\mathbf{x}_{i-1} = \frac{1}{\sqrt{1-\beta_{i}}}(\mathbf{x}_{i}+\beta_{i}\boldsymbol{s_{\theta^{*}}}(\mathbf{x}_{i},i))+\sqrt{\beta_{i}}\mathbf{z}_{i},\quad i=N,N-1,\dots,1
$$

我们将这种方法称为Ancestral Sampling，因为它相当于从图模型$\prod_{i=1}^{N}{p_{\boldsymbol{\theta}}(\mathbf{x}_{i-1}|\mathbf{x}_{i})}$中进行采样。与SMLD中的目标函数类似，DDPM中的目标函数也是一个去噪得分匹配目标的加权求和，这意味着最优模型$\boldsymbol{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x},i)$匹配扰动数据分布的得分$\nabla_{\mathbf{x}}\log{p_{\alpha_{i}}(\mathbf{x})}$。值得注意的是，两个目标函数中第$i$项的权重，即$\sigma_{i}^{2}$和$(1-\alpha_{i})$，与统一函数形式的相应扰动和相关，即

$$
\begin{aligned}
\sigma_{i}^{2}&\propto\frac{1}{\mathbb{E}\left[\left\| \nabla_{\mathbf{x}}\log{p_{\sigma_{i}}(\tilde{\mathbf{x}}|\mathbf{x})} \right\|_{2}^{2}\right]}
\\[10pt]
(1-\alpha_{i})&\propto\frac{1}{\mathbb{E}\left[\left\| \nabla_{\mathbf{x}}\log{p_{\alpha_{i}}(\tilde{\mathbf{x}}|\mathbf{x})} \right\|_{2}^{2}\right]}
\end{aligned}
$$

## 基于SDE的得分生成模型

对数据进行多尺度噪声扰动是先前方法成功的关键。我们提出将这一思想推广到无限多个噪声尺度，使得扰动数据分布随着噪声强度的增加而按照 SDE 演化。框架概述如下图所示：

![\<img alt="Refer to caption" data-attachment-key="GL2VFNBS" width="598" height="202" src="attachments/GL2VFNBS.png" ztype="zimage"> | 598](attachments/GL2VFNBS.png)

### 用随机微分方程扰动数据

我们的目标是构建一个由连续时间变量$t\in[0,T]$索引的扩散过程$\left\{ \mathbf{x}(t) \right\}_{t=0}^{T}$，使得$\mathbf{x}(0)\sim p_{0}$，对于它我们有一个独立同分布样本的数据库，以及$\mathbf{x}(T)\sim p_{T}$，对于它我们有一个高效生成样本的可行形式。换句话说，$p_{0}$是数据分布，而$p_{T}$是先验分布。这个扩散过程可以建模为 Itô 随机微分方程的解：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x},t)dt+g(t)d\mathbf{w}
$$

其中$\mathbf{w}$是标准布朗运动，$\mathbf{f}(\cdot,t):\R^{d}\mapsto\R^{d}$是一个向量值函数，称为$\mathbf{x}(t)$的漂移系数，而$g(\cdot):\R\mapsto\R$是一个标量函数，称为$\mathbf{x}(t)$的扩散系数。为了简化表述，我们假设扩散系数是一个标量（而不是$d\times d$矩阵），并且不依赖于$\mathbf{x}$，但我们的理论可以推广到这些情况。只要系数在状态和时间上都是全局 Lipschitz 的，SDE 就有唯一的强解。我们此后用$p_{t}(\mathbf{x})$表示$\mathbf{x}(t)$的概率密度，并用$p_{st}(\mathbf{x}(t)|\mathbf{x}(s))$表示从$\mathbf{x}(s)$到$\mathbf{x}(t)$的转移核，其中$0\le s\lt t\le T$。

通常情况下，$p_{T}$是一个无结构先验分布，其中不包含$p_{0}$的任何信息，例如均值为固定值和方差为固定值的正态分布。

### 通过逆转SDE生成样本

从$\mathbf{x}(T)\sim p_{T}$的样本开始并逆转过程，我们可以获得$\mathbf{x}(0)\sim p_{0}$的样本，扩散过程的逆转仍然是一个扩散过程，它在时间上反向运行，并由逆转时间 SDE 给出：

$$
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x},t)-g(t)^{2}\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})} \right]dt+g(t)d\hat{\mathbf{w}}
$$

其中当时间从$T$反向流动到0时，$\hat{\mathbf{w}}$是一个标准布朗运动，$dt$是无穷小负时间步长。一旦所有$t$的边际分布的分数$\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})}$已知，我们就可以推导出逆转扩散过程，并模拟它以从$p_{0}$中采样。

### 估计SDE的分数

分布的得分可以通过在具有得分匹配的样本上训练得分模型来估计。为了估计$\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})}$，我们可以通过将SMLD和DDPM中的目标函数推广到连续形式来训练一个时变得分模型$\boldsymbol{s_{\theta}}(\mathbf{x},t)$：

$$
\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}}\mathbb{E}_{t}\left\{ \lambda(t)\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)}\left[\left\| \boldsymbol{s_{\theta}}(\mathbf{x}(t),t)-\nabla_{\mathbf{x}(t)}\log{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))} \right\|_{2}^{2}\right] \right\}
$$

其中$\lambda:[0,T]\mapsto\R_{\gt0}$是一个正加权函数。在足够的数据和模型参数量的支持下，得分匹配确保最优解为$\boldsymbol{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x},t)$，对于几乎所有的$\mathbf{x}$和$t$都等于$\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})}$。与SMLD和DDPM类似，我们通常可以选择

$$
\lambda\propto\frac{1}{\mathbb{E}\left[\left\| \nabla_{\mathbf{x}(t)}\log{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))} \right\|_{2}^{2}\right]}
$$

请注意，虽然此处使用了去噪得分匹配，但其他得分匹配目标，例如切片得分匹配和有限差分得分匹配也适用于此处。

我们通常需要知道转移核$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$才能够高效地求解方程。当$\mathbf{f}(\cdot,t)$是仿射时，转移核总是高斯分布，其中均值和方差通常以封闭形式给出，并且可以通过标准技术获得。对于更一般的 SDE，我们可以求解 Kolmogorov 前向方程来获得$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$。或者，我们可以模拟SDE来从$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$中采样，并将去噪分数匹配替换为切片分数匹配进行模型训练，从而绕过$\nabla_{\mathbf{x}(t)}\log{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))}$的计算，具体实现方式如下。

当随机微分方程（SDE）的漂移系数和扩散系数不是仿射函数时，计算转移核$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$的解析解可能很困难。这阻碍了基于分数的模型的训练，因为目标函数需要知道$\nabla_{\mathbf{x}(t)}\log{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))}$。为了克服这一困难，我们可以用其他不需要计算$\nabla_{\mathbf{x}(t)}\log{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))}$的分数匹配的高效变体来替换目标函数中的去噪分数匹配。例如，在使用切片分数匹配时，我们的训练公式变为

$$
\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}}\mathbb{E}_{t}\left\{ \lambda(t)\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)}\mathbb{E}_{\mathbf{v}\sim p_{\mathbf{v}}}\left[ \frac{1}{2}\left\| \boldsymbol{s_{\theta}}(\mathbf{x}(t),t) \right\|_{2}^{2}+\mathbf{v}^{T}\boldsymbol{s_{\theta}}(\mathbf{x}(t),t)\mathbf{v} \right] \right\}
$$

其中$\lambda:[0,T]\mapsto\R_{\gt0}$是一个正加权函数，$t\sim\mathcal{U}(0,T),\mathbb{E}[\mathbf{v}]=0,\text{Cov}[\mathbf{v}] = \mathbf{I}$。我们可以始终通过模拟SDE来从$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$中采样，并求解上述公式来训练时变分数模型$\boldsymbol{s_{\theta}}(\mathbf{x}(t),t)$。

### 示例：VE, VP SDEs及其扩展

SMLD和DDPM中使用的噪声扰动可以被视为两个不同 SDE 的离散化。下面我们进行简要讨论

当使用总共$N$个噪声尺度时，SMLD 的每个扰动核$p_{\sigma_{i}}(\mathbf{x}|\mathbf{x}_{0})$对应于以下马尔可夫链中$\mathbf{x}_{i}$的分布：

$$
\mathbf{x}_{i} = \mathbf{x}_{i-1}+\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}}\mathbf{z}_{i-1},\quad i=1,\dots,N
$$

其中$\mathbf{z}_{i-1}\sim\mathcal{N}(0,I)$，并且我们引入$\sigma_{0} = 0$来简化符号。当$N\to\infin$时，$\left\{\sigma_{i}\right\}_{i=1}^{N}$变成一个函数$\sigma(t)$，$\mathbf{z}_{i}$也变成一个函数$\mathbf{z}(t)$，马尔可夫链$\left\{\mathbf{x}_{i}\right\}_{i=1}^{N}$变成一个连续随机过程$\left\{ \mathbf{x}(t) \right\}_{t=0}^{1}$，我们在索引中使用了连续时间变量$t\in[0,1]$而不是整数$i$。过程$\left\{ \mathbf{x}(t) \right\}_{t=0}^{1}$由以下随机微分方程给出

$$
d\mathbf{x} = \sqrt{\frac{d[\sigma^{2}(t)]}{dt}}d\mathbf{w}
$$

同样的，对于DDPM的扰动核$\left\{ p_{\alpha_{i}}(\mathbf{x}|\mathbf{x}_0) \right\}_{i=1}^{N}$，离散马尔可夫链是

$$
\mathbf{x}_{i} = \sqrt{1-\beta_{t}}\mathbf{x}_{i-1}+\sqrt{\beta_{i}}\mathbf{z}_{i-1},\quad i=1,\dots,N
$$

当$N\to\infin$时，上式收敛到以下随机微分方程：

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)}d\mathbf{w}
$$

因此，SMLD 和 DDPM 中使用的噪声扰动对应于随机微分方程等式的离散化。有趣的是，当$t\to\infin$时，SMLD对应的SDE总是给出一个方差爆炸的过程，而DDPM对应的SDE在初始分布具有单位方差时产生一个方差为1的过程。由于这种差异，我们在此后称SMLD的SDE为方差爆炸(Variance Exploding)随机微分方程，称DDPM的SDE为方差保持(Variance Preserving)随机微分方程。

受到VP随机微分方程的启发，可以提出一种新的随机微分方程类型，它在似然函数上表现特别出色，其形式为：

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)(1-e^{-2\int_{0}^{t}{\beta(s)ds}})}d\mathbf{w}
$$

当使用相同的$\beta(t)$并从相同的初始分布开始时，等式上式诱导的随机过程的方差在每一中间时间步都始终被 VP 随机微分方程所限制。因此，我们称等式 12 为亚 VP (sub-VP随机微分方程。

由于VE、VP和sub-VP都具有仿射漂移系数，它们的扰动核$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$都是高斯分布的，并且可以以封闭形式计算。

下面是详细推导过程，以证明 SMLD 和 DDPM 的噪声扰动分别是方差爆炸（VE）和方差保持（VP）SDEs 的离散化。

#### VE SDE

首先，在SMLD中，令$\mathbf{x}\left( \frac{i}{N} \right) = \mathbf{x}_{i},\;\sigma\left( \frac{i}{N} \right) = \sigma_{i},\;\mathbf{z}\left( \frac{i}{N} \right) = \mathbf{z}_{i},\;i=1,2,\dots,N$。我们可以使用$\Delta{t}=\frac{1}{N}$和$t\in\left\{0,\frac{1}{N},\dots,\frac{N-1}{N}\right\}$将马尔可夫链重写如下：

$$
\mathbf{x}(t+\Delta{t})=\mathbf{x}(t)+\sqrt{\sigma^{2}(t+\Delta{t})-\sigma^{2}(t)}\,\mathbf{z}(t)\approx\mathbf{x}(t)+\sqrt{\frac{d[\sigma^{2}(t)]}{dt}\Delta{t}}\,\mathbf{z}(t)
$$

当$\Delta{t}\ll1$时近似相等。当$\Delta{t}\to0$时，收敛到

$$
d\mathbf{x} = \sqrt{\frac{d[\sigma^{2}(t)]}{dt}}d\mathbf{w}
$$

其中$d\mathbf{w} = \sqrt{\Delta{t}}\,\mathbf{z}(t)$，这就是VE SDE的详细推导过程。

#### VP SDE

我们定义一组辅助噪声尺度$\left\{ \hat{\beta}_{i} = N\beta_{i} \right\}_{i=1}^{N}$，并且将离散马尔可夫链重写如下：

$$
\mathbf{x}_{i} = \sqrt{1-\frac{\hat{\beta_{i}}}{N}}\mathbf{x}_{i-1}+\sqrt{\frac{\hat{\beta_{i}}}{N}}\mathbf{z}_{i-1},\quad i=1,\dots,N
$$

在$N\to\infin$的极限情况下，$\left\{\hat{\beta}_{i}\right\}_{i=1}^{N}$称为由$t\in[0,1]$索引的函数。令$\beta\left(\frac{i}{N}\right) = \hat{\beta_{i}},\;\mathbf{x}\left( \frac{i}{N} \right) = \mathbf{x}_{i},\;\mathbf{z}\left( \frac{i}{N} \right) = \mathbf{z}_{i},\;i=1,2,\dots,N$，我们可以将马尔可夫链用$\Delta{t} = \frac{1}{N}$和$t\in\left\{0,\frac{1}{N},\dots,\frac{N-1}{N}\right\}$重写为以下形式：

$$
\begin{aligned}
\mathbf{x}(t+\Delta{t}) &= \sqrt{1-\beta(t+\Delta{t})\Delta{t}}\,\mathbf{x}(t)+\sqrt{\beta(t+\Delta{t})\Delta{t}}\,\mathbf{z}(t)
\\[10pt]
&\approx\mathbf{x}(t)-\frac{1}{2}\beta(t+\Delta{t})\Delta{t}\,\mathbf{x}(t)+\sqrt{\beta(t+\Delta{t})\Delta{t}}\,\mathbf{z}(t)
\\[10pt]
&\approx\mathbf{x}(t)-\frac{1}{2}\beta(t)\Delta{t}\,\mathbf{x}(t)+\sqrt{\beta(t)\Delta{t}}\,\mathbf{z}(t)
\end{aligned}
$$

当$\Delta{t}\ll1$时，等式近似成立。因此，当$\Delta{t}\to0$时，上述公式收敛到VP随机微分方程

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{w}
$$

其中$d\mathbf{w} = \sqrt{\Delta{t}}\,\mathbf{z}(t)$，这就是VP SDE的详细推导过程。

## 求解反向SDE

在训练一个时变得分模型$\boldsymbol{s_{\theta}}$后，我们可以使用它来构建反向时间 SDE，然后通过数值方法模拟它，以从$p_{0}$生成样本。

### 通用数值SDE求解器

数值求解器为随机微分方程提供近似轨迹。存在许多通用数值方法用于求解随机微分方程，例如 Euler-Maruyama 方法和随机 Runge-Kutta 方法，它们对应于随机动力学的不同离散化。我们可以将它们中的任何一种应用于反向时间随机微分方程以进行样本生成。

Ancestral Sampling实际上对应于反向时间 VP 随机微分方程的一种特殊离散化。然而为新的随机微分方程推导Ancestral Sampling规则可能并不简单。为此，我们提出了反向扩散采样器，它以与正向相同的方式离散化反向时间随机微分方程，因此给定正向离散化即可轻易推导。

### 预测-校正采样器

与通用 SDE 不同，我们拥有可用于改进解的额外信息。由于我们有一个基于分数的模型$\boldsymbol{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x},t)\approx\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})}$，我们可以采用基于分数的 MCMC 方法或HMC，直接从$p_{t}$中采样，并校正数值SDE求解器的解。

具体来说，在每个时间步，数值 SDE 求解器首先给出下一个时间步样本的估计值，起到“预测器”的作用。然后，基于分数的 MCMC 方法校正估计样本的边缘分布，起到“校正器”的作用。我们同样将我们的混合采样算法命名为预测-校正（PC）采样器。PC 采样器推广了 SMLD 和 DDPM 的原始采样方法：前者使用恒等函数作为预测器，退火 Langevin 动力学作为校正器，而后者使用Ancestral Sampling作为预测器，恒等函数作为校正器。

### 概率流与神经ODEs的关联

基于分数的模型为求解反向时间随机微分方程提供了一种数值方法。对于所有扩散过程，都存在一个相应的确定性过程，其轨迹与随机微分方程具有相同的边际概率密度。该确定性过程满足一个常微分方程

$$
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x},t)-\frac{1}{2}g(t)^{2}\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})} \right]dt
$$

一旦已知分数，就可以从随机微分方程中确定该常微分方程。我们将上式中的常微分方程称为概率流常微分方程。当分数函数通过时变分数模型近似时（通常是一个神经网络），这便是一个神经常微分方程的例子。

**精确似然计算**：利用与神经常微分方程的联系，我们可以通过变量瞬时变化公式计算由上式定义的密度。这使我们能够在任何输入数据上计算精确似然。

**操控潜在表示**：通过对上式进行积分，我们可以将任意数据点$\mathbf{x}(0)$编码到潜在空间$\mathbf{x}(T)$中。解码可以通过对反向时间 SDE 积分相应的常微分方程来实现。与其他可逆模型（如神经 ODE 和归一化流）所做的一样，我们可以操纵这种潜在表示进行图像编辑，例如插值和温度缩放。

**唯一可识别编码**：与大多数当前可逆模型不同，我们的编码是唯一可识别的，这意味着在足够多的训练数据、模型容量和优化精度下，输入的编码由数据分布唯一确定。这是因为我们的正向随机微分方程没有可训练参数，而其相关的概率流常微分方程，即上式，在完美估计分数时提供相同的轨迹。

**高效采样**：与神经 ODE 类似，我们可以通过从不同的最终条件$\mathbf{x}(T)\sim p_{T}$求解上式来采样$\mathbf{x}(0)\sim p_{0}$。使用固定的离散化策略，我们可以生成具有竞争力的样本，尤其是在与校正器结合使用时。使用黑盒 ODE 求解器不仅能够产生高质量的样本，而且允许我们明确地在准确性和效率之间进行权衡。通过更大的误差容限，可以在不影响样本视觉质量的情况下，将函数评估次数减少超过 90%。

## 可控生成

我们框架的连续结构使我们不仅能够从$p_{0}$生成数据样本，而且如果$p_{t}(\mathbf{y}|\mathbf{x}(t))$已知，还能从$p_{0}(\mathbf{x}(0)|\mathbf{y})$生成。给定正向SDE，我们可以通过从$p_{T}(\mathbf{x}(T)|\mathbf{y})$开始并求解条件反向时间SDE来从$p_{t}(\mathbf{x}(t)|\mathbf(y))$采样。

$$
d\mathbf{x} = \left\{ \mathbf{f}(\mathbf{x},t)-g(t)^{2}\left[\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{x})}+\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{y}|\mathbf{x})} \right]\right\}dt+g(t)d\hat{\mathbf{w}}
$$

通常情况下，一旦给定正向过程的梯度估计值$\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{y}|\mathbf{x}(t))}$，我们可以使用上式解决一类大型逆问题。在某些情况下，我们可以训练一个单独的模型来学习正向过程$\nabla_{\mathbf{x}}\log{p_{t}(\mathbf{y}|\mathbf{x}(t))}$并计算其梯度。否则，我们可以使用启发式方法和领域知识来估计梯度。

我们考虑了这种方法在三个可控生成应用中的使用：类条件生成、图像补全和上色。当$\mathbf{y}$代表类标签时，我们可以训练一个时变分类器$p_{t}(\mathbf{y}|\mathbf{x}(t))$进行类条件采样。由于正向 SDE 是可处理的，我们可以通过首先从数据集中采样$(\mathbf{x}(0),\mathbf{y})$，然后采样$\mathbf{x}(t)\sim p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$，轻松地为时变分类器创建训练数据$(\mathbf{x}(t),\mathbf{y})$。之后，我们可以采用不同时间步长的交叉熵损失混合来训练时变分类器$p_{t}(\mathbf{y}|\mathbf{x}(t))$。

## 结论

我们提出了基于随机微分方程（SDEs）的基于分数的生成模型框架。我们的工作有助于更好地理解现有方法、新的采样算法、精确似然计算、唯一可识别的编码、潜在代码操作，并为基于分数的生成模型家族带来了新的条件生成能力。

虽然我们提出的采样方法提高了结果并实现了更高效的采样，但在相同数据集上，它们的采样速度仍然比 GANs慢。将基于分数的生成模型的稳定学习与 GANs 等隐式模型的快速采样相结合的方法仍然是一个重要的研究方向。此外，当可以访问分数函数时，人们可以使用各种采样器，这引入了一系列超参数。未来的工作将受益于改进自动选择和调整这些超参数的方法，以及对各种采样器的优缺点进行更广泛的研究。
