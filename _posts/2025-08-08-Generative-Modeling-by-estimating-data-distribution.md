---
title: "Generatie Modeling by Estimating Gradients of the Data Distribution"
mathjax: true
layout: post
categories: Generative Modeling
---

# Generatie Modeling by Estimating Gradients of the Data Distribution

***

## 引言

在这篇论文中，我们探索了一种基于分数采样的生成模型新原理，该分数是数据密度对数在输入数据点的梯度。这是一个指向对数数据密度增长最快的方向的矢量场。我们使用一个通过分数匹配训练的神经网络来从数据中学习这个矢量场。然后我们使用朗之万动力学产生样本，其近似工作原理是沿着（估计的）分数矢量场，逐渐将一个随机的初始样本移动到高密度区域。然而，这种方法存在两个主要挑战。首先，如果数据分布在低维流形上——正如许多真实世界数据集通常被假设的那样——分数在环境空间中将没有定义，分数匹配将无法提供一个一致的分数估计器。其次，在低数据密度区域（例如远离流形的地方）训练数据的稀缺性阻碍了分数估计的准确性，并减缓了朗之万动力学采样的混合速度。 由于Langevin动力学通常会在数据分布的低密度区域初始化，因此这些区域的分数估计不准确会负面影响采样过程。此外，由于需要穿越低密度区域以在不同分布模式之间进行转换，混合过程可能会很困难。

为了应对这两个挑战，我们提出用不同幅度的随机高斯噪声扰动数据。添加随机噪声可以确保生成的分布不会坍缩到低维流形。高噪声水平会在原始（未扰动）数据分布的低密度区域产生样本，从而提高评分估计。关键在于，我们训练一个基于噪声水平的评分网络，并在所有噪声幅度下估计评分。然后，我们提出了一种退火版本的朗之万动力学，初始时使用对应最高噪声水平的评分，并逐渐降低噪声水平，直到它足够小，以至于与原始数据分布无法区分。

## 基于分数的生成建模

假设我们的数据集由来自未知数据分布$p_{\text{data}}(\mathbf{x})$的独立同分布样本$\left\{ \mathbf{x}_{i}\in\R^{D} \right\}_{i=1}^{N}$组成。我们将概率密度$p(\mathbf{x})$的得分函数定义为$\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$。得分网络$\boldsymbol{s_{\theta}}:\R^{D}\mapsto\R^{D}$是一个由$\boldsymbol{\theta}$参数化的神经网络，它被训练以近似$p_{\text{data}}(\mathbf{x})$的得分。生成式建模的目标是利用数据集学习一个模型，用于从$p_{\text{data}}(\mathbf{x})$生成新样本。基于得分的生成式建模框架包含两个要素：得分匹配和朗之万动力学。

### 用于得分估计的Score Matching

Score Matching最初是为基于来自未知数据分布的独立同分布样本学习非归一化统计模型而设计的。我们将它用于得分估计。使用得分匹配，我们可以直接训练得分网络$\boldsymbol{s_{\theta}}(\mathbf{x})$来估计$\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$，而无需首先训练一个模型来估计$p_{\text{data}}(\mathbf{x})$。与得分匹配的典型用法不同，我们选择不使用基于能量的模型的梯度作为得分网络，以避免由于高阶梯度导致的额外计算。目标是最小化以下公式：

$$
\begin{equation}
\frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[ \left\| \boldsymbol{s_{\theta}}(\mathbf{x})-\nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})} \right\|_{2}^{2} \right]\label{def}
\end{equation}
$$

这可以证明在常数因子下等效于下式：

$$
\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\left[ \tr{\left( \nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x}) \right)} + \frac{1}{2}\left\| \boldsymbol{s_{\theta}}(\mathbf{x}) \right\|_{2}^{2} \right]
$$

其中$\nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x})$表示$\boldsymbol{s_{\theta}}(\mathbf{x})$的雅可比矩阵。在满足某些正则性条件的情况下，能够使得上述目标公式最小的$\boldsymbol{s_{\theta^{*}}}(\mathbf{x})$必然满足$\boldsymbol{s_{\theta^{*}}}(\mathbf{x}) = \nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})}$。在实践中，可以使用数据样本快速估计方程中的$p_{\text{data}}(\mathbf{x})$的期望。然而，由于$\tr{\left( \nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x}) \right)}$的计算，分数匹配无法扩展到深度网络和高维数据。下面讨论两种大规模分数匹配的流行方法。

#### Denoising Score Matching

去噪分数匹配是分数匹配的一种变体，它完全绕过了$\tr{\left( \nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x}) \right)}$的计算。它首先使用预定义的噪声分布$p(\tilde{\mathbf{x}}|\mathbf{x})$扰动数据点$\mathbf{x}$，然后采用分数匹配来估计扰动数据分布$q_{\sigma}(\tilde{\mathbf{x}})=\int{q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})p_{\text{data}}(\mathbf{x})d\mathbf{x}}$的分数。该目标被证明等同于以下公式：

$$
\frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})p_{\text{data}}(\mathbf{x})}\left[\left\| \boldsymbol{s_{\theta}}(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log{q_{\sigma}}(\tilde{\mathbf{x}}|\mathbf{x}) \right\|_{2}^{2}\right]
$$

能够使得上述目标公式最小的$\boldsymbol{s_{\theta^{*}}}(\mathbf{x})$必然满足$\boldsymbol{s_{\theta^{*}}}(\mathbf{x}) = \nabla_{\mathbf{x}}\log{q_{\sigma}(\mathbf{x})}$。然而，只有当噪声足够小，使得$q_{\sigma}(\mathbf{x})\approx p_{\text{data}}(\mathbf{x})$的时候，$\boldsymbol{s_{\theta^{*}}}(\mathbf{x}) = \nabla_{\mathbf{x}}\log{q_{\sigma}(\mathbf{x})}\approx \nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})}$才能够成立。

#### Sliced Score Matching

切片得分匹配使用随机投影来近似得分匹配中的$\tr{\left( \nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x}) \right)}$ 。目标是

$$
\mathbb{E}_{p_{\mathbf{v}}}\mathbb{E}_{p_{\text{data}}}\left[ \mathbf{v}^{T}\nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x})\mathbf{v}+\frac{1}{2}\left\| \boldsymbol{s_{\theta}}(\mathbf{x}) \right\|_{2}^{2} \right]
$$

其中$p_{\mathbf{v}}$是一个简单的随机向量分布，例如多元标准正态分布。项$\mathbf{v}^{T}\nabla_{\mathbf{x}}\boldsymbol{s_{\theta}}(\mathbf{x})\mathbf{v}$可以通过前向模式自动微分高效计算。与估计扰动数据的去噪分数匹配不同，切片分数匹配为原始未扰动数据分布提供分数估计，但由于前向模式自动微分，需要大约四倍的计算量。

### 使用朗之万动力学采样

朗之万动力学可以使用仅有的评分函数$\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$从概率密度$p(\mathbf{x})$中生成样本。给定一个固定的步长$\epsilon\gt0$，以及一个初始值$\tilde{\mathbf{x}}_{0}\sim\pi(\mathbf{x})$，其中$\pi$是一个先验分布，朗之万方法递归地计算如下

$$
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1}+\frac{\epsilon}{2}\nabla_{\mathbf{x}}\log{p(\tilde{\mathbf{x}}_{t-1})}+\sqrt{\epsilon}\cdot\mathbf{z}_{t}
$$

其中$\mathbf{z}_{t}\sim\mathcal{N}(0,I)$。当$\epsilon\to0,T\to\infin$的时候，$\tilde{\mathbf{x}}_{T}$的分布等于$p(\mathbf{x})$，在这种情况下，$\tilde{\mathbf{x}}_{T}$在某些正则条件下成为$p(\mathbf{x})$的一个精确样本。当$\epsilon\gt0,T\lt\infin$的时候，在实践中通常可以忽略误差。在此处，我们假设当$\epsilon$很小并且$T$很大的时候，这个误差可以忽略不计。

请注意，采样的时候只需要分数函数$\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$。因此，为了获得$p_{\text{data}}(\mathbf{x})$的样本，我们可以首先训练我们的分数网络使得$\boldsymbol{s_{\theta}}(\mathbf{x})\approx \nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})}$，然后使用$\boldsymbol{s_{\theta}}(\mathbf{x})$近似地通过朗之万动力学获得样本。这是基于分数的生成模型框架的关键思想。

## 基于分数的生成式建模的挑战

有两个主要障碍阻碍了基于分数的生成模型的简单应用，分别是**流形假设**和**低数据密度区域**。

### 流形假设

流形假设认为，现实世界中的数据倾向于集中在嵌入在高维空间（也称为环境空间）中的低维流形上。这一假设在许多数据集上得到了经验验证，并成为流形学习的基础。在流形假设下，基于分数的生成模型将面临两个关键困难。首先，由于分数$\nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})}$是在环境空间中计算的梯度，当$\mathbf{x}$被限制在低维流形上时，它是未定义的。其次，分数匹配目标函数只有在数据分布的支持集是整个空间时，才能提供一个一致的分数估计器，而当数据存在于低维流形上时，它将是不一致的。

### 低数据密度区域

低密度区域的数据稀缺会给分数匹配的分数估计和朗之万动力学下的 MCMC 采样带来困难。

#### Score Matching的Score估计不准确

在数据密度低的区域，由于缺乏数据样本，得分匹配可能没有足够的证据来准确估计得分函数。在2.1节中由定义$\eqref{def}$，得分匹配的目的是最小化得分估计的预期平方误差。在实践中，关于数据分布的期望总是使用独立同分布样本$\left\{ \mathbf{x}_{i} \right\}_{i=1}^{N}\sim p_{\text{data}}(\mathbf{x})$来估计。考虑$p_{\text{data}}(\mathbf{x})\approx0$的任何区域$\mathcal{R}\sub\R^{D}$，在大多数情况下$\left\{ \mathbf{x}_{i} \right\}_{i=1}^{N}\cap\mathcal{R}=\O$，对于$\mathbf{x}\in\mathcal{R}$中的样本，得分匹配将没有足够的数据样本来准确估计$\boldsymbol{s_{\theta}}(\mathbf{x})\approx \nabla_{\mathbf{x}}\log{p_{\text{data}}(\mathbf{x})}$。

#### 朗之万动力学的缓慢混合

当数据分布的两个模式被低密度区域隔开时，朗之万动力学将无法在合理时间内正确恢复这两个模式的相对权重，因此可能无法收敛到真实分布。

考虑一个混合分布$p_{\text{data}}(\mathbf{x}) = \pi p_{1}(\mathbf{x})+(1-\pi)p_{2}(\mathbf{x})$，其中$p_{1}(\mathbf{x})$和$p_{2}(\mathbf{x})$是不相交的归一化概率分布，$\pi\in(0,1)$。那么我们可以得到：

$$
\begin{aligned}
\nabla_{\mathbf{x}}\log{p_{\text{data}}}(\mathbf{x}) &= \nabla_{\mathbf{x}}(\log{\pi}+\log{p_{1}(\mathbf{x})}) = \nabla_{\mathbf{x}}\log{p_{1}(\mathbf{x})}
\\[10pt]
\nabla_{\mathbf{x}}\log{p_{\text{data}}}(\mathbf{x}) &= \nabla_{\mathbf{x}}(\log{(1-\pi)}+\log{p_{2}(\mathbf{x})}) = \nabla_{\mathbf{x}}\log{p_{2}(\mathbf{x})}
\end{aligned}
$$

在任意情况下，得分函数$\nabla_{\mathbf{x}}\log{p_{\text{data}}}(\mathbf{x})$不依赖于$\pi$。由于朗之万动力学使用$\nabla_{\mathbf{x}}\log{p_{\text{data}}}(\mathbf{x})$从$p_{\text{data}}(\mathbf{x})$中采样，因此，获得的样本将不依赖于$\pi$。在实践中，当不同的模式具有近似不相交的支撑时，这一分析同样成立——它们可能共享相同的支撑，但通过数据密度很小的区域相连。在这种情况下，Langevin 动力学在理论上可以产生正确的样本，但可能需要非常小的步长和非常大的步数来进行混合。

## Noise Conditional Score Networks: Learning and Inference

我们观察到用随机高斯噪声扰动数据会使数据分布更易于基于分数的生成建模。首先，由于我们的高斯噪声分布的支持集是整个空间，扰动后的数据不会局限于低维流形，这消除了流形假设带来的困难，并使分数估计具有明确定义。其次，较大的高斯噪声具有填充原始未扰动数据分布中低密度区域的效果；因此分数匹配可能获得更多训练信号来改进分数估计。此外，通过使用多个噪声级别，我们可以获得一系列噪声扰动分布，这些分布收敛到真实数据分布。我们可以利用这些中间分布，基于模拟退火和退火重要性采样的，提高朗之万动力学在多模态分布上的混合率。

基于这一直觉，我们提出通过：

1.  使用不同级别的噪声扰动数据
2.  通过训练单个条件分数网络同时估计所有噪声级别的分数来改进基于分数的生成模型。

训练完成后，在利用朗之万动力学生成样本时，我们首先使用对应较大噪声的分数，然后逐渐降低噪声级别。这有助于将较大噪声级别的优势平滑地转移到低噪声级别，在低噪声级别处扰动数据几乎与原始数据无法区分。接下来，详细阐述我们方法的具体细节，包括分数网络的架构、训练目标以及朗之万动力学的退火调度。

### Noise Consitional Score Networks

令$\left\{ \sigma_{i} \right\}_{i=1}^{L}$为一个满足$\frac{\sigma_{1}}{\sigma_{2}} = \dots=\frac{\sigma_{L-1}}{\sigma_{L}}\gt1$的正几何序列。令$q_{\sigma}(\mathbf{x}) = \int{p_{\text{data}}(\mathbf{t})\mathcal{N}(\mathbf{x}|\mathbf{t},\sigma^{2}I)d\mathbf{t}}$表示扰动数据分布。我们构造的$\sigma$序列能够使得$\sigma_{1}$足够大以缓解前文提到的困难，并且$\sigma_{L}$坐骨小以最小化对数据的影响。我们的目标是训练一个条件分数网络，以联合估计所有扰动数据分布的分数，即

$$
\forall\sigma\in\left\{ \sigma_{i} \right\}_{i=1}^{L}:\boldsymbol{s_{\theta}}(\mathbf{x};\sigma)\approx\nabla_{\mathbf{x}}\log{q_{\sigma}(\mathbf{x})}
$$

我们称$\boldsymbol{s_{\theta}}(\mathbf{x};\sigma)$为Noise Conditional Score Networks (NSCN)。

### 通过Score Matching训练NCSNs

切片和去噪分数匹配都可以训练 NCSNs。我们采用去噪分数匹配，因为它稍快且自然地适用于估计噪声扰动数据分布的分数的任务。然而，我们强调经验上切片分数匹配也能像去噪分数匹配一样训练 NCSNs。我们选择噪声分布为$q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x},\sigma^{2}I)$，因此其分数函数为$\nabla_{\mathbf{x}}\log{q_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})} = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^{2}}$。对于给定的$\sigma$，去噪分数匹配目标是：

$$
\mathcal{l}(\boldsymbol{\theta};\sigma) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\mathbb{E}_{\tilde{\mathbf{x}}\sim\mathcal{N}(\mathbf{x},\sigma^{2}I)}\left[ \left\| \boldsymbol{s_{\theta}}(\tilde{\mathbf{x}},\sigma) + \frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^{2}} \right\|_{2}^{2} \right]
$$

然后，我们将所有的$\sigma\in\left\{ \sigma_{i} \right\}_{i=1}^{L}$进行组合，得到一个统一的目标函数

$$
\mathcal{L}(\boldsymbol{\theta};\left\{ \sigma_{i} \right\}_{i=1}^{L}) = \frac{1}{L}\sum_{i=1}^{L}{\lambda(\sigma_{i})\cdot l(\boldsymbol{\theta};\sigma_{i})}
$$

其中$\lambda(\sigma_{i})\gt0$是一个依赖于$\sigma_{i}$的系数函数，假设$\boldsymbol{s_{\theta}}(\mathbf{x};\sigma)$具备足够的能力，$\boldsymbol{s_{\theta^{*}}}(\mathbf{x};\sigma)$当且仅当$\boldsymbol{s_{\theta^{*}}}(\mathbf{x};\sigma)=\nabla_{\mathbf{x}}\log{q_{\sigma_{i}}(\mathbf{x})}$几乎处处成立时才能够最小化目标函数，因为它是去噪得分匹配目标的一个锥形组合。

$\lambda(\cdot)$有多个可能的选项，理想情况下，我们希望所有的$\left\{ \sigma_{i} \right\}_{i=1}^{L}$的$\lambda(\sigma_{i})\cdot l(\boldsymbol{\theta};\sigma_{i})$大致处于同一个数量级。通过实验发现，$\left\| \boldsymbol{s_{\theta}}(\mathbf{x};\sigma) \right\|\propto\frac{1}{\sigma}$。因此我们可以选择$\lambda(\sigma) = \sigma^{2}$。因为在这种选择下，我们有

$$
\lambda(\sigma_{i})\cdot l(\boldsymbol{\theta};\sigma_{i}) = \sigma^{2}\cdot l(\boldsymbol{\theta};\sigma_{i}) = \frac{1}{2}\mathbb{E}\left[ \left\| \sigma\boldsymbol{s_{\theta}}(\tilde{\mathbf{x}},\sigma) + \frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma}  \right\|_{2}^{2} \right]
$$

由于$\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma}\sim\mathcal{N}(0,I)$，并且$\left\| \sigma\boldsymbol{s_{\theta}}(\mathbf{x};\sigma) \right\|\propto1$，容易得到$\lambda(\sigma_{i})\cdot l(\boldsymbol{\theta};\sigma_{i})$的数量级不依赖于$\sigma$

### 通过退火朗之万动力学进行NCSN推理

在NCSN$\boldsymbol{s_{\theta}}(\mathbf{x};\sigma)$训练完成后，可以利用退火朗之万动力学算法来生成样本。我们通过某个固定先验分布初始化样本（例如均匀噪声或者高斯分布）开始退火朗之万动力学。然后，我们以步长$\alpha_{1}$运行朗之万动力学，从$q_{\sigma_{1}}(\mathbf{x})$中采样。接下来，我们以步长$\alpha_{2}$运行朗之万动力学，从$q_{\sigma_{2}}(\mathbf{x})$中采样，起始样本为上一轮模拟的最终样本。我们继续这种方式，将朗之万动力学对$q_{\sigma_{i-1}}(\mathbf{x})$的最终样本作为朗之万动力学对$q_{\sigma_{i}}(\mathbf{x})$的厨师样本，并逐渐减小步长$\alpha_{i}$。最后，我们运行朗之万动力学从$q_{\sigma_{L}}(\mathbf{x})$中采样，当$\sigma_{L}\approx0$的时候，$p_{\sigma_{L}}(\mathbf{x})$接近于$p_{\text{data}}(\mathbf{x})$。

由于分布$\left\{ q_{\sigma} \right\}_{i=1}^{L}$都受到高斯噪声的扰动，它们的支撑集覆盖整个空间，并且它们的得分是良好定义的，从而避免了流形假设带来的困难。当$\sigma_{1}$足够大时，$q_{\sigma_{1}}(\mathbf{x})$的低密度区域会变小，模式变得不那么孤立。如前所述，这可以使得分估计更加准确，并加快朗之万动力学的混合速度。因此，我们可以假设朗之万动力学为$q_{\sigma_{1}}(\mathbf{x})$生成良好的样本。这些样本很可能来自$q_{\sigma_{1}}(\mathbf{x})$的高密度区域，这意味着它们也很有可能存在于$q_{\sigma_{2}}(\mathbf{x})$的高密度区域，因为$q_{\sigma_{1}}(\mathbf{x})$和$q_{\sigma_{2}}(\mathbf{x})$之间只有微小的差异。由于得分估计和朗之万动力学在高密度区域表现更好，$q_{\sigma_{1}}(\mathbf{x})$的样本将作为$q_{\sigma_{2}}(\mathbf{x})$朗之万动力学的良好初始样本。类似地，$q_{\sigma_{i-1}}(\mathbf{x})$为$q_{\sigma_{i}}(\mathbf{x})$提供良好的初始样本，最终我们从$q_{\sigma_{L}}(\mathbf{x})$获得高质量的样本。

调整$\alpha_{i}$的方法有很多种，可以选择$\alpha_{i}\propto\sigma_{i}^{2}$。其动机为固定朗之万动力学中的信噪比的幅度，即

$$
\frac{\alpha_{i}\boldsymbol{s_{\theta}}(\mathbf{x},\sigma_{i})}{2\sqrt{\alpha_{i}}\,\mathbf{z}}
$$

我们计算其平方的期望，可以得到：

$$
\mathbb{E}\left[ \left\| \frac{\alpha_{i}\boldsymbol{s_{\theta}}(\mathbf{x},\sigma_{i})}{2\sqrt{\alpha_{i}}\,\mathbf{z}} \right\|_{2}^{2} \right]\approx\mathbb{E}\left[  \frac{\alpha_{i}\left\|\boldsymbol{s_{\theta}}(\mathbf{x},\sigma_{i}) \right\|_{2}^{2}}{4} \right] \propto \frac{1}{4}\mathbb{E}\left[\left\| \sigma_{i}\boldsymbol{s_{\theta}}(\mathbf{x},\sigma_{i}) \right\|_{2}^{2}\right]\propto1
$$

因此就可以得到信噪比的幅度和$\alpha_{i}$无关。
