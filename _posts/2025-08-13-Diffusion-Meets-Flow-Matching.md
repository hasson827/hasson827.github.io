---
title: "Diffusion Meets Flow Matching: Two Sides of the Same Coin"
mathjax: true
layout: post
categories: Generative Modeling
---
# Diffusion Meets Flow Matching: Two Sides of the Same Coin

---

## Overview

### Diffusion Models

扩散过程随时间$t$逐渐破坏观测到的数据点$x$（例如图像），通过将数据与高斯噪声混合。时间$t$的噪声数据由争相过程给出：
$$
z_t = \alpha_tx+\sigma_t\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)
$$
$\alpha_t$和$\sigma_t$定义了**噪声调度**。如果$\alpha_t^2+\sigma_t^2=1$，那么噪声调度也被称为Variance-Preserving。噪声调度的设计使得$z_0$接近干净数据，而$z_1$接近高斯噪声。

要生成新的样本，我们可以“逆转”正向过程：我们从标准高斯分布初始化$z_1$。给定时间步$t$的样本$z_t$，我们用神经网络$\hat{x} = \hat{x}(z_t;t)$来预测干净样本可能的样子，然后我们又用相同的前向变换将其投影回较低的噪声水平$s$：
$$
z_s = \alpha_s\hat{x}+\sigma_s\hat{\epsilon}
$$
 其中$\hat{\epsilon} = \frac{z_t-\alpha_t\hat{x}}{\sigma_t}$（或者我们可以训练一个神经网络来预测噪声$\hat{\epsilon}$）。我们不断在预测干净数据和将其投影回较低噪声水平之间交替，直到得到干净样本。这就是DDIM采样器。样本的随机性仅来自初始的高斯样本，整个反向过程是确定的。我们在后面讨论随机采样器。

### Flow Matching

在流匹配中，我们将正向过程视为数据$x$和噪声项$\epsilon$之间的线性插值：
$$
z_t = (1-t)x+t\epsilon
$$
如果我们使用调度$\alpha_t=1-t,\sigma_t=t$，这就对应于高斯噪声的扩散正向过程。

对于$s\lt t$，我们可以推导出$z_t=z_s+u(t-s)$，其中$u=\epsilon-x$是“速度”、“流”或“向量场”。因此，要在给定$z_t$的情况下采样$z_s$，我们需要逆转时间，并将向量场替换为时间$t$下的最佳猜测：$\hat{u}=\hat{u}(z_t;t)=\hat{\epsilon}-\hat{x}$，由神经网络表示，得到：
$$
z_s=z_t+\hat{u}(s-t)
$$
从标准高斯分布初始化样本$z_1$后，我们持续在比$z_t$更低的噪声水平下得到$z_s$，直到获得干净样本。

### Comparison

到目前为止，我们已经可以分辨出这两个框架中的相似本质：

1. 相同的前向过程，如果我们假设流匹配的一端是高斯分布，并且扩散模型的噪声调度具有特定形式。
2. “相似”采样过程：两者都遵循一个迭代更新过程，该过程涉及在当前时间步对干净数据进行猜测（下面证明两者完全相同）。

## Sampling

人们普遍认为这两个框架在生成样本的方式上有所不同：Flow Matching的采样是确定性的，具有“直线”路径，而扩散模型采样是随机的，遵循“曲线”路径。我们将首先关注确定性采样，稍后会讨论随机情况。

想象一下，您想使用训练好的去噪模型将随机噪声转换为数据点。回想一下，DDIM 更新由$z_s = \alpha_s\hat{x}+\sigma_s\hat{\epsilon}$给出。有趣的是，通过重新排列项，它可以表示为以下公式，涉及多个网络输出和重新参数化：
$$
\tilde{z}_s = \tilde{z}_t+\text{Network Output}\cdot(\eta_s-\eta_t)
$$

| Network Output                       |                      Reparametrization                       |
| :----------------------------------- | :----------------------------------------------------------: |
| $\hat{x}$-Prediction                 | $\tilde{z}_t=\frac{z_t}{\sigma_t}$ and $\eta_t=\frac{\alpha_t}{\sigma_t}$ |
| $\hat{\epsilon}$-Prediction          | $\tilde{z}_t=\frac{z_t}{\alpha_t}$ and $\eta_t=\frac{\sigma_t}{\alpha_t}$ |
| $\hat{u}$-Flow Matching Vector Field | $\tilde{z}_t=\frac{z_t}{\alpha_t+\sigma_t}$ and $\eta_t = \frac{\sigma_t}{\alpha_t+\sigma_t}$ |

如果我们把网络输出设置为$\hat{u}$，并让$\alpha_t=1-t$、$\sigma_t=t$，我们就有$\tilde{z}_t=z_t$和$\eta_t=t$，这就是流匹配更新。更准确的说，流匹配更新是采样ODE（即$dz_t=\hat{u}dt$）的欧拉采样器，并且在使用流匹配噪声调度时，
$$
\text{Diffusion with DDIM Sampler == Flow Matching Sampler (Euler)}
$$
关于DDIM采样器的其他说明：

1. DDIM采样器在网络的输出随时间恒定时，能够解析地整合重参数化采样常微分方程（即$d\tilde{z}_t=[\text{Network Output}]\cdot d\eta_t$）。当然，网络的预测并非恒定值，但这意味着DDIM采样器的误差仅仅来自于对玩过输出不可解积分的近似（与概率流常微分方程的Euler采样器不同，其涉及一个额外的线性项$z_t$）。DDIM可以被视为重参数化采样ODE的一阶欧拉采样器，它对不同网络输出的更新规则相同。然而如果使用更高阶的 ODE 求解器，网络输出可能会有所不同，这意味着Flow Matching提出的$\hat{u}$输出可能与扩散模型有所不同。

2. DDIM采样器对于应用于噪声调度$\alpha_t$和$\sigma_t$的线性缩放是不变的，因为缩放不会影响$\tilde{z}_t$和$\eta_t$。这雨其他采样器来说并不成立，例如概率流ODE的Euler采样器。

## Training

扩散模型通过估计$\hat{x}=\hat{x}(z_t;t)$或使用神经网络估计$\hat{\epsilon}=\hat{\epsilon}(z_t;t)$进行训练。通过最小化加权均方误差(MSE)损失来学习模型：
$$
\mathcal{L}(x) = \mathbb{E}_{t\sim\mathcal{U}(0,1),\epsilon\sim\mathcal{N}(0,I)}\left[ \omega(\lambda_t)\cdot\frac{d\lambda}{dt}\cdot\left\|\hat{\epsilon}-\epsilon\right\|_2^2 \right]
$$
其中$\lambda_t = \log\left(\frac{\alpha_t^2}{\sigma_t^2}\right)$是信噪比，$\omega(\lambda_t)$是加权函数，平衡不同噪声水平下损失的重要性。训练目标中的项$\frac{d\lambda}{dt}$看起来不自然，在文献中通常和加权函数合并。然而，它们的分离有助于清晰地分离训练噪声调度和加权函数的因子，并有助于强调更重要的设计选择：加权函数。

流匹配也适用于以上训练目标，回想一下条件流匹配目标：
$$
\mathcal{L}_{CFM}(x) = \mathbb{E}_{t\sim\mathcal{U}(0,1),\epsilon\sim\mathcal{N}(0,I)}\left[\left\|\hat{u}-u\right\|_2^2\right]
$$
由于$\hat{u}$可以表示为$\hat{\epsilon}$和$z_t$的线性组合，CFM训练目标可以重写为对$\epsilon$的均方误差，并具有特定的加权。

### How to choose what the network should output?

下面我们总结了几种文献中提出的网络输出，包括扩散模型使用的几个版本和流匹配模型使用的版本。在当前数据$z_t$的条件下，它们可以相互推导。在文献中，可以看到针对不同网络输出的均方误差（MSE）定义的训练目标。从训练目标的角度来看，它们都对应于在$\epsilon$-MSE 前面有额外的权重，这个权重可以吸收到权重函数中。

| Network Output                       |                       Formulation                       |                                        MSE on Network Output |
| :----------------------------------- | :-----------------------------------------------------: | -----------------------------------------------------------: |
| $\hat{\epsilon}$-Prediction          |                    $\hat{\epsilon}$                     |                 $\left\|\hat{\epsilon}-\epsilon\right\|_2^2$ |
| $\hat{x}$-Prediction                 | $\hat{x} = \frac{z_t-\sigma_t\hat{\epsilon}}{\alpha_t}$ | $\left\|\hat{x}-x\right\|_2^2=e^{-\lambda}\left\|\hat{\epsilon}-\epsilon\right\|_2^2$ |
| $\hat{v}$-Prediction                 |    $\hat{v}=\alpha_t\hat{\epsilon}-\sigma_t\hat{x}$     | $\left\|\hat{v}-v\right\|_2^2=\alpha_t^2\left(e^{-\lambda}+1\right)^2\left\|\hat{\epsilon}-\epsilon\right\|_2^2$ |
| $\hat{u}$-Flow Matching Vector Field |            $\hat{u}=\hat{\epsilon}-\hat{x}$             | $\left\|\hat{u}-u\right\|_2^2=\left(e^{-\frac{\lambda}{2}}+1\right)^2\left\|\hat{\epsilon}-\epsilon\right\|_2^2$ |

然而在实践中，模型输出可能会有所不同。例如

- $\hat{\epsilon}$-Prediction在高噪声水平下也可能有问题，因为$\hat{\epsilon}$的任何误差都会在$\hat{x} = \frac{z_t-\sigma_t\hat{\epsilon}}{\alpha_t}$中被放大，因为$\alpha_t$接近于0。这意味着在某些权重下，小的变化会导致大的损失。
- 由于类似的原因在，在低噪声水平下，$\hat{x}$-Prediction是有问题的，因为当添加的噪声较小时，$x$作为目标并不具有信息量，而误差会在$\hat{\epsilon}$中被放大。

因此，一种启发式方法是选择一个网络输出，它是$\hat{x}$-Prediction和$\hat{\epsilon}$-Prediction的组合，这适用于$\hat{v}$-Prediction和流匹配向量场$\hat{u}$。

### Summary

关于扩散模型/流匹配训练的一些要点：

1. 权重等价性：权重函数对训练很重要，它平衡了感知数据不同频率分量的重要性。流匹配权重恰好与文献中常用的扩散训练权重相匹配。
2. 训练噪声调度的重要性不大：噪声调度对训练目标影响很小，但会影响训练效率。
3. 网络输出的差异：流匹配提出的网络输出是新的，它很好地平衡了$\hat{x}$-Prediction和$\hat{\epsilon}$-Prediction，类似于$\hat{v}$-Prediction。

