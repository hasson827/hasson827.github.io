---
title: "Variational Rectified Flow Matching"
mathjax: true
layout: post
categories: Generative Modeling
---

# Variational Rectified Flow Matching

---

## 预备知识

### 变分自编码器的证据下界

**符号定义**：记数据分布为 $D$，对于从数据集中采样的点 $x_1$，其真实边际概率为 $p(x_1)$。假设我们引入潜变量 $z$，先验分布为 $p(z)$，观测模型（解码器）为 $p_\theta(x_1|z)$。我们还定义近似后验（识别模型） $q_\phi(z|x_1)$。

记 $\mathbb{E}_{x_1\sim D}$ 为对数据分布 $D$ 的期望，$\mathbb{E}_{z\sim q_\phi(z|x_1)}$ 为对 $z$ 的期望，$D_{KL}(q|p)$ 为Kullback–Leibler 散度。

**ELBO推导**：根据经典的变分自编码器理论，通过应用 **Jensen 不等式** 或变分不等式，可以得到下界：
$$
\mathbb{E}_{x_1\sim D}\left[\log{p(x_1)}\right] \ge \mathbb{E}_{x_1\sim D}\left[ \mathbb{E}_{z\sim q_{\phi}(z|x_1)}\left[ \log{p_{\theta}(x_1|z)}\right] - D_{KL}\left( q_{\phi}(z|x_1)||p(z) \right)  \right]
$$
其中内部的 $\mathbb{E}_{z\sim q*\phi}[\log p_\theta(x_1|z)]$ 称为重建项，$D_{KL}(\cdot|\cdot)$ 是正则化项。该公式表明最大化数据对数似然 $\log p(x_1)$ 的下界等价于优化上述两项之和。

### 流匹配基本公式

在**流匹配**中，假设有源分布 $p_0(x_0)$ 和目标分布 $p_1(x_1)$。我们定义如下符号：

- $x_0 \sim p_0$ 为源域采样，$x_1 \sim p_1$ 为目标域采样。
- 时间变量 $t\in[0,1]$，数据位置为 $x_t$。
- 参数化的速度场 $\displaystyle v_\theta(x_t,t)$ 表示在 $(x_t,t)$ 处的速度（由可学习的网络参数 $\theta$ 确定）。
- $\mathrm{div}\,v = \sum_i \frac{\partial v_i}{\partial x_i}$ 表示向量场的**散度（divergence）**。

流匹配使用**常微分方程（ODE）**将源样本推向目标样本：$x_0$ 作为初始条件，按照速度场 $v_\theta$ 做演化。对于从源分布 $x_0$ 演化至终点 $x_1$，其对数概率的变化由**瞬时变换公式**给出：
$$
\log{p_1(x_1)} = \log{p_0(x_0)}+\int_0^1\mathrm{div}\,v_{\theta}(x_t,t)dt
$$
这里 $p_1(x_1)$ 是最终时刻 $t=1$ 的概率密度，$p_0(x_0)$ 是初始时刻 $t=0$ 的概率密度。该公式来源于连续可逆变换的积分公式，用以评估变分生成过程的边际似然。

进一步，对时间导数求偏导可以得到相应的连续性方程（Transport PDE）：
$$
\frac{\partial}{\partial t}\log{p_t(x_t)} = -\mathrm{div}\,v_{\theta}(x_t,t)
$$
这里 $p_t(x)$ 表示时间 $t$ 时刻的分布密度，该方程说明了速度场 $v_\theta$ 与分布演化之间的关系：分布密度随时间的变化率等于负的散度。

### 经典整流流匹配的训练目标

在**整流流匹配**（Rectified Flow Matching）中，训练时我们随机配对一个源样本 $x_0$ 和一个目标样本 $x_1$，并令两者在数据空间中沿直线相连：即定义 $x_t=(1-t)x_0 + t x_1$。此时“真实”的速度为
$$
v(x_0,x_1,t) = \frac{\partial}{\partial t}[(1-t)x_0+tx_1] = x_1-x_0
$$
经典流匹配的方法是让网络预测的速度场 $v_\theta(x_t,t)$ 逼近这一目标速度，用均方误差损失：
$$
\min_{\theta}\mathbb{E}_{t,x_0,x_1}\left[\left\| v_{\theta}(x_t,t)-(x_1-x_0) \right\|_2^2\right]
$$
也就是说，我们通过最小化 $|v_\theta - v(x_0,x_1,t)|^2$ 使网络学习速度场。如上所述，多对 $(x_0,x_1)$ 的线性连接造成了相同 $(x_t,t)$ 点上可能存在多个不同目标速度，从而经典方法往往学习到各真速度的**平均值**。

## 变分整流流匹配的推导

### 常数速度场的分析

为了理解整流流匹配，论文提出一个简单分析：假设源分布和目标分布都是单个高斯，速度场为常数。具体地，令
$$
\tilde{p}_0(\xi_0) = \mathcal{N}(\xi_0;x_0,I)
\\
\tilde{p}_1(\xi_1) = \mathcal{N}(\xi_1;x_1,I)
$$
其中 $\xi_0,\xi_1$ 是变量，$x_0,x_1$ 分别是均值。假设速度场 $v_\theta(\xi_t,t)=\theta$ 为常矢量，不随位置改变。对于偏微分方程 (3) 和积分形式 (2) ，取$\theta = x_1 - x_0$可以满足这些方程，而且此时解为线性插值：$x_t = (1-t)x_0 + t x_1$。其证明步骤如下：

1. **常数速度下散度为 0**：由于 $v_\theta(\xi_t,t)=\theta$ 恒定，故 $\mathrm{div}\,v_\theta(\xi_t,t) = 0$。于是在 (2) 中有
   $$
   \int_0^1\mathrm{div}\,v_{\theta}(x_t,t)dt=0
   $$

2. **对数密度相等**：将 (2) 中的概率密度替换为高斯形式（省略归一化常数），得到
   $$
   \log{\tilde{p}_1(\xi_1)}-\log{\tilde{p}_0(\xi_0)} = \frac{1}{2}\left( \left\| \xi_0-x_0 \right\|_2^2-\left\| \xi_1-x_1 \right\|_2^2 \right) = 0
   $$
   即
   $$
   \forall\xi_0,\xi_1\qquad\left\| \xi_0-x_0 \right\|_2^2-\left\| \xi_1-x_1 \right\|_2^2=0
   $$

3. **利用速度关系**：由于 $\xi_t$ 从 $\xi_0$ 沿速度 $\theta$ 移动到 $\xi_1$，有 $\xi_1 = \xi_0 + \int_0^1 v_\theta(\xi_t,t),dt = \xi_0 + \theta$。 将其代入上式，得到
   $$
   \forall\xi_0\qquad\left\| \xi_0-x_0 \right\|_2^2-\left\| \xi_0+\theta-x_1 \right\|_2^2=0
   $$
   展开并整理为
   $$
   \forall\xi_0\qquad(x_1-x_0-\theta)(2\xi_0-x_0-x_1+\theta)=0
   $$
   由于等式需对任意 $\xi_0$ 成立，必须有 $x_1 - x_0 - \theta = 0$。因此得到**结论**：
   $$
   \theta=x_1-x_0\quad且\quad\xi_t=\xi_0+t\theta=x_0+t(x_1-x_0)=(1-t)x_0+tx_1
   $$

此推导说明：在高斯假设和恒定速度场下，最佳的速度参数为 $\theta = x_1 - x_0$，对应于样本 $x_0$ 到 $x_1$ 的线性流。该结果也直观地支持了经典整流流匹配中使用线性插值的合理性。

### 经典目标的概率视角和单模局限

以更概率的角度看问题，我们可将真实速度视为**观测数据**：令变量 $v = x_1 - x_0$ 表示真实速度，则在给定 $(x_t,t)$ 的条件下，我们可以假设模型对速度服从高斯分布
$$
p(v|x_t,t)=\mathcal{N}(v;v_{\theta}(x_t,t),I)
$$
其中均值由网络 $v_\theta(x_t,t)$ 给出，协方差取为单位。这样最大化数据对数似然 $\log p(v|x_t,t)$ 等价于最大化高斯似然，相应的负对数似然为平方误差。具体地，我们有
$$
\mathbb{E}_{t,x_0,x_1}\left[ \log{p(x_1-x_0|x_t,t)} \right]\propto-\mathbb{E}_{t,x_0,x_1}\left[\left\| v_{\theta}(x_t,t)-(x_1-x_0) \right\|_2^2\right]
$$
其中等号“$\propto$”忽略了常数项，该表达式与前述最小化均方误差目标等价。该解释强调了**单峰高斯**模型的隐含假设：模型试图学习一个**高斯分布的均值**来匹配真实速度。这导致在出现多模态速度场时模型只能学习到平均值，从而不能表达多样性。

### 引入潜变量进行变分扩展

为了捕捉多模态的速度分布，论文引入了潜在变量 $z$。假设潜变量 $z$ 服从先验分布 $p(z)$，并令条件速度模型为
$$
p(v|x_t,t,z)=\mathcal{N}(v;v_{\theta}(x_t,t,z),I)
$$
其中均值现在依赖于 $z$。于是对速度的边缘分布为
$$
p(v|x_t,t) = \int p(v|x_t,t,z)p(z)dz
$$
这是一个多模态高斯混合（latent 为调制因子）。训练时，我们对未知的 $z$ 采用**变分近似**：引入识别网络（encoder）$q_\phi(z\mid x_0,x_1,x_t,t)$ 来逼近后验。

根据变分下界原理，对单个速度样本的对数边缘概率进行下界：
$$
\log{p(v|x_t,t)}\ge\mathbb{E}_{z\sim q_{\phi}(z|x_0,x_1,x_t,t)}\left[ \log{p(v|x_t,t,z)} \right]-D_{KL}\left(q_{\phi}(z|x_0,x_1,x_t,t)||p(z)\right)
$$
此处，$\mathbb{E}_{z\sim q_\phi}[\log p(v|x_t,t,z)]$ 是对条件似然的期望，减去识别分布与先验的 KL 散度。这是标准的变分下界（ELBO）形式。

### 变分流匹配目标的推导

将 (17) 中的下界代回经典目标 (14) 中的对数似然，我们得到整个模型的训练目标。

令 $v = x_1 - x_0$ 为真实速度样本，使用式 (17) 的右端替换 (14) 中的 $\log p(v|x_t,t)$。由于 (14) 中的期望在 $t,x_0,x_1$ 上，因此应用下界后得到：
$$
\begin{aligned}
\mathbb{E}_{t,x_0,x_1}\left[ \log{p(x_1-x_0|x_t,t)} \right]&\ge\mathbb{E}_{t,x_0,x_1}\left[ \mathbb{E}_{z\sim q_{\phi}(z|\cdot)}\left[ \log{p(v|x_t,t,z)} \right]-D_{KL}\left(q_{\phi}(z|\cdot)||p(z)\right) \right]
\\[10pt]
&=\mathbb{E}_{t,x_0,x_1}\left[ -\mathbb{E}_{z\sim q_{\phi}}\left[\left\| v_{\theta}(x_t,t,z)-(x_1-x_0) \right\|_2^2\right] -D_{KL}\left(q_{\phi}(z|\cdot)||p(z)\right) \right]
\end{aligned}
$$
这里第二步利用了对数高斯密度 $\log \mathcal{N}(v;v_\theta, I) \propto -|v_\theta - v|^2$ 的性质。给出了这个不等式的形式。

因此，变分整流流匹配的目标是在 (18) 的右端最大化（相当于最小化负目标）。该目标包含两部分：**数据拟合项** $-\mathbb{E}_{z}[|v_\theta(x_t,t,z)-(x_1-x_0)|^2]$ 和 **KL 正则项** $-D_{KL}(q_\phi|p)$。直观上，前者让网络的条件速度均值接近真实速度，后者让识别分布 $q_\phi$ 接近先验 $p(z)$。所得目标将潜变量引入到流匹配中，以捕捉同一时空点下的多模态速度分布。需要注意，如果不引入 $z$（即 $z$ 服从退化的点分布），上式就退化为经典的均方误差目标 (14)。

### 损失优化和假设条件

在优化 (18) 时，通常令先验 $p(z) = \mathcal{N}(0,I)$，并将识别网络 $q_\phi(z|x_0,x_1,x_t,t)$ 设为高斯分布，其均值 $\mu_\phi$ 和对数方差由神经网络输出。这样 $D_{KL}$ 可以解析计算。网络训练使用重参数化技巧对 $z$ 进行采样，然后使用单样本估计近似期望。在算法中，每个小批次采样 ${x_0,x_1,t}$ 后，计算
$$
z=\mu_{\phi}(x_0,x_1,x_t,t)+\epsilon\sigma_{\phi}(x_0,x_1,x_t,t),\quad\epsilon\sim\mathcal{N}(0,I)
$$
并按照式 (18) 构造损失。这种优化过程与标准 VAE 非常类似，仅增加了对速度场的依赖。