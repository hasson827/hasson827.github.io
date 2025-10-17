---
title: "Variational Flow Matching for Graph Generation"
mathjax: true
layout: post
categories: Generative Modeling
---

# Variational Flow Matching for Graph Generation

## 变分流匹配的基本概念

在流匹配中，向量场可以表示为条件轨迹的期望：

$$
u_t(x) = \int u_t(x|x_1)p_t(x_1|x)dx_1 = E_{p_t(x_1|x)}[u_t(x|x_1)]
$$
其中 $p_t(x_1|x)$ 是后验概率路径，即通过点 $x$ 在时间 $t$ 的路径可能的终点 $x_1$ 的分布。

**定理1及其推导**：假设条件向量场 $u_t(x|x_1)$ 在 $x_1$ 上线性。那么，对于任何分布 $r_t(x_1|x)$ ，如果其边际分布与 $p_t(x_1|x)$ 相同，则对应的 $u_t(x|x_1)$ 期望相等。

应用线性条件，将条件向量场重写为：
$$
u_t(x|x_1) = A_t(x)x_1 + b_t(x)
$$
其中 $A_t(x): [0, 1] \times \mathbb{R}^D \to \mathbb{R}^{D \times D}$ 和 $b_t(x): [0, 1] \times \mathbb{R}^D \to \mathbb{R}^D$ 。

然后，将其代入期望计算：
$$
E_{p_t(x_1|x)}[u_t(x|x_1)] = E_{p_t(x_1|x)}[A_t(x)x_1 + b_t(x)] = A_t(x)E_{p_t(x_1|x)}[x_1] + b_t(x)
$$
我们知道，对于任何分布 $r$ ，如果其边际分布与 $p$ 相同，则有：
$$
E_r[x] = E_p[x]
$$
将此事实应用于上式：
$$
E_{p_t(x_1|x)}[u_t(x|x_1)] = A_t(x)E_{r_t(x_1|x)}[x_1] + b_t(x) = E_{r_t(x_1|x)}[A_t(x)x_1 + b_t(x)] = E_{r_t(x_1|x)}[u_t(x|x_1)]
$$
这表明，只要边际分布匹配，即使高维分布不同，期望的条件向量场也会相同。

## 变分流匹配的变分视角

论文提出了一种变分视角：不是直接预测向量场，而是定义一个基于变分分布 $q_t^\theta(x_1|x)$ 的近似向量场：

$$
v_t^\theta(x) := \int u_t(x|x_1)q_t^\theta(x_1|x)dx_1
$$
显然，当 $q_t^\theta(x_1|x)$ 和 $p_t(x_1|x)$ 相同时， $v_t^\theta(x)$ 将等于 $u_t(x)$ 。

接着，论文定义了变分流匹配问题：最小化从 $p_t$ 到 $q_t^\theta$ 的KL散度：

$$
E_t[KL(p_t(x_1|x) \| q_t^\theta(x_1|x))] = -E_{t,x,x_1}[\log q_t^\theta(x_1|x)] + \text{const}
$$
其中 $t \sim \text{Uniform}(0, 1)$ 且 $x, x_1 \sim p_t(x, x_1)$ 。

**VFM目标的推导**：
最小化KL散度等价于最大化期望对数似然：
$$
\min_\theta E_{t,x}[KL(p_t(x_1|x) \| q_t^\theta(x_1|x))] = \max_\theta E_{t,x,x_1}[\log q_t^\theta(x_1|x)]
$$
首先，将KL散度重写为熵和交叉熵的组合：
$$
E_{t,x}[KL(p_t(x_1|x) \| q_t^\theta(x_1|x))] = E_{t,x}[H(p_t(x_1|x), q_t^\theta(x_1|x))] - E_{t,x}[H(p_t(x_1|x))]
$$
其中 $H(p, q)$ 是交叉熵， $H(p)$ 是熵。我们观察到熵项 $H(p_t(x_1|x))$ 不依赖于参数 $\theta$ ，因此在优化时可以忽略。

交叉熵项可以重写为：
$$
E_{t,x}[H(p_t(x_1|x), q_t^\theta(x_1|x))] = -E_{t,x,x_1}[\log q_t^\theta(x_1|x)]
$$
因此，最小化KL散度等价于最大化 $E_{t,x,x_1}[\log q_t^\theta(x_1|x)]$ ，即：
$$
L_{VFM}(\theta) = -E_{t,x,x_1}[\log q_t^\theta(x_1|x)]
$$

## 均值场变分流匹配

在条件向量场 $u_t(x|x_1)$ 在 $x_1$ 上线性的假设下，可以使用完全分解的变分近似：

$$
q_t^\theta(x_1|x) := \prod_{d=1}^D q_t^\theta(x_{1d}|x)
$$
VFM目标简化为：

$$
L_{MF-VFM}(\theta) = -E_{t,x,x_1}[\log q_t^\theta(x_1|x)] = -E_{t,x,x_1}[\sum_{d=1}^D \log q_t^\theta(x_{1d}|x)]
$$
**向量场计算的推导**：当条件向量场线性时：
$$
u_t(x|x_1) = A_t(x)x_1 + b_t(x)
$$
则近似向量场为：
$$
v_t^\theta(x) = E_{q_t^\theta(x_1|x)}[A_t(x)x_1 + b_t(x)] = A_t(x)E_{q_t^\theta(x_1|x)}[x_1] + b_t(x)
$$
对于标准的流匹配情况，使用线性插值的条件向量场：
$$
u_t(x|x_1) = \frac{x_1 - x}{1 - t}
$$
近似向量场可以表示为：
$$
v_t^\theta(x) = E_{q_t^\theta(x_1|x)}[\frac{x_1 - x}{1 - t}] = \frac{E_{q_t^\theta(x_1|x)}[x_1] - x}{1 - t} = \frac{\mu_1 - x}{1 - t}
$$
其中 $\mu_1 := E_{q_t^\theta(x_1|x)}[x_1]$ 是变分分布的期望。

## CatFlow：分类数据的变分流匹配

对于分类数据，CatFlow使用参数化的变分分布 $q_t^\theta(x_{1d}|x) = \text{Cat}(x_{1d}|\theta_t^d(x))$ ，其中 $\theta_t^{dk}(x) := q_t(x_{1d} = k|x)$ 表示第 $d$ 个变量取第 $k$ 类的概率。

**CatFlow目标的推导**：
MF-VFM目标可以明确写出：
$$
\log q_t^\theta(x_{1d}|x) = \log \prod_{k=1}^{K_d} (\theta_t^{dk}(x))^{I[x_{1d}=k]} = \sum_{k=1}^{K_d} I[x_{1d} = k] \log \theta_t^{dk}(x)
$$
因此，CatFlow目标是标准的交叉熵损失：
$$
L_{\text{CatFlow}}(\theta) = -E_{t,x,x_1} \left[ \sum_{d=1}^D \sum_{k=1}^{K_d} I[x_{1d} = k] \log \theta_t^{dk}(x) \right]
$$
计算向量场时，可以高效计算：
$$
E_{q_t^\theta(x_{1d}|x)}[u_t(x_d|x_{1d})] = E_{q_t^\theta(x_{1d}|x)}[\frac{x_{1d} - x_d}{1 - t}] = \frac{E_{q_t^\theta(x_{1d}|x)}[x_{1d}] - x_d}{1 - t} = \frac{\theta^d(x) - x_d}{1 - t}
$$
其中 $\theta^d(x) := E_{q_t^\theta(x_{1d}|x)}[x_{1d}]$ 是分类分布的期望。

## 与流匹配的联系

假设条件向量场 $u_t(x|x_1)$ 在 $x_1$ 上线性且形式为
$$
u_t(x|x_1) = A_t(x)x_1 + b_t(x)
$$
且变分分布满足高斯分布形式
$$
q_t^\theta(x_1|x) = \mathcal{N}(x_1|\mu_t^\theta(x), \Sigma_t(x))
\\[10pt]
\Sigma_t(x) = \frac{1}{2}(A_t^\top(x)A_t(x))^{-1}
$$
将假设形式代入VFM目标：
$$
\begin{aligned}
L_{VFM}(\theta) &= -E_{t,x,x_1}[\log q_t^\theta(x_1|x)]
\\[10pt]
&= -E_{t,x,x_1}[\log((2\pi)^{-D/2} |\Sigma_t(x)|^{-1/2} \exp(-\frac{\|A_t(x)(x_1 - \mu_t^\theta(x))\|^2}{2}))]
\\[5pt]
&= E_{t,x,x_1}[\frac{|A_t(x)(x_1 - \mu_t^\theta(x))\|^2}{2}] + \frac{1}{2}E_{t,x,x_1}[D \log(2\pi) + \log |\Sigma_t(x)|]
\\[10pt]
&= E_{t,x,x_1}[\frac{|(A_t(x)x_1 + b_t(x)) - (A_t(x)\mu_t^\theta(x) + b_t(x))\|^2}{2}] + C
\\[10pt]
&= E_{t,x,x_1}[\|u_t(x|x_1) - v_t^\theta(x)\|^2_2]
\end{aligned}
$$
这正是流匹配的目标函数，表明标准流匹配是VFM的一个特例。

## 与基于分数的模型的联系

在基于分数的模型中，目标是近似分数函数 $\nabla_x \log p_t(x)$ 。论文表明分数函数也可以表示为关于 $p_t(x_1|x)$ 的期望：

**分数函数的推导**：
$$
\begin{aligned}
\nabla_x \log p_t(x) &= \frac{1}{p_t(x)}\nabla_x p_t(x)
\\[10pt]
&= \frac{1}{p_t(x)}\nabla_x \int p_t(x|x_1)p(x_1)dx_1
\\[10pt]
&= \frac{1}{p_t(x)} \int p(x_1)\nabla_x p_t(x|x_1)dx_1
\\[10pt]
&= \int \frac{p_t(x|x_1)p(x_1)}{p_t(x)} \nabla_x \log p_t(x|x_1)dx_1
\\[10pt]
&= \int p_t(x_1|x)\nabla_x \log p_t(x|x_1)dx_1
\\[10pt]
&= E_{p_t(x_1|x)}[\nabla_x \log p_t(x|x_1)]
\end{aligned}
$$
类似地，可以将 $s_t^\theta(x)$ 参数化为关于变分近似 $q_t^\theta(x_1|x)$ 的期望：
$$
s_t^\theta(x) := \int q_t^\theta(x_1|x)\nabla_x \log p_t(x|x_1) dx_1
$$
当 $q_t^\theta(x_1|x) = p_t(x_1|x)$ 时， $s_t^\theta(x) = \nabla_x \log p_t(x)$ 。

论文还构造了随机生成动态：
$$
dx = \tilde{v}_t^\theta(x)dt + g_t dw
$$
其中
$$
\tilde{u}_t(x) := E_{p_t(x_1|x)}[u_t(x|x_1) + \frac{g_t^2}{2} \nabla_x \log p_t(x|x_1)], \quad \tilde{v}_t^\theta(x) := v_t^\theta(x) + \frac{g_t^2}{2} s_t^\theta(x)
$$
**定理4的推导**：
将变分流匹配目标重写为：
$$
L_{VFM}(\theta) = E_t[L_\theta(t, x)] \quad \text{其中} \quad L_\theta(t, x) = -E_{x_1}[\log q_t^\theta(x_1|x)]
$$
考虑两个随机过程：
$$
dx = \tilde{u}_t(x)dt + g_t dw, \quad dx = \tilde{v}_t^\theta(x)dt + g_t dw
$$
它们都从相同的先验分布 $p_0(x)$ 开始。第一个过程生成概率路径 $p_t(x)$ 并以数据分布 $p_{data}(x) = p_1(x)$ 结束。第二个过程生成依赖于变分分布 $q_t^\theta(x_1|x)$ 的概率路径 $q_t^\theta(x)$ 。

应用相关结果：
$$
\begin{aligned}
KL(p_1(x_1) \| q_1^\theta(x_1)) &\leq E_{t,x}[\frac{1}{2g_t^2} \|\tilde{u}_t(x) - \tilde{v}_t^\theta(x)\|^2]
\\[10pt]
&= E_{t,x}[\frac{1}{2g_t^2} |\int(p_t(x_1|x) - q_t^\theta(x_1|x)) \tilde{u}_t(x|x_1)dx_1|^2]
\\[10pt]
&\leq E_{t,x}[\frac{1}{2g_t^2}(\int |p_t(x_1|x) - q_t^\theta(x_1|x)| \cdot \|\tilde{u}_t(x|x_1)\|dx_1)^2]
\end{aligned}
$$
引入辅助函数：
$$
l_t(x) = \sup_{x_1} \|\tilde{u}_t(x|x_1)\|, \qquad \lambda_t(x) = \frac{l_t(x)}{g_t^2}
$$
得到：
$$
KL(p_1(x_1) \| q_1^\theta(x_1)) \leq E_{t,x}[\frac{\lambda_t(x)}{2}(\int |p_t(x_1|x) - q_t^\theta(x_1|x)|dx_1)^2]
$$
应用Pinsker不等式：
$$
\int |p(x) - q(x)|dx \leq \sqrt{2KL(p \| q)}
$$
应用于内积分：
$$
KL(p_1(x_1) \| q_1^\theta(x_1)) \leq E_{t,x}[\lambda_t(x)KL(p_t(x_1|x) \| q_t^\theta(x_1|x))]
$$
将左边重写为：
$$
KL(p_1(x_1) \| q_1^\theta(x_1)) = -H(p_1(x_1)) - E_{x_1}[\log q_1^\theta(x_1)]
$$
右边重写为：
$$
E_{t,x}[\lambda_t(x)KL(p_t(x_1|x) \| q_t^\theta(x_1|x))] = -E_{t,x}[\lambda_t(x)H(p_t(x_1|x))] - E_{t,x,x_1}[\lambda_t(x)q_t^\theta(x_1|x)]
$$
因此：
$$
-E_{x_1}[\log q_{1}^{\theta}(x_1)] \leq E_{t,x}[\lambda_t(x)L_\theta(t, x)] + C
$$
其中 $C$ 是常数，这表明重加权的VFM目标提供了模型似然的上界。

## 实验结果

CatFlow不仅性能优越，还具有以下优势：
1. 添加的归纳偏置确保生成路径与真实轨迹对齐，提高了性能和收敛性
2. 使用交叉熵损失代替均方误差改善了训练过程中的梯度行为
3. 能够学习概率向量而非直接选择类别，允许在特定时间表达对变量的不确定性

此外，CatFlow满足排列等变性，即生成所有图排列的概率相等，这通过以下证明：
如果 $\theta_t(x)$ 相对于排列群 $S_{|V|}$ 是等变的，则 $v_t$ 也是等变的：
$$
v_t(\pi \cdot x) = \frac{\theta_t(\pi \cdot x) - \pi \cdot x}{1 - t} = \frac{\pi \cdot \theta_t(x) - \pi \cdot x}{1 - t} = \pi \cdot v_t(x)
$$
由于 $p_0$ 为交换分布，最终分布 $p_1$ 也保持这一特性，因此所有图排列以相等概率生成。