---
chapter: 10
title: Actor-Critic Methods
key_topics: [actor-critic structure, QAC, baseline invariance, optimal baseline, advantage function, advantage actor-critic (A2C), TD actor-critic, importance sampling, importance weight, off-policy policy gradient theorem, off-policy actor-critic, deterministic policy, deterministic policy gradient theorem, deterministic actor-critic (DPG), DDPG, on-policy vs off-policy, variance reduction]
depends_on: [1, 2, 3, 6, 7, 8, 9]
required_by: []
---

# Chapter 10: Actor-Critic Methods

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 10, pp. 215-236
> Supplemented by: Lecture slides L10 (55 slides)
> Errata: Section 10.3.3 -- In the off-policy actor-critic update rule, $\theta$ should be $\theta_t$ (corrected inline below)

## Purpose and Context

This chapter introduces **actor-critic methods**, the final class of algorithms in the book. From one viewpoint, "actor-critic" refers to a structure that incorporates both **policy-based** (actor) and **value-based** (critic) methods. From another viewpoint, actor-critic methods are still **policy gradient algorithms** -- they are obtained by extending the policy gradient algorithm from Chapter 9 with TD-based value estimation.

**Position in the book**: Chapter 10 is the culmination of the book, sitting at the intersection of policy-based methods (Chapter 9) and value-based methods (Chapter 8). The book's overall progression is: *with model* -> *without model*; *tabular representation* -> *function representation*; *value-based* -> *policy-based* -> **combined (actor-critic)**.

**Key definitions** (from lecture slides):
- **Actor**: The policy update step. Called "actor" because the policy determines which actions the agent takes.
- **Critic**: The value estimation/policy evaluation step. Called "critic" because it evaluates (criticizes) the actor's policy by computing value estimates.

**Chapter roadmap**: The chapter presents four actor-critic algorithms of increasing sophistication:
1. **QAC** (simplest actor-critic) -- Section 10.1
2. **A2C** (advantage actor-critic) -- Section 10.2
3. **Off-policy actor-critic** (via importance sampling) -- Section 10.3
4. **DPG** (deterministic actor-critic) -- Section 10.4

---

## 10.1 The Simplest Actor-Critic Algorithm (QAC)

### Motivation: From Policy Gradient to Actor-Critic

Recall from Chapter 9 that the gradient-ascent algorithm for maximizing the scalar metric $J(\theta)$ is:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t) = \theta_t + \alpha \mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) q_\pi(S, A)\right]$$
(10.1)

where $\eta$ is a distribution of the states (see Theorem 9.1). Since the true gradient is unknown, we use a **stochastic gradient** to approximate it:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t, \theta_t) q_t(s_t, a_t)$$
(10.2)

This is the algorithm from equation (9.32) in Chapter 9.

**Key insight from the slides**: Equation (10.2) is critically important because it **directly reveals** the actor-critic structure:
- The **expression itself** (the policy parameter update) corresponds to the **actor**.
- The **algorithm that estimates** $q_t(s_t, a_t)$ corresponds to the **critic**.

### Two Ways to Estimate Action Values

Equation (10.2) requires knowing $q_t(s_t, a_t)$, an estimate of $q_\pi(s_t, a_t)$. Two approaches:

| Estimation Method | Algorithm Name | Chapter |
|---|---|---|
| Monte Carlo learning | REINFORCE (Monte Carlo policy gradient) | Chapter 9 |
| Temporal-difference (TD) learning | **Actor-critic** | Chapter 10 |

**Defining distinction**: When $q_t(s_t, a_t)$ is estimated by TD learning, the resulting algorithms are called **actor-critic**. Thus, actor-critic methods = policy gradient + TD-based value estimation.

### Algorithm 10.1: Q Actor-Critic (QAC)

**Initialization:**
- A policy function $\pi(a|s, \theta_0)$ with initial parameter $\theta_0$
- A value function $q(s, a, w_0)$ with initial parameter $w_0$
- Learning rates $\alpha_w, \alpha_\theta > 0$

**Goal:** Learn an optimal policy to maximize $J(\theta)$.

**At time step $t$ in each episode, do:**

1. Generate $a_t$ following $\pi(a|s_t, \theta_t)$, observe $r_{t+1}, s_{t+1}$, then generate $a_{t+1}$ following $\pi(a|s_{t+1}, \theta_t)$.

2. **Actor** (policy update):
$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \, q(s_t, a_t, w_t)$$

3. **Critic** (value update via Sarsa + function approximation, from equation (8.35)):
$$w_{t+1} = w_t + \alpha_w \left[r_{t+1} + \gamma q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t)\right] \nabla_w q(s_t, a_t, w_t)$$

**Remarks** (from slides):
- The critic corresponds to "Sarsa + value function approximation" (Chapter 8).
- The actor corresponds to the policy update algorithm from Chapter 9.
- Although simple, QAC reveals the **core idea** of actor-critic methods and can be extended to generate many advanced algorithms.

---

## 10.2 Advantage Actor-Critic (A2C)

The core idea of A2C is to introduce a **baseline** to reduce estimation variance.

### 10.2.1 Baseline Invariance

**Property**: The policy gradient is **invariant** to an additional baseline $b(S)$:

$$\mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \, q_\pi(S, A)\right] = \mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \left(q_\pi(S, A) - b(S)\right)\right]$$
(10.3)

where the additional baseline $b(S)$ is a scalar function of $S$.

#### Why is (10.3) valid?

Equation (10.3) holds if and only if:

$$\mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \, b(S)\right] = 0$$

**Proof**:

$$\mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \, b(S)\right] = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \pi(a|s, \theta_t) \nabla_\theta \ln \pi(a|s, \theta_t) \, b(s)$$

$$= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta_t) \, b(s) = \sum_{s \in \mathcal{S}} \eta(s) \, b(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta_t)$$

$$= \sum_{s \in \mathcal{S}} \eta(s) \, b(s) \, \nabla_\theta \sum_{a \in \mathcal{A}} \pi(a|s, \theta_t) = \sum_{s \in \mathcal{S}} \eta(s) \, b(s) \, \nabla_\theta 1 = 0$$

The key step uses the fact that $\sum_{a} \pi(a|s, \theta_t) = 1$ for any $s$, so its gradient with respect to $\theta$ is zero.

#### Why is the baseline useful?

The baseline is useful because it can **reduce the approximation variance** when using samples to approximate the true gradient. Define:

$$X(S, A) \doteq \nabla_\theta \ln \pi(A|S, \theta_t) \left[q_\pi(S, A) - b(S)\right]$$
(10.4)

The true gradient is $\mathbb{E}[X(S, A)]$. Since we use a stochastic sample $x$ to approximate $\mathbb{E}[X]$, it is favorable if $\text{var}(X)$ is small:
- If $\text{var}(X) \approx 0$, then any sample $x$ accurately approximates $\mathbb{E}[X]$.
- If $\text{var}(X)$ is large, a sample may be far from $\mathbb{E}[X]$.

**Critical observation**: $\mathbb{E}[X]$ is invariant to the baseline, but $\text{var}(X)$ is **not**. Our goal is to design a good baseline to **minimize** $\text{var}(X)$.

**In REINFORCE and QAC**: $b = 0$ (no baseline), which is not guaranteed to be good.

### Optimal Baseline (Box 10.1)

**Result**: The optimal baseline that minimizes $\text{var}(X)$ is:

$$b^*(s) = \frac{\mathbb{E}_{A \sim \pi}\left[\|\nabla_\theta \ln \pi(A|s, \theta_t)\|^2 \, q_\pi(s, A)\right]}{\mathbb{E}_{A \sim \pi}\left[\|\nabla_\theta \ln \pi(A|s, \theta_t)\|^2\right]}, \quad s \in \mathcal{S}$$
(10.5)

**Derivation sketch** (Box 10.1): When $X$ is a vector, its variance is a matrix. We select the trace of $\text{var}(X)$ as a scalar objective:

$$\text{tr}[\text{var}(X)] = \text{tr}\,\mathbb{E}[(X - \bar{x})(X - \bar{x})^T] = \mathbb{E}[X^T X] - \bar{x}^T \bar{x}$$
(10.6)

Since $\bar{x} = \mathbb{E}[X]$ is invariant, we only need to minimize $\mathbb{E}[X^T X]$:

$$\mathbb{E}[X^T X] = \mathbb{E}\left[\|\nabla_\theta \ln \pi\|^2 (q_\pi(S, A) - b(S))^2\right] = \sum_{s \in \mathcal{S}} \eta(s) \, \mathbb{E}_{A \sim \pi}\left[\|\nabla_\theta \ln \pi\|^2 (q_\pi(s, A) - b(s))^2\right]$$

Setting $\nabla_b \mathbb{E}[X^T X] = 0$ yields $b^*(s)$.

### Suboptimal but Practical Baseline

Although $b^*(s)$ is optimal, it is too complex for practical use. Removing the weight $\|\nabla_\theta \ln \pi(A|s, \theta_t)\|^2$ from (10.5) gives the **suboptimal baseline**:

$$b^\dagger(s) = \mathbb{E}_{A \sim \pi}[q_\pi(s, A)] = v_\pi(s), \quad s \in \mathcal{S}$$

**Key insight**: The suboptimal baseline is simply the **state value** $v_\pi(s)$.

### 10.2.2 Algorithm Description

#### The Advantage Function

When $b(s) = v_\pi(s)$, the gradient-ascent algorithm becomes:

$$\theta_{t+1} = \theta_t + \alpha \, \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \left[q_\pi(S, A) - v_\pi(S)\right]\right] \doteq \theta_t + \alpha \, \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \, \delta_\pi(S, A)\right]$$
(10.7)

Here:

$$\delta_\pi(S, A) \doteq q_\pi(S, A) - v_\pi(S)$$

is called the **advantage function**.

**Why "advantage"?** Since $v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s, a)$ is the **mean** of the action values:
- If $\delta_\pi(s, a) > 0$: action $a$ has a **greater** value than the mean -- it is advantageous.
- If $\delta_\pi(s, a) < 0$: action $a$ has a **lower** value than the mean -- it is disadvantageous.

#### Stochastic Version

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \left[q_t(s_t, a_t) - v_t(s_t)\right] = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \, \delta_t(s_t, a_t)$$
(10.8)

This updates the policy based on the **relative** value of $q_t$ with respect to $v_t$ rather than the **absolute** value of $q_t$.

**Intuition from slides**: Rewriting (10.8):

$$\theta_{t+1} = \theta_t + \alpha \frac{\nabla_\theta \pi(a_t|s_t, \theta_t)}{\pi(a_t|s_t, \theta_t)} \delta_t(s_t, a_t) = \theta_t + \underbrace{\alpha \frac{\delta_t(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}}_{\beta_t} \nabla_\theta \pi(a_t|s_t, \theta_t)$$

This reveals:
- Greater $\delta_t(s_t, a_t)$ $\Rightarrow$ greater $\beta_t$ $\Rightarrow$ greater $\pi(a_t|s_t, \theta_{t+1})$: actions with higher advantage get reinforced.
- Smaller $\pi(a_t|s_t, \theta_t)$ $\Rightarrow$ greater $\beta_t$ $\Rightarrow$ greater $\pi(a_t|s_t, \theta_{t+1})$: rare actions that turn out to be good get more strongly reinforced. This provides a natural **exploration-exploitation balance**.

#### TD Error Approximation of the Advantage Function

The advantage function $q_t(s_t, a_t) - v_t(s_t)$ is approximated by the **TD error**:

$$q_t(s_t, a_t) - v_t(s_t) \approx r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t)$$

This approximation is valid because:

$$q_\pi(s_t, a_t) - v_\pi(s_t) = \mathbb{E}\left[R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t) \mid S_t = s_t, A_t = a_t\right]$$

**Key benefit**: Using the TD error, we only need a **single neural network** to represent $v_\pi(s)$. Otherwise, we would need two networks for $v_\pi(s)$ and $q_\pi(s, a)$ separately.

| Estimation Method for $q_t - v_t$ | Algorithm Name |
|---|---|
| Monte Carlo learning | REINFORCE with a baseline |
| TD learning | **Advantage actor-critic (A2C)** or **TD actor-critic** |

### Algorithm 10.2: Advantage Actor-Critic (A2C) / TD Actor-Critic

**Initialization:**
- A policy function $\pi(a|s, \theta_0)$ with initial parameter $\theta_0$
- A value function $v(s, w_0)$ with initial parameter $w_0$
- Learning rates $\alpha_w, \alpha_\theta > 0$

**Goal:** Learn an optimal policy to maximize $J(\theta)$.

**At time step $t$ in each episode, do:**

1. Generate $a_t$ following $\pi(a|s_t, \theta_t)$ and then observe $r_{t+1}, s_{t+1}$.

2. **Advantage** (TD error):
$$\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$$

3. **Actor** (policy update):
$$\theta_{t+1} = \theta_t + \alpha_\theta \, \delta_t \, \nabla_\theta \ln \pi(a_t|s_t, \theta_t)$$

4. **Critic** (value update):
$$w_{t+1} = w_t + \alpha_w \, \delta_t \, \nabla_w v(s_t, w_t)$$

**Properties**:
- **On-policy**: The policy $\pi(\theta_t)$ is stochastic and hence exploratory. It can be directly used to generate experience samples **without** techniques like $\varepsilon$-greedy.
- **Variants**: Asynchronous advantage actor-critic (**A3C**) is a well-known variant.

### Comparison: QAC vs. A2C

| Property | QAC (Algorithm 10.1) | A2C (Algorithm 10.2) |
|---|---|---|
| Critic estimates | $q(s, a, w)$ (action value) | $v(s, w)$ (state value) |
| Actor uses | $q(s_t, a_t, w_t)$ | $\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$ |
| Networks needed | Policy + Q-network | Policy + V-network |
| Baseline | None ($b = 0$) | $v_\pi(s)$ (state value) |
| Variance | Higher | Lower (due to baseline) |
| Next action needed | Yes ($a_{t+1}$ for Sarsa) | No |

---

## 10.3 Off-Policy Actor-Critic

The policy gradient methods studied so far (REINFORCE, QAC, A2C) are all **on-policy**. The reason is visible from the true gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \left(q_\pi(S, A) - v_\pi(S)\right)\right]$$

To approximate this gradient with samples, we must generate action samples by following $\pi(\theta)$. Since $\pi(\theta)$ is **both** the behavior policy (used for sampling) **and** the target policy (being improved), these methods are on-policy.

**Goal of this section**: Convert the policy gradient to **off-policy** using the **importance sampling** technique, so that samples from a given behavior policy $\beta$ can be reused.

### 10.3.1 Importance Sampling

Importance sampling is a **general technique** (not restricted to RL) for estimating expected values defined over one probability distribution using samples drawn from another distribution.

#### Setup

Consider a random variable $X \in \mathcal{X}$ with probability distribution $p_0(X)$. Our goal is to estimate $\mathbb{E}_{X \sim p_0}[X]$ using i.i.d. samples $\{x_i\}_{i=1}^n$.

**Case 1** -- Samples from $p_0$: If $\{x_i\}$ are generated by $p_0$, then $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is an unbiased estimate of $\mathbb{E}_{X \sim p_0}[X]$ with variance converging to zero as $n \to \infty$ (law of large numbers).

**Case 2** -- Samples from a different distribution $p_1$: If $\{x_i\}$ are generated by $p_1 \neq p_0$, then $\bar{x} \approx \mathbb{E}_{X \sim p_1}[X] \neq \mathbb{E}_{X \sim p_0}[X]$ in general. However, **importance sampling** allows us to still estimate $\mathbb{E}_{X \sim p_0}[X]$.

#### The Importance Sampling Formula

$$\mathbb{E}_{X \sim p_0}[X] = \sum_{x \in \mathcal{X}} p_0(x) \, x = \sum_{x \in \mathcal{X}} p_1(x) \underbrace{\frac{p_0(x)}{p_1(x)} x}_{f(x)} = \mathbb{E}_{X \sim p_1}[f(X)]$$
(10.9)

Thus, estimating $\mathbb{E}_{X \sim p_0}[X]$ becomes the problem of estimating $\mathbb{E}_{X \sim p_1}[f(X)]$. Let:

$$\bar{f} \doteq \frac{1}{n} \sum_{i=1}^n f(x_i)$$

Since $\bar{f}$ approximates $\mathbb{E}_{X \sim p_1}[f(X)]$, it follows from (10.9) that:

$$\mathbb{E}_{X \sim p_0}[X] = \mathbb{E}_{X \sim p_1}[f(X)] \approx \bar{f} = \frac{1}{n} \sum_{i=1}^n \underbrace{\frac{p_0(x_i)}{p_1(x_i)}}_{\text{importance weight}} x_i$$
(10.10)

**Interpretation of the importance weight** $\frac{p_0(x_i)}{p_1(x_i)}$:
- When $p_1 = p_0$: the weight is 1, and $\bar{f}$ reduces to $\bar{x}$.
- When $p_0(x_i) \geq p_1(x_i)$: sample $x_i$ is more frequently drawn by $p_0$ than $p_1$. The weight (greater than 1) **emphasizes** this sample's importance.

**Why not compute $\mathbb{E}_{X \sim p_0}[X]$ directly from $p_0$?** To use the definition $\mathbb{E}_{X \sim p_0}[X] = \sum_{x} p_0(x) x$, we need $p_0(x)$ for **every** $x \in \mathcal{X}$. This is difficult when $p_0$ is represented by a neural network or when $\mathcal{X}$ is large. By contrast, (10.10) only requires $p_0(x_i)$ for a few samples.

#### Illustrative Example

Consider $X \in \mathcal{X} = \{+1, -1\}$ with:

$$p_0(X = +1) = 0.5, \quad p_0(X = -1) = 0.5$$

so $\mathbb{E}_{X \sim p_0}[X] = 0$. Suppose $p_1$ is:

$$p_1(X = +1) = 0.8, \quad p_1(X = -1) = 0.2$$

so $\mathbb{E}_{X \sim p_1}[X] = 0.6$.

Given samples $\{x_i\}$ drawn from $p_1$ (more $+1$s than $-1$s):
- **Direct average** $\frac{1}{n}\sum x_i$ converges to $\mathbb{E}_{X \sim p_1}[X] = 0.6$ (incorrect for our purpose).
- **Importance-weighted average** $\frac{1}{n}\sum \frac{p_0(x_i)}{p_1(x_i)} x_i$ converges to $\mathbb{E}_{X \sim p_0}[X] = 0$ (correct).

**Worked calculation for a single sample**: If $x_i = +1$, the importance weight is $\frac{p_0(+1)}{p_1(+1)} = \frac{0.5}{0.8} = 0.625$, so the weighted sample is $0.625 \times 1 = 0.625$. If $x_i = -1$, the weight is $\frac{0.5}{0.2} = 2.5$, so the weighted sample is $2.5 \times (-1) = -2.5$. The negative samples are weighted more heavily, pulling the average toward 0.

#### Requirement on $p_1$

The distribution $p_1$ must satisfy: $p_1(x) \neq 0$ whenever $p_0(x) \neq 0$.

**Counterexample**: If $p_1(X = +1) = 1$ and $p_1(X = -1) = 0$, then all samples are $+1$. The importance-weighted average becomes:

$$\frac{1}{n}\sum_{i=1}^n \frac{p_0(+1)}{p_1(+1)} \cdot 1 = \frac{1}{n} \sum_{i=1}^n \frac{0.5}{1} \cdot 1 \equiv 0.5 \neq 0$$

No matter how large $n$ is, the estimate is wrong.

### 10.3.2 The Off-Policy Policy Gradient Theorem

Suppose $\beta$ is a behavior policy. Our goal is to use samples generated by $\beta$ to learn a target policy $\pi$ that maximizes:

$$J(\theta) = \sum_{s \in \mathcal{S}} d_\beta(s) v_\pi(s) = \mathbb{E}_{S \sim d_\beta}[v_\pi(S)]$$

where $d_\beta$ is the **stationary distribution** under policy $\beta$ and $v_\pi$ is the state value under policy $\pi$.

#### Theorem 10.1 (Off-Policy Policy Gradient Theorem)

**Statement**: In the discounted case where $\gamma \in (0, 1)$, the gradient of $J(\theta)$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \rho, A \sim \beta}\left[\underbrace{\frac{\pi(A|S, \theta)}{\beta(A|S)}}_{\text{importance weight}} \nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right]$$
(10.11)

where the state distribution $\rho$ is:

$$\rho(s) \doteq \sum_{s' \in \mathcal{S}} d_\beta(s') \Pr_\pi(s|s'), \quad s \in \mathcal{S}$$

and $\Pr_\pi(s|s') = \sum_{k=0}^\infty \gamma^k [P_\pi^k]_{s's} = [(I - \gamma P_\pi)^{-1}]_{s's}$ is the discounted total probability of transitioning from $s'$ to $s$ under policy $\pi$.

**Two key differences from the on-policy case** (Theorem 9.1):
1. The **importance weight** $\frac{\pi(A|S,\theta)}{\beta(A|S)}$ appears.
2. The action distribution is $A \sim \beta$ instead of $A \sim \pi$.

Therefore, we can use action samples generated by $\beta$ to approximate the true gradient.

#### Proof of Theorem 10.1 (Box 10.2)

Since $d_\beta$ is independent of $\theta$:

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d_\beta(s) \nabla_\theta v_\pi(s)$$
(10.12)

From Lemma 9.2:

$$\nabla_\theta v_\pi(s) = \sum_{s' \in \mathcal{S}} \Pr_\pi(s'|s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s', \theta) \, q_\pi(s', a)$$
(10.13)

Substituting (10.13) into (10.12) and swapping summation order:

$$\nabla_\theta J(\theta) = \sum_{s' \in \mathcal{S}} \underbrace{\left(\sum_{s \in \mathcal{S}} d_\beta(s) \Pr_\pi(s'|s)\right)}_{\rho(s')} \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s', \theta) \, q_\pi(s', a)$$

$$= \mathbb{E}_{S \sim \rho}\left[\sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|S, \theta) \, q_\pi(S, a)\right]$$

Applying importance sampling by multiplying and dividing by $\beta(a|S)$ and $\pi(a|S, \theta)$:

$$= \mathbb{E}_{S \sim \rho}\left[\sum_{a \in \mathcal{A}} \beta(a|S) \frac{\pi(a|S, \theta)}{\beta(a|S)} \nabla_\theta \ln \pi(a|S, \theta) \, q_\pi(S, a)\right]$$

$$= \mathbb{E}_{S \sim \rho, A \sim \beta}\left[\frac{\pi(A|S, \theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right]$$

### 10.3.3 Algorithm Description

**Baseline invariance in the off-policy case**: The off-policy policy gradient is also invariant to any additional baseline $b(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \rho, A \sim \beta}\left[\frac{\pi(A|S, \theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S, \theta) \left(q_\pi(S, A) - b(S)\right)\right]$$

because $\mathbb{E}\left[\frac{\pi(A|S,\theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S, \theta) \, b(S)\right] = 0$.

Setting $b(S) = v_\pi(S)$:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\frac{\pi(A|S, \theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S, \theta) \left(q_\pi(S, A) - v_\pi(S)\right)\right]$$

The stochastic gradient-ascent algorithm is:

$$\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \left[q_t(s_t, a_t) - v_t(s_t)\right]$$

Replacing the advantage function with the TD error $\delta_t(s_t, a_t) \doteq r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t)$:

$$\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \, \delta_t(s_t, a_t)$$

> **Errata correction**: The original text wrote $\theta$ without the subscript $t$ in some instances of this update rule. The correct version uses $\theta_t$ consistently: $\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \, \delta_t(s_t, a_t)$.

**Interpretation from slides**: The update can be rewritten as:

$$\theta_{t+1} = \theta_t + \alpha_\theta \frac{\delta_t(s_t, a_t)}{\beta(a_t|s_t)} \nabla_\theta \pi(a_t|s_t, \theta_t)$$

### Algorithm 10.3: Off-Policy Actor-Critic Based on Importance Sampling

**Initialization:**
- A given behavior policy $\beta(a|s)$
- A target policy $\pi(a|s, \theta_0)$ with initial parameter $\theta_0$
- A value function $v(s, w_0)$ with initial parameter $w_0$
- Learning rates $\alpha_w, \alpha_\theta > 0$

**Goal:** Learn an optimal policy to maximize $J(\theta)$.

**At time step $t$ in each episode, do:**

1. Generate $a_t$ following $\beta(s_t)$ and then observe $r_{t+1}, s_{t+1}$.

2. **Advantage** (TD error):
$$\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$$

3. **Actor** (policy update):
$$\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \, \delta_t \, \nabla_\theta \ln \pi(a_t|s_t, \theta_t)$$

4. **Critic** (value update):
$$w_{t+1} = w_t + \alpha_w \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \, \delta_t \, \nabla_w v(s_t, w_t)$$

**Key observations**:
- The algorithm is **identical** to A2C except that the importance weight $\frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)}$ is included in **both** the actor and the critic.
- The **critic is also off-policy**: importance sampling is applied to the value-based component as well (importance sampling is a general technique applicable to both policy-based and value-based algorithms).
- This algorithm can be extended to incorporate techniques such as **eligibility traces**.

---

## 10.4 Deterministic Actor-Critic

Up to this point, all policies in the policy gradient methods have been **stochastic** (requiring $\pi(a|s, \theta) > 0$ for every $(s, a)$). This section shows that **deterministic policies** can also be used.

### Deterministic Policy Notation

A deterministic policy is denoted as:

$$a = \mu(s, \theta) \doteq \mu(s)$$

**Key differences from stochastic policies**:

| Property | Stochastic Policy $\pi(a|s, \theta)$ | Deterministic Policy $\mu(s, \theta)$ |
|---|---|---|
| Output | Probability distribution over actions | A single action directly |
| Type | $\pi: \mathcal{S} \times \mathcal{A} \to [0, 1]$ | $\mu: \mathcal{S} \to \mathcal{A}$ |
| Representation | e.g., softmax neural network | e.g., neural network (input: $s$, output: $a$, parameter: $\theta$) |

**Why study deterministic policies?**
1. The deterministic case is **naturally off-policy** (explained below).
2. It can **effectively handle continuous action spaces** (from slides: a key benefit).

### 10.4.1 The Deterministic Policy Gradient Theorem

The policy gradient theorem from Chapter 9 is only valid for stochastic policies. A new theorem is needed for deterministic policies.

#### Theorem 10.2 (Deterministic Policy Gradient Theorem -- General Form)

**Statement**: The gradient of $J(\theta)$ is:

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)} = \mathbb{E}_{S \sim \eta}\left[\nabla_\theta \mu(S) \left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}\right]$$
(10.14)

where $\eta$ is a distribution of the states. Theorem 10.2 summarizes the results of Theorem 10.3 (discounted case) and Theorem 10.4 (undiscounted case), which have similar expressions but differ in the specific forms of $J(\theta)$ and $\eta$.

**Critical difference from the stochastic case**: The gradient in (10.14) **does not involve the action random variable** $A$. Consequently:
- When using samples to approximate the true gradient, it is **not required to sample actions**.
- Therefore, the deterministic policy gradient method is **naturally off-policy**.

**Notation clarification**: $\left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}$ cannot be written as $\nabla_a q_\mu(S, \mu(S))$ because it would be unclear that $q_\mu(S, \cdot)$ is being differentiated with respect to $a$. A less confusing alternative is $\nabla_a q_\mu(S, a = \mu(S))$.

#### Metric 1: Average Value (Discounted Case)

$$J(\theta) = \mathbb{E}[v_\mu(s)] = \sum_{s \in \mathcal{S}} d_0(s) v_\mu(s)$$
(10.15)

where $d_0$ is a probability distribution of states, selected to be **independent of** $\mu$ for simplicity.

**Two important special cases of $d_0$**:
1. $d_0(s_0) = 1$ and $d_0(s \neq s_0) = 0$ for a specific starting state $s_0$ -- maximizes discounted return from $s_0$.
2. $d_0$ is the stationary distribution of a behavior policy different from $\mu$.

##### Lemma 10.1 (Gradient of $v_\mu(s)$)

**Statement**: In the discounted case, for any $s \in \mathcal{S}$:

$$\nabla_\theta v_\mu(s) = \sum_{s' \in \mathcal{S}} \Pr_\mu(s'|s) \nabla_\theta \mu(s') \left(\nabla_a q_\mu(s', a)\right)\big|_{a=\mu(s')}$$
(10.16)

where $\Pr_\mu(s'|s) \doteq \sum_{k=0}^\infty \gamma^k [P_\mu^k]_{ss'} = [(I - \gamma P_\mu)^{-1}]_{ss'}$ is the discounted total probability of transitioning from $s$ to $s'$ under policy $\mu$.

**Proof of Lemma 10.1** (Box 10.3):

Since the policy is deterministic: $v_\mu(s) = q_\mu(s, \mu(s))$.

Since both $q_\mu$ and $\mu$ depend on $\theta$, applying the chain rule:

$$\nabla_\theta v_\mu(s) = \left(\nabla_\theta q_\mu(s, a)\right)\big|_{a=\mu(s)} + \nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)}$$
(10.17)

By the definition of action values: $q_\mu(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|s, a) v_\mu(s')$.

Since $r(s, a)$ is independent of $\mu$:

$$\nabla_\theta q_\mu(s, a) = \gamma \sum_{s'} p(s'|s, a) \nabla_\theta v_\mu(s')$$

Substituting into (10.17):

$$\nabla_\theta v_\mu(s) = \gamma \sum_{s'} p(s'|s, \mu(s)) \nabla_\theta v_\mu(s') + \underbrace{\nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)}}_{u(s)}$$

This holds for all $s \in \mathcal{S}$. In matrix-vector form:

$$\nabla_\theta v_\mu = u + \gamma (P_\mu \otimes I_m) \nabla_\theta v_\mu$$

where $n = |\mathcal{S}|$, $m$ is the dimensionality of $\theta$, $[P_\mu]_{ss'} = p(s'|s, \mu(s))$, and $\otimes$ is the Kronecker product. Solving:

$$\nabla_\theta v_\mu = (I_{mn} - \gamma P_\mu \otimes I_m)^{-1} u = \left((I_n - \gamma P_\mu)^{-1} \otimes I_m\right) u$$
(10.18)

The elementwise form gives (10.16). The quantity $[(I - \gamma P_\mu)^{-1}]_{ss'}$ has a clear probabilistic interpretation: since $(I - \gamma P_\mu)^{-1} = I + \gamma P_\mu + \gamma^2 P_\mu^2 + \cdots$:

$$[(I - \gamma P_\mu)^{-1}]_{ss'} = \sum_{k=0}^\infty \gamma^k [P_\mu^k]_{ss'}$$

Here $[P_\mu^k]_{ss'}$ is the probability of transitioning from $s$ to $s'$ in exactly $k$ steps. So $[(I - \gamma P_\mu)^{-1}]_{ss'}$ is the **discounted total probability** of transitioning from $s$ to $s'$ over any number of steps.

##### Theorem 10.3 (Deterministic Policy Gradient -- Discounted Case)

**Statement**: In the discounted case where $\gamma \in (0, 1)$, the gradient of $J(\theta)$ in (10.15) is:

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \rho_\mu(s) \nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)} = \mathbb{E}_{S \sim \rho_\mu}\left[\nabla_\theta \mu(S) \left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}\right]$$

where $\rho_\mu(s) = \sum_{s' \in \mathcal{S}} d_0(s') \Pr_\mu(s|s')$ and $\Pr_\mu(s|s') = \sum_{k=0}^\infty \gamma^k [P_\mu^k]_{s's} = [(I - \gamma P_\mu)^{-1}]_{s's}$.

**Proof** (Box 10.4): Since $d_0$ is independent of $\mu$:

$$\nabla_\theta J(\theta) = \sum_{s} d_0(s) \nabla_\theta v_\mu(s)$$

Substituting Lemma 10.1 and swapping summation order:

$$= \sum_{s'} \underbrace{\left(\sum_s d_0(s) \Pr_\mu(s'|s)\right)}_{\rho_\mu(s')} \nabla_\theta \mu(s') \left(\nabla_a q_\mu(s', a)\right)\big|_{a=\mu(s')}$$

$$= \mathbb{E}_{S \sim \rho_\mu}\left[\nabla_\theta \mu(S) \left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}\right]$$

#### Metric 2: Average Reward (Undiscounted Case)

$$J(\theta) = \bar{r}_\mu = \sum_{s \in \mathcal{S}} d_\mu(s) r_\mu(s) = \mathbb{E}_{S \sim d_\mu}[r_\mu(S)]$$
(10.20)

where $r_\mu(s) = \mathbb{E}[R|s, a = \mu(s)] = \sum_r r \, p(r|s, a = \mu(s))$ and $d_\mu$ is the stationary distribution under policy $\mu$.

##### Theorem 10.4 (Deterministic Policy Gradient -- Undiscounted Case)

**Statement**: In the undiscounted case, the gradient of $J(\theta)$ in (10.20) is:

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} d_\mu(s) \nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)} = \mathbb{E}_{S \sim d_\mu}\left[\nabla_\theta \mu(S) \left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}\right]$$

where $d_\mu$ is the stationary distribution of the states under policy $\mu$.

**Proof sketch** (Box 10.5): Starting from $v_\mu(s) = q_\mu(s, \mu(s))$, applying the chain rule (10.21), and using the undiscounted action value definition $q_\mu(s, a) = r(s, a) - \bar{r}_\mu + \sum_{s'} p(s'|s, a) v_\mu(s')$, one arrives at a matrix-vector form:

$$\nabla_\theta v_\mu = u - \mathbf{1}_n \otimes \nabla_\theta \bar{r}_\mu + (P_\mu \otimes I_m) \nabla_\theta v_\mu$$

Using the stationary distribution property $d_\mu^T P_\mu = d_\mu^T$ and multiplying both sides by $d_\mu^T \otimes I_m$:

$$\nabla_\theta \bar{r}_\mu = d_\mu^T \otimes I_m \, u = \sum_{s} d_\mu(s) \, u(s) = \sum_{s} d_\mu(s) \nabla_\theta \mu(s) \left(\nabla_a q_\mu(s, a)\right)\big|_{a=\mu(s)}$$

### 10.4.2 Algorithm Description

Based on Theorem 10.2, the gradient-ascent algorithm is:

$$\theta_{t+1} = \theta_t + \alpha_\theta \, \mathbb{E}_{S \sim \eta}\left[\nabla_\theta \mu(S) \left(\nabla_a q_\mu(S, a)\right)\big|_{a=\mu(S)}\right]$$

The stochastic gradient-ascent algorithm is:

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t) \left(\nabla_a q_\mu(s_t, a)\right)\big|_{a=\mu(s_t)}$$

### Algorithm 10.4: Deterministic Policy Gradient (DPG) / Deterministic Actor-Critic

**Initialization:**
- A given behavior policy $\beta(a|s)$
- A deterministic target policy $\mu(s, \theta_0)$ with initial parameter $\theta_0$
- A value function $q(s, a, w_0)$ with initial parameter $w_0$
- Learning rates $\alpha_w, \alpha_\theta > 0$

**Goal:** Learn an optimal policy to maximize $J(\theta)$.

**At time step $t$ in each episode, do:**

1. Generate $a_t$ following $\beta$ and then observe $r_{t+1}, s_{t+1}$.

2. **TD error**:
$$\delta_t = r_{t+1} + \gamma q(s_{t+1}, \mu(s_{t+1}, \theta_t), w_t) - q(s_t, a_t, w_t)$$

3. **Actor** (policy update):
$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t, \theta_t) \left(\nabla_a q(s_t, a, w_t)\right)\big|_{a=\mu(s_t)}$$

4. **Critic** (value update):
$$w_{t+1} = w_t + \alpha_w \, \delta_t \, \nabla_w q(s_t, a_t, w_t)$$

#### Why This Algorithm Is Off-Policy

**Actor is off-policy**: The gradient (10.14) does not involve the action random variable $A$. Therefore, when approximating the gradient with samples, we do not need to sample actions from $\mu$, and any behavior policy $\beta$ can be used.

**Critic is off-policy** (subtle point): The experience sample required by the critic is $(s_t, a_t, r_{t+1}, s_{t+1}, \tilde{a}_{t+1})$ where $\tilde{a}_{t+1} = \mu(s_{t+1})$. Two policies are involved:
- $a_t$ at $s_t$ is generated by the **behavior policy** $\beta$ (used to interact with the environment).
- $\tilde{a}_{t+1} = \mu(s_{t+1})$ is generated by the **target policy** $\mu$ (used for evaluation, but **not** used to interact with the environment in the next step).

Since $\beta$ (behavior) and $\mu$ (target) differ, the critic is off-policy. Importantly, the critic does **not** require the importance sampling technique.

#### How to Select $q(s, a, w)$?

| Representation | Details | Reference |
|---|---|---|
| Linear function | $q(s, a, w) = \phi^T(s, a) w$ where $\phi(s, a)$ is the feature vector | DPG paper [74] |
| Neural network | Deep deterministic policy gradient (**DDPG**) | [75] |

#### How to Select the Behavior Policy $\beta$?

- It can be **any exploratory policy**.
- It can be a stochastic policy obtained by **adding noise to** $\mu$ (e.g., $\beta = \mu + \text{noise}$). In this case, $\mu$ is effectively also the behavior policy, making this an **on-policy implementation**.

---

## 10.5 Summary

| Section | Algorithm | Key Idea | On/Off-Policy | Critic Estimates |
|---|---|---|---|---|
| 10.1 | QAC | Simplest actor-critic; TD-based $q$-estimation in policy gradient | On-policy | $q(s, a, w)$ |
| 10.2 | A2C / TD actor-critic | Baseline $v_\pi(s)$ reduces variance; advantage function | On-policy | $v(s, w)$ |
| 10.3 | Off-policy actor-critic | Importance sampling enables off-policy learning | Off-policy | $v(s, w)$ |
| 10.4 | DPG / Deterministic actor-critic | Deterministic policy; gradient has no action variable | Off-policy | $q(s, a, w)$ |

**Progression of algorithms**:
- QAC $\to$ A2C: Add baseline to reduce variance.
- A2C $\to$ Off-policy AC: Add importance sampling for off-policy learning.
- Stochastic $\to$ DPG: Switch to deterministic policy; naturally off-policy; handles continuous actions.

**Beyond this book**: Policy gradient and actor-critic methods are widely used in modern RL. Advanced algorithms include:
- **SAC** (Soft Actor-Critic) [76, 77]
- **TRPO** (Trust Region Policy Optimization) [78]
- **PPO** (Proximal Policy Optimization) [79]
- **TD3** (Twin Delayed DDPG) [80]
- Multi-agent RL extensions [81-85]
- Model-based RL [15, 86, 87]
- Distributional RL [88, 89]
- Connections between RL and control theory [90-95]

---

## 10.6 Q&A -- Important Clarifications

### Q: What is the relationship between actor-critic and policy gradient methods?

**A**: Actor-critic methods **are** policy gradient methods. Sometimes the terms are used interchangeably. Every policy gradient algorithm requires estimating action values. When the action values are estimated using **TD learning with value function approximation**, the algorithm is called actor-critic. The name highlights the algorithmic structure combining **policy update** (actor) and **value update** (critic). This structure is the fundamental structure used in all RL algorithms.

### Q: Why is it important to introduce additional baselines?

**A**: Since the policy gradient is invariant to any additional baseline, we can utilize the baseline to **reduce estimation variance**. The resulting algorithm is called advantage actor-critic. The key insight is that the **mean** ($\mathbb{E}[X]$) is invariant but the **variance** ($\text{var}(X)$) is not, so we can choose the baseline to minimize variance without changing the expected gradient.

### Q: Can importance sampling be used in value-based algorithms?

**A**: Yes. Importance sampling is a general technique for estimating the expectation of a random variable over one distribution using samples drawn from another. It is useful in RL because many problems involve estimating expectations:
- Value-based methods: action/state values are defined as expectations.
- Policy gradient: the true gradient is an expectation.
- In Algorithm 10.3, importance sampling is applied to **both** the policy-based (actor) and value-based (critic) components.

### Q: Why is the deterministic policy gradient method off-policy?

**A**: The true gradient in the deterministic case **does not involve the action random variable**. As a result, when using samples to approximate the true gradient, it is **not required to sample actions** from any particular policy. Therefore, any policy can be used to generate state samples, making the method off-policy.

---

## Concept Index

| Concept | Notation / Formula | Section |
|---|---|---|
| Actor | Policy update step | 10.1 |
| Critic | Value estimation / policy evaluation step | 10.1 |
| Q Actor-Critic (QAC) | Algorithm 10.1 | 10.1 |
| Baseline | $b(S)$, scalar function of $S$ | 10.2.1 |
| Baseline invariance | $\mathbb{E}[\nabla_\theta \ln \pi \cdot b(S)] = 0$ | 10.2.1 |
| Optimal baseline | $b^*(s) = \frac{\mathbb{E}[\|\nabla_\theta \ln \pi\|^2 q_\pi]}{\mathbb{E}[\|\nabla_\theta \ln \pi\|^2]}$ | 10.2.1 |
| Suboptimal baseline (state value) | $b^\dagger(s) = v_\pi(s)$ | 10.2.1 |
| Advantage function | $\delta_\pi(s, a) = q_\pi(s, a) - v_\pi(s)$ | 10.2.2 |
| TD error (as advantage approx.) | $\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$ | 10.2.2 |
| Advantage actor-critic (A2C) | Algorithm 10.2 | 10.2.2 |
| TD actor-critic | Same as A2C | 10.2.2 |
| REINFORCE with baseline | MC version of A2C | 10.2.2 |
| Importance sampling | $\mathbb{E}_{p_0}[X] = \mathbb{E}_{p_1}\left[\frac{p_0(X)}{p_1(X)} X\right]$ | 10.3.1 |
| Importance weight | $\frac{p_0(x)}{p_1(x)}$ or $\frac{\pi(a|s,\theta)}{\beta(a|s)}$ | 10.3.1 |
| Off-policy policy gradient theorem | Theorem 10.1 | 10.3.2 |
| Off-policy actor-critic | Algorithm 10.3 | 10.3.3 |
| Deterministic policy | $a = \mu(s, \theta)$ | 10.4 |
| Deterministic policy gradient theorem | Theorem 10.2 (general), 10.3 (discounted), 10.4 (undiscounted) | 10.4.1 |
| Discounted total probability | $\Pr_\mu(s'|s) = [(I - \gamma P_\mu)^{-1}]_{ss'}$ | 10.4.1 |
| Deterministic actor-critic (DPG) | Algorithm 10.4 | 10.4.2 |
| Deep Deterministic Policy Gradient (DDPG) | DPG with neural network $q(s,a,w)$ | 10.4.2 |
| A3C | Asynchronous advantage actor-critic (variant of A2C) | 10.2.2 |

---

## Dependencies and Forward References

| This chapter concept | Depends on |
|---|---|
| Policy gradient $\nabla_\theta J(\theta)$ and Theorem 9.1 | Ch 9 (policy gradient methods) |
| Stochastic gradient ascent for policy update (10.2) | Ch 9, equation (9.32) |
| $q_\pi(s, a)$ estimation via Sarsa + function approx. | Ch 7 (TD methods), Ch 8 (value function approximation) |
| Baseline invariance proof ($\sum_a \nabla_\theta \pi = \nabla_\theta 1 = 0$) | Ch 9 (log-derivative trick), Ch 1 (policy as probability) |
| TD error $\delta_t$ as advantage approximation | Ch 7 (TD learning) |
| Stationary distribution $d_\beta$, $d_\mu$ | Ch 9, Section 9.2 |
| Lemma 9.2 ($\nabla_\theta v_\pi(s)$) | Ch 9 |
| Bellman equation for $q_\mu(s,a)$ in deterministic case | Ch 2 (Bellman equation) |
| Kronecker product derivation for gradient | Linear algebra (Ch 6 / appendix) |
| Discounted return, $\gamma$ | Ch 1 (basic concepts), Ch 2-3 |
| Monte Carlo vs. TD estimation | Ch 5 (Monte Carlo), Ch 7 (TD) |

| This chapter concept | Extended by / Referenced in |
|---|---|
| QAC, A2C, Off-policy AC, DPG | SAC, TRPO, PPO, TD3 (advanced algorithms beyond this book) |
| Importance sampling | General technique applicable across all RL algorithms |
| Deterministic policy gradient | DDPG, TD3 (deep RL) |
| Actor-critic structure | Fundamental structure of all modern RL algorithms |
| Single-agent actor-critic | Multi-agent RL extensions |
