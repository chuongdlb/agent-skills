---
chapter: 5
title: Monte Carlo Methods
key_topics: [mean estimation, law of large numbers, Monte Carlo estimation, model-free reinforcement learning, MC Basic algorithm, policy iteration to model-free conversion, action value estimation, episode length, sparse rewards, sample efficiency, initial-visit strategy, first-visit strategy, every-visit strategy, MC Exploring Starts algorithm, exploring starts condition, policy update strategies, soft policies, epsilon-greedy policies, MC epsilon-Greedy algorithm, exploration vs exploitation tradeoff, generalized policy iteration]
depends_on: [1, 2, 3, 4]
required_by: [6, 7, 8, 9, 10]
---

# Chapter 5: Monte Carlo Methods

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 5, pp. 77-99
> Supplemented by: Lecture slides L5 (47 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter marks the **transition from model-based to model-free** reinforcement learning. In the previous chapter (Chapter 4), algorithms such as value iteration and policy iteration required knowledge of the system model $\{p(r|s,a), p(s'|s,a)\}$. This chapter introduces algorithms that learn optimal policies directly from **experience samples (data)** without needing the model.

**Core philosophy**: If we do not have a model, we must have some data. If we do not have data, we must have a model. If we have neither, then we cannot find optimal policies. The "data" in RL refers to the agent's interaction experiences with the environment.

**Position in the book**: Chapter 5 is the bridge between model-based methods (Chapters 2-4) and the more advanced model-free algorithms (Chapters 7-10). It introduces three progressively refined MC algorithms:
1. **MC Basic** -- reveals the core idea (simple but impractical)
2. **MC Exploring Starts** -- improves sample efficiency
3. **MC $\epsilon$-Greedy** -- removes the exploring starts requirement

**Key insight from slides**: The chapter is built on two pillars: (1) understanding policy iteration well (Chapter 4), and (2) understanding Monte Carlo mean estimation. Combining these two ideas yields MC-based model-free RL.

---

## 5.1 Motivating Example: Mean Estimation

### Why Mean Estimation Matters for RL

State values and action values are both defined as **expected values (means) of returns**:

$$v_\pi(s) = \mathbb{E}[G_t | S_t = s], \qquad q_\pi(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

Therefore, estimating a state or action value is fundamentally a **mean estimation problem**.

### Two Approaches to Computing $\mathbb{E}[X]$

Consider a random variable $X$ that takes values from a finite set $\mathcal{X}$ of real numbers.

**Approach 1: Model-based.** If the probability distribution $p(x)$ is known:

$$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} p(x) \, x$$

**Approach 2: Model-free (Monte Carlo).** If $p(x)$ is unknown but we have i.i.d. samples $\{x_1, x_2, \ldots, x_n\}$:

$$\mathbb{E}[X] \approx \bar{x} = \frac{1}{n} \sum_{j=1}^{n} x_j$$

When $n$ is small, the approximation may be inaccurate. As $n \to \infty$, we have $\bar{x} \to \mathbb{E}[X]$.

### Coin Flipping Example

Let $X$ denote the outcome of a coin flip: $X = +1$ (head) or $X = -1$ (tail).

**Model-based**: If $p(X=1) = 0.5$ and $p(X=-1) = 0.5$:
$$\mathbb{E}[X] = 0.5 \cdot 1 + 0.5 \cdot (-1) = 0$$

**Model-free**: Flip the coin many times, record $\{x_i\}_{i=1}^n$, and compute the running average. The estimated mean converges to 0 as $n$ increases.

### Law of Large Numbers

> **Box 5.1: Law of Large Numbers**
>
> For a random variable $X$, suppose $\{x_i\}_{i=1}^n$ are i.i.d. samples. Let $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$. Then:
>
> $$\mathbb{E}[\bar{x}] = \mathbb{E}[X]$$
> $$\text{var}[\bar{x}] = \frac{1}{n}\text{var}[X]$$
>
> **Interpretation**: $\bar{x}$ is an **unbiased estimate** of $\mathbb{E}[X]$, and its variance decreases to zero as $n \to \infty$.

**Proof**:

1. $\mathbb{E}[\bar{x}] = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n x_i\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[x_i] = \mathbb{E}[X]$, where the last equality uses the fact that samples are **identically distributed** ($\mathbb{E}[x_i] = \mathbb{E}[X]$).

2. $\text{var}(\bar{x}) = \text{var}\left(\frac{1}{n}\sum_{i=1}^n x_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{var}[x_i] = \frac{n \cdot \text{var}[X]}{n^2} = \frac{\text{var}[X]}{n}$, where the second equality uses **independence** of samples, and the third uses identical distribution ($\text{var}[x_i] = \text{var}[X]$).

**Critical requirement**: Samples must be **i.i.d.** (independent and identically distributed). If samples are correlated, correct estimation may be impossible. Extreme case: if all samples equal the first one, $\bar{x}$ always equals the first sample regardless of $n$.

### Definition: Monte Carlo Estimation

**Monte Carlo estimation** refers to a broad class of techniques that rely on repeated random sampling to solve approximation problems. The term is used broadly -- any method that uses stochastic samples to estimate quantities.

---

## 5.2 MC Basic: The Simplest MC-Based Algorithm

### 5.2.1 Converting Policy Iteration to Be Model-Free

Recall the policy iteration algorithm (Section 4.2) has two steps per iteration:

- **Step 1 (Policy evaluation)**: Compute $v_{\pi_k}$ by solving $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$
- **Step 2 (Policy improvement)**: Compute $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$

The elementwise form of the policy improvement step is:

$$\pi_{k+1}(s) = \arg\max_\pi \sum_a \pi(a|s) \left[\sum_r p(r|s,a)\,r + \gamma \sum_{s'} p(s'|s,a)\,v_{\pi_k}(s')\right] = \arg\max_\pi \sum_a \pi(a|s)\, q_{\pi_k}(s,a), \quad s \in \mathcal{S}$$

**Key observation**: Action values lie at the core of both steps. In Step 1, state values are computed *for the purpose of* computing action values. In Step 2, the new policy is generated based on action values.

### Two Approaches to Computing Action Values

**Approach 1 (Model-based)**: First compute $v_{\pi_k}$ from the Bellman equation, then:

$$q_{\pi_k}(s,a) = \sum_r p(r|s,a)\,r + \gamma \sum_{s'} p(s'|s,a)\,v_{\pi_k}(s') \tag{5.1}$$

This requires the system model $\{p(r|s,a), p(s'|s,a)\}$.

**Approach 2 (Model-free)**: Use the definition of action value directly:

$$q_{\pi_k}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t = s, A_t = a]$$

Since $q_{\pi_k}(s,a)$ is an expectation, it can be estimated by MC methods. Starting from $(s,a)$, the agent follows $\pi_k$ and collects $n$ episodes with returns $g^{(1)}_{\pi_k}(s,a), \ldots, g^{(n)}_{\pi_k}(s,a)$. Then:

$$q_{\pi_k}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a] \approx \frac{1}{n}\sum_{i=1}^n g^{(i)}_{\pi_k}(s,a) \tag{5.2}$$

By the law of large numbers, this approximation becomes accurate when $n$ is sufficiently large.

**Fundamental idea of MC-based RL**: Replace the model-based policy evaluation in policy iteration with model-free MC estimation of action values, as in (5.2).

**Why action values instead of state values?** (Slide insight) If we estimate state values $v_\pi(s)$ instead, we would still need the model to compute action values via (5.1) for the policy improvement step. Estimating action values directly avoids this model dependency entirely.

### 5.2.2 The MC Basic Algorithm

Starting from an initial policy $\pi_0$, the algorithm iterates ($k = 0, 1, 2, \ldots$):

**Step 1 (Policy evaluation)**: For every $(s,a)$, collect sufficiently many episodes starting from $(s,a)$ by following $\pi_k$. Use the average return $q_k(s,a)$ to approximate $q_{\pi_k}(s,a)$.

**Step 2 (Policy improvement)**: Solve $\pi_{k+1}(s) = \arg\max_\pi \sum_a \pi(a|s)\,q_k(s,a)$ for all $s \in \mathcal{S}$. The greedy optimal policy is $\pi_{k+1}(a^*_k|s) = 1$ where $a^*_k = \arg\max_a q_k(s,a)$.

> **Algorithm 5.1: MC Basic (a model-free variant of policy iteration)**
>
> **Initialization**: Initial guess $\pi_0$.
> **Goal**: Search for an optimal policy.
>
> **For** the $k$-th iteration ($k = 0, 1, 2, \ldots$), **do**
> $\quad$ **For** every state $s \in \mathcal{S}$, **do**
> $\quad\quad$ **For** every action $a \in \mathcal{A}(s)$, **do**
> $\quad\quad\quad$ Collect sufficiently many episodes starting from $(s,a)$ by following $\pi_k$
> $\quad\quad\quad$ **Policy evaluation:**
> $\quad\quad\quad\quad$ $q_{\pi_k}(s,a) \approx q_k(s,a) =$ average return of all episodes starting from $(s,a)$
> $\quad\quad$ **Policy improvement:**
> $\quad\quad\quad$ $a^*_k(s) = \arg\max_a q_k(s,a)$
> $\quad\quad\quad$ $\pi_{k+1}(a|s) = 1$ if $a = a^*_k$, and $\pi_{k+1}(a|s) = 0$ otherwise

### Key Properties of MC Basic

1. **Very similar to policy iteration**: The only difference is that MC Basic estimates action values directly from experience, while policy iteration computes state values first and then derives action values using the model.

2. **Convergence**: Since policy iteration converges, MC Basic also converges **when given sufficient samples**. For every $(s,a)$, sufficiently many episodes are needed so that the average return accurately approximates the action value.

3. **Robustness to inaccurate estimates**: Even with insufficient episodes, the algorithm usually still works. This is analogous to truncated policy iteration (Chapter 4), where action values are not accurately calculated either.

4. **Not practical**: MC Basic is too simple and has low sample efficiency. It is introduced to reveal the core idea of MC-based RL, serving as the foundation for more efficient algorithms.

### 5.2.3 Illustrative Examples

#### Simple Example: Step-by-Step (3x3 Grid World)

**Setup**: 3x3 grid world with $r_\text{boundary} = r_\text{forbidden} = -1$, $r_\text{target} = 1$, $\gamma = 0.9$. The initial policy $\pi_0$ is shown in Figure 5.3 (already optimal for all states except $s_1$ and $s_3$).

Since the policy and model are both **deterministic**, a single episode suffices for each action value (multiple runs yield the same trajectory).

**Computing $q_{\pi_0}(s_1, a)$ for all five actions**:

| Starting pair | Episode | Return formula | Value |
|---|---|---|---|
| $(s_1, a_1)$ | $s_1 \xrightarrow{a_1} s_1 \xrightarrow{a_1} s_1 \xrightarrow{a_1} \cdots$ | $-1 + \gamma(-1) + \gamma^2(-1) + \cdots$ | $\frac{-1}{1-\gamma}$ |
| $(s_1, a_2)$ | $s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_3} \cdots$ | $0 + \gamma(0) + \gamma^2(0) + \gamma^3(1) + \gamma^4(1) + \cdots$ | $\frac{\gamma^3}{1-\gamma}$ |
| $(s_1, a_3)$ | $s_1 \xrightarrow{a_3} s_4 \xrightarrow{a_2} s_5 \xrightarrow{a_3} \cdots$ | $0 + \gamma(0) + \gamma^2(0) + \gamma^3(1) + \gamma^4(1) + \cdots$ | $\frac{\gamma^3}{1-\gamma}$ |
| $(s_1, a_4)$ | $s_1 \xrightarrow{a_4} s_1 \xrightarrow{a_1} s_1 \xrightarrow{a_1} \cdots$ | $-1 + \gamma(-1) + \gamma^2(-1) + \cdots$ | $\frac{-1}{1-\gamma}$ |
| $(s_1, a_5)$ | $s_1 \xrightarrow{a_5} s_1 \xrightarrow{a_1} s_1 \xrightarrow{a_1} \cdots$ | $0 + \gamma(-1) + \gamma^2(-1) + \cdots$ | $\frac{-\gamma}{1-\gamma}$ |

**With $\gamma = 0.9$**: $\frac{-1}{1-0.9} = -10$, $\frac{\gamma^3}{1-0.9} = \frac{0.729}{0.1} = 7.29$, $\frac{-\gamma}{1-0.9} = -9$.

**Policy improvement for $s_1$**: $q_{\pi_0}(s_1, a_2) = q_{\pi_0}(s_1, a_3) = \frac{\gamma^3}{1-\gamma}$ are the maximum values. So:

$$\pi_1(a_2|s_1) = 1 \quad \text{or} \quad \pi_1(a_3|s_1) = 1$$

Either choice gives an optimal policy for $s_1$. **One iteration suffices** for this simple example since $\pi_0$ was already optimal at all other states.

#### Comprehensive Example: Episode Length and Sparse Rewards (5x5 Grid World)

**Setup**: 5x5 grid world with $r_\text{boundary} = -1$, $r_\text{forbidden} = -10$, $r_\text{target} = 1$, $\gamma = 0.9$.

**Key findings from Figure 5.4** (results for different episode lengths):

| Episode Length | Result |
|---|---|
| 1 | Only states adjacent to target have nonzero values. All others are 0. |
| 2 | Slightly more states get nonzero values (up to 1.9). |
| 3 | Values propagate further (up to 2.7). |
| 4 | More propagation (up to 3.4). |
| 14 | Most states have nonzero values, but bottom-left still has value 0. |
| 15 | All states have nonzero values. Policy is near-optimal. |
| 30 | Algorithm finds optimal policy, though values not yet fully converged. |
| 100 | Optimal policy and near-optimal state values (target value $= 10.0$). |

**Spatial pattern**: States closer to the target get nonzero values earlier than states farther away. The agent must travel at least a minimum number of steps to reach the target. If the episode length is shorter than this minimum, the return is zero.

**Critical threshold**: Episode length must be $\geq 15$, which is the minimum number of steps from the bottom-left corner to the target in the 5x5 grid.

**Key insight**: While episodes must be sufficiently long, they need not be infinitely long. Length 30 already yields optimal policies even though value estimates are not yet fully converged.

#### Sparse Rewards

**Definition**: A **sparse reward** setting is one in which no positive rewards can be obtained unless the target is reached.

**Problem**: Sparse rewards require long episodes that can reach the target. When the state space is large, this is challenging, degrading learning efficiency.

**Simple solution**: Design **non-sparse rewards**. For example, assign small positive rewards for reaching states near the target, creating an "attractive field" that guides the agent toward the target more easily.

---

## 5.3 MC Exploring Starts

MC Exploring Starts extends MC Basic to be more **sample-efficient** via two key improvements: (1) utilizing samples within episodes more efficiently, and (2) updating policies more efficiently.

### 5.3.1 Utilizing Samples More Efficiently

Consider an episode generated by following policy $\pi$:

$$s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots \tag{5.3}$$

**Definition -- Visit**: Every time a state-action pair appears in an episode, it is called a **visit** of that state-action pair.

#### Three Strategies for Using Visits

**1. Initial-visit strategy** (used by MC Basic):
- Only use the episode to estimate the action value of the **initial** state-action pair.
- For episode (5.3), only $q_\pi(s_1, a_2)$ is estimated.
- **Disadvantage**: Not sample-efficient -- the episode visits many other pairs whose data is wasted.

**2. First-visit strategy**:
- For each state-action pair, only count the **first** time it is visited in the episode.
- The sub-trajectory from the first visit onward is used to compute the return for that pair.

**3. Every-visit strategy**:
- Count **every** visit of a state-action pair.
- Each visit generates a sub-episode whose return contributes to the action value estimate.

#### Decomposing an Episode into Sub-Episodes

The episode in (5.3) can be decomposed:

| Sub-episode | Estimates |
|---|---|
| $s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots$ | $q_\pi(s_1, a_2)$ (first visit) |
| $s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots$ | $q_\pi(s_2, a_4)$ |
| $s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots$ | $q_\pi(s_1, a_2)$ (second visit) |
| $s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots$ | $q_\pi(s_2, a_3)$ |
| $s_5 \xrightarrow{a_1} \cdots$ | $q_\pi(s_5, a_1)$ |

**Sample efficiency comparison**: Every-visit > First-visit > Initial-visit.

**Correlation caveat for every-visit**: Samples from the every-visit strategy are correlated because the trajectory from the second visit is a subset of the trajectory from the first visit. However, if the two visits are far apart in the episode, the correlation is not strong.

### 5.3.2 Updating Policies More Efficiently

**Strategy 1** (MC Basic): Collect **all** episodes for each state-action pair, then compute the average return. **Drawback**: Must wait until all episodes are collected before updating.

**Strategy 2** (MC Exploring Starts): Use the return of a **single episode** to approximate the action value, and improve the policy **episode-by-episode**.

**Concern**: A single episode's return cannot accurately approximate the action value. **Justification**: This falls under **generalized policy iteration** (Chapter 4) -- the policy can be updated even when value estimates are not fully accurate. This is analogous to truncated policy iteration.

**Definition -- Generalized Policy Iteration (GPI)** (from slides): Not a specific algorithm, but the general idea/framework of switching between policy-evaluation and policy-improvement processes. Many RL algorithms fall into this framework.

### 5.3.3 Algorithm Description

> **Algorithm 5.2: MC Exploring Starts (an efficient variant of MC Basic)**
>
> **Initialization**: Initial policy $\pi_0(a|s)$ and initial value $q(s,a)$ for all $(s,a)$.
> $\quad$ $\text{Returns}(s,a) = 0$ and $\text{Num}(s,a) = 0$ for all $(s,a)$.
> **Goal**: Search for an optimal policy.
>
> **For each** episode, **do**
> $\quad$ **Episode generation**: Select a starting state-action pair $(s_0, a_0)$ and ensure that all pairs can be possibly selected (**exploring-starts condition**). Following the current policy, generate an episode of length $T$: $s_0, a_0, r_1, \ldots, s_{T-1}, a_{T-1}, r_T$.
> $\quad$ **Initialization for each episode**: $g \leftarrow 0$
> $\quad$ **For** each step of the episode, $t = T-1, T-2, \ldots, 0$, **do**
> $\quad\quad$ $g \leftarrow \gamma g + r_{t+1}$
> $\quad\quad$ $\text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + g$
> $\quad\quad$ $\text{Num}(s_t, a_t) \leftarrow \text{Num}(s_t, a_t) + 1$
> $\quad\quad$ **Policy evaluation:**
> $\quad\quad\quad$ $q(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) / \text{Num}(s_t, a_t)$
> $\quad\quad$ **Policy improvement:**
> $\quad\quad\quad$ $\pi(a|s_t) = 1$ if $a = \arg\max_a q(s_t, a)$ and $\pi(a|s_t) = 0$ otherwise

**Implementation detail**: The algorithm computes discounted returns by iterating **backwards** from the end of the episode to the start. The update $g \leftarrow \gamma g + r_{t+1}$ efficiently accumulates the discounted return. This backward computation makes the algorithm more efficient but also more complex, which is why MC Basic (free of such techniques) is introduced first.

**Visit strategy**: This algorithm uses the **every-visit** strategy.

### The Exploring Starts Condition

**Definition**: The **exploring starts** condition requires that sufficiently many episodes are generated starting from **every** state-action pair.

**Why it is needed**: Only if every action value for every state is well explored can we accurately estimate all action values (by the law of large numbers) and correctly select optimal actions. Otherwise, an unexplored action might actually be optimal but be missed.

**Practical limitation**: Exploring starts is difficult to achieve in many applications, especially those involving physical interactions with environments. It is hard to collect episodes starting from every state-action pair.

**Key slide insight**: "Exploring starts" means we need episodes starting from $\{(s_1, a_j)\}_{j=1}^5$, $\{(s_2, a_j)\}_{j=1}^5$, $\ldots$, $\{(s_9, a_j)\}_{j=1}^5$. For a 3x3 grid with 5 actions, that is $9 \times 5 = 45$ state-action pairs. Both MC Basic and MC Exploring Starts need this assumption.

---

## 5.4 MC $\epsilon$-Greedy: Learning Without Exploring Starts

### 5.4.1 Soft Policies and $\epsilon$-Greedy Policies

#### Soft Policies

**Definition**: A policy is **soft** if it has a positive probability of taking any action at any state, i.e., $\pi(a|s) > 0$ for all $a \in \mathcal{A}(s)$ and all $s \in \mathcal{S}$.

**Why soft policies help**: With a soft policy, a single episode that is sufficiently long can visit **every** state-action pair many times. This eliminates the need to generate many episodes starting from different state-action pairs, thereby removing the exploring starts requirement.

#### $\epsilon$-Greedy Policies

**Definition**: An $\epsilon$-greedy policy ($\epsilon \in [0, 1]$) has the form:

$$\pi(a|s) = \begin{cases} 1 - \dfrac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & \text{for the greedy action} \\[8pt] \dfrac{\epsilon}{|\mathcal{A}(s)|}, & \text{for the other } |\mathcal{A}(s)| - 1 \text{ actions} \end{cases}$$

where $|\mathcal{A}(s)|$ is the number of actions available at state $s$, and the **greedy action** is $a^* = \arg\max_a q(s,a)$.

**Equivalent form for the greedy action probability**:

$$1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1) = 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$$

#### Numerical Example

If $\epsilon = 0.2$ and $|\mathcal{A}(s)| = 5$:
- Probability of each non-greedy action: $\frac{\epsilon}{|\mathcal{A}(s)|} = \frac{0.2}{5} = 0.04$
- Probability of greedy action: $1 - 0.04 \times 4 = 0.84$

#### Key Properties

1. **Greedy action always preferred**: The greedy action probability is always $\geq$ the probability of any other action:
$$1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} \geq \frac{\epsilon}{|\mathcal{A}(s)|}$$
for any $\epsilon \in [0,1]$.

2. **Special cases**:
   - $\epsilon \to 0$: The policy becomes **greedy** (deterministic). More exploitation, less exploration.
     $$\pi(a|s) = \begin{cases} 1, & \text{greedy action} \\ 0, & \text{otherwise} \end{cases}$$
   - $\epsilon = 1$: The policy becomes a **uniform distribution** over all actions. More exploration, less exploitation.
     $$\pi(a|s) = \frac{1}{|\mathcal{A}(s)|} \quad \text{for all } a$$

3. **Practical action selection**: Generate a random number $x \sim \text{Uniform}[0,1]$.
   - If $x \geq \epsilon$: select the greedy action.
   - If $x < \epsilon$: randomly select any action in $\mathcal{A}(s)$ with probability $\frac{1}{|\mathcal{A}(s)|}$ (may re-select greedy).
   - Total probability of greedy action: $1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$. Probability of each other action: $\frac{\epsilon}{|\mathcal{A}(s)|}$.

### 5.4.2 Algorithm Description

To integrate $\epsilon$-greedy policies, only the **policy improvement step** changes.

**Original (MC Basic / MC Exploring Starts)**: Solve over all policies $\Pi$:

$$\pi_{k+1}(s) = \arg\max_{\pi \in \Pi} \sum_a \pi(a|s)\, q_{\pi_k}(s,a) \tag{5.4}$$

Solution is a greedy policy: $\pi_{k+1}(a|s) = \begin{cases} 1, & a = a^*_k \\ 0, & a \neq a^*_k \end{cases}$

**MC $\epsilon$-Greedy**: Solve over all $\epsilon$-greedy policies $\Pi_\epsilon$:

$$\pi_{k+1}(s) = \arg\max_{\pi \in \Pi_\epsilon} \sum_a \pi(a|s)\, q_{\pi_k}(s,a) \tag{5.5}$$

Solution is an $\epsilon$-greedy policy:

$$\pi_{k+1}(a|s) = \begin{cases} 1 - \dfrac{|\mathcal{A}(s)| - 1}{|\mathcal{A}(s)|}\,\epsilon, & a = a^*_k \\[8pt] \dfrac{1}{|\mathcal{A}(s)|}\,\epsilon, & a \neq a^*_k \end{cases}$$

where $a^*_k = \arg\max_a q_{\pi_k}(s,a)$.

> **Algorithm 5.3: MC $\epsilon$-Greedy (a variant of MC Exploring Starts)**
>
> **Initialization**: Initial policy $\pi_0(a|s)$ and initial value $q(s,a)$ for all $(s,a)$.
> $\quad$ $\text{Returns}(s,a) = 0$ and $\text{Num}(s,a) = 0$ for all $(s,a)$. $\epsilon \in (0, 1]$.
> **Goal**: Search for an optimal policy.
>
> **For each** episode, **do**
> $\quad$ **Episode generation**: Select a starting state-action pair $(s_0, a_0)$ (**exploring starts not required**). Following the current policy, generate an episode of length $T$: $s_0, a_0, r_1, \ldots, s_{T-1}, a_{T-1}, r_T$.
> $\quad$ **Initialization for each episode**: $g \leftarrow 0$
> $\quad$ **For** each step of the episode, $t = T-1, T-2, \ldots, 0$, **do**
> $\quad\quad$ $g \leftarrow \gamma g + r_{t+1}$
> $\quad\quad$ $\text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + g$
> $\quad\quad$ $\text{Num}(s_t, a_t) \leftarrow \text{Num}(s_t, a_t) + 1$
> $\quad\quad$ **Policy evaluation:**
> $\quad\quad\quad$ $q(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) / \text{Num}(s_t, a_t)$
> $\quad\quad$ **Policy improvement:**
> $\quad\quad\quad$ Let $a^* = \arg\max_a q(s_t, a)$ and
> $\quad\quad\quad$ $\pi(a|s_t) = \begin{cases} 1 - \frac{|\mathcal{A}(s_t)|-1}{|\mathcal{A}(s_t)|}\epsilon, & a = a^* \\ \frac{1}{|\mathcal{A}(s_t)|}\epsilon, & a \neq a^* \end{cases}$

### Convergence and Optimality

- **With sufficient samples**: MC $\epsilon$-Greedy converges to an $\epsilon$-greedy policy that is **optimal within $\Pi_\epsilon$** (the set of all $\epsilon$-greedy policies with the given $\epsilon$).
- **Not globally optimal**: The converged policy is generally **not** optimal in $\Pi$ (the set of all policies).
- **Small $\epsilon$ approximation**: If $\epsilon$ is sufficiently small, the optimal policies in $\Pi_\epsilon$ are close to the globally optimal ones in $\Pi$.
- MC $\epsilon$-Greedy **still requires** visiting all state-action pairs, but achieves this through the exploration inherent in $\epsilon$-greedy policies rather than through the exploring starts condition.

### 5.4.3 Illustrative Example

**Setup**: 5x5 grid world, $r_\text{boundary} = r_\text{forbidden} = -1$, $r_\text{target} = 1$, $\gamma = 0.9$, $\epsilon = 0.5$.

**Procedure**: In every iteration, generate a **single episode of 1 million steps** using the current policy, then update the policy using that single episode.

**Results** (Figure 5.5):
- **Initial policy**: Uniform distribution (each action has probability 0.2)
- **After 1st iteration**: Policy significantly improves
- **After 2nd iteration**: Optimal $\epsilon$-greedy policy is obtained

**Key insight**: Even with a single (but very long) episode per iteration, the algorithm works because $\epsilon$-greedy policies visit all state-action pairs sufficiently.

---

## 5.5 Exploration and Exploitation of $\epsilon$-Greedy Policies

### Fundamental Tradeoff

**Exploration**: The policy takes as many different actions as possible so that all actions can be visited and evaluated well.

**Exploitation**: The improved policy takes the **greedy action** (greatest action value) to maximize return.

**Tension**: Action values obtained at the current moment may be inaccurate due to insufficient exploration. The agent should keep exploring while conducting exploitation to avoid missing optimal actions.

**$\epsilon$-greedy provides a balance**: Higher probability on the greedy action (exploitation) combined with nonzero probability on all actions (exploration).

### Optimality of $\epsilon$-Greedy Policies

**Setup**: 5x5 grid world, $r_\text{boundary} = -1$, $r_\text{forbidden} = -10$, $r_\text{target} = 1$, $\gamma = 0.9$.

#### State Values of Consistent $\epsilon$-Greedy Policies (Figure 5.6)

Policies are **consistent** if the actions with the greatest probabilities are the same across different $\epsilon$ values.

| $\epsilon$ | Target state value | Overall trend |
|---|---|---|
| 0 (greedy) | 10.0 | All values positive and large |
| 0.1 | 3.4 | Values significantly reduced |
| 0.2 | $-2.5$ | Some values become negative |
| 0.5 | $-15.3$ | All values strongly negative |

**Key observation**: As $\epsilon$ increases, state values **decrease** -- the optimality of $\epsilon$-greedy policies becomes worse.

**Why the target state value can become negative**: When $\epsilon$ is large, the agent starting from the target area has a high probability of entering surrounding forbidden areas and receiving negative rewards ($r_\text{forbidden} = -10$).

#### Optimal $\epsilon$-Greedy Policies Are Not Always Consistent (Figure 5.7)

| $\epsilon$ | Consistency with greedy optimal? | Notes |
|---|---|---|
| 0 | Yes (it is the greedy optimal) | Target state value = 10.0 |
| 0.1 | Yes | Greedy actions match, but values lower (target = 3.4) |
| 0.2 | **No** | Some greedy actions differ from the $\epsilon=0$ optimal |
| 0.5 | **No** | Significantly different policy structure |

**Why inconsistency occurs at the target state**: In the greedy case, the optimal policy at the target is to **stay still** and collect positive rewards. However, when $\epsilon$ is large, there is a high chance of entering forbidden areas. Therefore, the optimal $\epsilon$-greedy policy at the target is to **escape** rather than stay.

**Conclusion**: To obtain $\epsilon$-greedy policies consistent with the optimal greedy policy, $\epsilon$ must be sufficiently small.

### Exploration Abilities of $\epsilon$-Greedy Policies (Figure 5.8)

#### $\epsilon = 1$ (Uniform Distribution -- Strongest Exploration)

Starting from $(s_1, a_1)$:
- **100 steps**: Visits a limited region
- **1,000 steps**: Visits most of the grid
- **10,000 steps**: Visits all state-action pairs
- **1 million steps**: All $9 \times 5 = 45$ (or $25 \times 5 = 125$ in 5x5 grid) state-action pairs visited approximately **evenly** (~7,600-8,300 times each)

#### $\epsilon = 0.5$ (Moderate Exploration)

Starting from $(s_1, a_1)$:
- All state-action pairs can still be visited when the episode is sufficiently long
- **1 million steps**: The distribution of visits is **extremely uneven** -- some actions visited $>250{,}000$ times while most are visited only hundreds or tens of times

### Practical Technique: Decaying $\epsilon$

Initially set $\epsilon$ to be **large** to enhance exploration, then **gradually reduce** it to ensure the optimality of the final policy.

---

## 5.6 Summary

The three MC algorithms form a progression:

| Algorithm | Sample Strategy | Policy Type | Exploring Starts? | Convergence |
|---|---|---|---|---|
| MC Basic | Initial-visit | Greedy | Required | Optimal policy (with sufficient samples) |
| MC Exploring Starts | Every-visit | Greedy | Required | Optimal policy (with sufficient samples) |
| MC $\epsilon$-Greedy | Every-visit | $\epsilon$-greedy | **Not required** | Optimal within $\Pi_\epsilon$ |

**Key progression**:
- MC Basic reveals the core idea
- MC Exploring Starts improves sample efficiency
- MC $\epsilon$-Greedy removes exploring starts at the cost of optimality within $\Pi_\epsilon$ only

**Slide summary**: While the basic idea is simple (replace model-based evaluation with MC estimation), complications appear when we want better performance. It is important to split the core idea from the complications.

---

## 5.7 Q&A -- Important Clarifications

### Q: What is Monte Carlo estimation?
**A**: A broad class of techniques that use stochastic samples to solve approximation problems.

### Q: What is the mean estimation problem?
**A**: Calculating the expected value of a random variable based on stochastic samples.

### Q: How to solve the mean estimation problem?
**A**: Two approaches. Model-based: use the probability distribution definition $\mathbb{E}[X] = \sum_x p(x) x$. Model-free: use MC estimation $\mathbb{E}[X] \approx \frac{1}{n}\sum_{i=1}^n x_i$, accurate when $n$ is large.

### Q: Why is mean estimation important for RL?
**A**: State and action values are both defined as expected values of returns. Estimating them is a mean estimation problem.

### Q: What is the core idea of model-free MC-based RL?
**A**: Convert policy iteration to a model-free algorithm by replacing the model-based policy evaluation step with an MC-based policy evaluation step that estimates action values from experience.

### Q: What are initial-visit, first-visit, and every-visit strategies?
**A**: Different strategies for utilizing samples within an episode:
- **Initial-visit**: Use the entire episode only for the initial state-action pair.
- **First-visit**: For each state-action pair, use only the first time it appears.
- **Every-visit**: Use every visit of each state-action pair.

### Q: What is exploring starts? Why is it important?
**A**: Requires sufficiently many episodes starting from **every** state-action pair. Theoretically necessary to find optimal policies: only if every action value is well explored can we accurately evaluate and correctly select optimal actions.

### Q: What is the idea used to avoid exploring starts?
**A**: Make policies **soft** (e.g., $\epsilon$-greedy). Soft policies are stochastic, enabling a single long episode to visit many state-action pairs, eliminating the need for many starting configurations.

### Q: Can an $\epsilon$-greedy policy be optimal?
**A**: Yes and no. Yes: MC $\epsilon$-Greedy converges to an $\epsilon$-greedy policy optimal within $\Pi_\epsilon$. No: this policy is generally not optimal among **all** policies (i.e., in $\Pi$).

### Q: Is it possible to use one episode to visit all state-action pairs?
**A**: Yes, if the policy is soft (e.g., $\epsilon$-greedy) and the episode is sufficiently long.

### Q: What is the relationship between MC Basic, MC Exploring Starts, and MC $\epsilon$-Greedy?
**A**: MC Basic is the simplest, revealing the fundamental idea. MC Exploring Starts improves sample usage (every-visit, episode-by-episode updates). MC $\epsilon$-Greedy further removes the exploring starts requirement using soft policies. The core idea is simple; complications arise from efficiency improvements.

---

## Concept Index

| Concept | Notation / Definition | Section |
|---|---|---|
| Monte Carlo estimation | Techniques using stochastic samples for approximation | 5.1 |
| Mean estimation (model-based) | $\mathbb{E}[X] = \sum_x p(x) x$ | 5.1 |
| Mean estimation (model-free) | $\mathbb{E}[X] \approx \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ | 5.1 |
| Law of large numbers | $\mathbb{E}[\bar{x}] = \mathbb{E}[X]$, $\text{var}[\bar{x}] = \frac{1}{n}\text{var}[X]$ | 5.1 |
| i.i.d. samples | Independent and identically distributed | 5.1 |
| Action value (model-based) | $q_\pi(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_\pi(s')$ | 5.2.1 |
| Action value (model-free) | $q_\pi(s,a) = \mathbb{E}[G_t|S_t=s, A_t=a] \approx \frac{1}{n}\sum g^{(i)}(s,a)$ | 5.2.1 |
| MC Basic algorithm | Algorithm 5.1: model-free variant of policy iteration | 5.2.2 |
| Episode length | Must be sufficiently long to reach target | 5.2.3 |
| Sparse reward | No positive reward unless target is reached | 5.2.3 |
| Visit (of a state-action pair) | Each appearance in an episode | 5.3.1 |
| Initial-visit strategy | Use episode only for initial pair (MC Basic) | 5.3.1 |
| First-visit strategy | Count only first visit of each pair | 5.3.1 |
| Every-visit strategy | Count all visits of each pair | 5.3.1 |
| MC Exploring Starts algorithm | Algorithm 5.2: every-visit, episode-by-episode updates | 5.3.3 |
| Exploring starts condition | Episodes must start from every $(s,a)$ pair | 5.3.3 |
| Backward return computation | $g \leftarrow \gamma g + r_{t+1}$ iterating $t = T{-}1, \ldots, 0$ | 5.3.3 |
| Generalized policy iteration (GPI) | Framework of alternating evaluation and improvement | 5.3.2 |
| Soft policy | $\pi(a|s) > 0$ for all $a, s$ | 5.4.1 |
| $\epsilon$-greedy policy | Greedy with prob $1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|}$; others with prob $\frac{\epsilon}{|\mathcal{A}|}$ | 5.4.1 |
| MC $\epsilon$-Greedy algorithm | Algorithm 5.3: no exploring starts needed | 5.4.2 |
| Policy set $\Pi$ | Set of all possible policies | 5.4.2 |
| Policy set $\Pi_\epsilon$ | Set of all $\epsilon$-greedy policies (fixed $\epsilon$) | 5.4.2 |
| Exploration | Visiting diverse state-action pairs | 5.5 |
| Exploitation | Selecting greedy (best-known) action | 5.5 |
| Exploration-exploitation tradeoff | Balancing via $\epsilon$ | 5.5 |
| Consistent $\epsilon$-greedy policies | Same greedy actions across different $\epsilon$ | 5.5 |
| Decaying $\epsilon$ | Start large, reduce over time | 5.5 |

---

## Dependencies and Forward References

| This chapter concept | Depends on |
|---|---|
| Policy iteration (model-based foundation) | Ch 4, Section 4.2 |
| Bellman equation for state values | Ch 2 |
| State values $v_\pi(s)$, action values $q_\pi(s,a)$ | Ch 2 (definitions), Ch 3 (optimality) |
| Greedy policy improvement | Ch 3 (Bellman optimality equation), Ch 4 (policy improvement theorem) |
| Truncated policy iteration / GPI | Ch 4, Section 4.3 |
| Discount rate $\gamma$, returns $G_t$ | Ch 1 |

| This chapter concept | Used in / Extended by |
|---|---|
| MC estimation of action values | Ch 7 (TD methods replace MC with bootstrapping) |
| $\epsilon$-greedy exploration | Ch 7 (used in Sarsa, Q-learning), Ch 8 (value function approximation) |
| Exploration-exploitation tradeoff | Ch 7, Ch 8, Ch 9, Ch 10 (fundamental theme throughout) |
| Model-free paradigm | Ch 7-10 (all are model-free) |
| Generalized policy iteration (GPI) | Ch 7 (TD-based GPI), Ch 9 (policy gradient), Ch 10 (actor-critic) |
| Every-visit / first-visit strategies | Ch 6 (stochastic approximation formalizes convergence) |
| Backward return computation ($g \leftarrow \gamma g + r_{t+1}$) | Ch 7 (modified for TD targets) |
