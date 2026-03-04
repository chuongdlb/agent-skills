---
chapter: 9
title: Policy Gradient Methods
key_topics: [policy function representation, parameterized policy, softmax policy, average state value metric, average reward metric, stationary distribution, discounted total probability, policy gradient theorem, gradient of average value, gradient of average reward, log-derivative trick, Poisson equation, undiscounted case, stochastic gradient ascent, REINFORCE, Monte Carlo policy gradient, exploration-exploitation in policy gradient, on-policy sampling]
depends_on: [1, 2, 3, 6, 8]
required_by: [10]
---

# Chapter 9: Policy Gradient Methods

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 9, pp. 191-214
> Supplemented by: Lecture slides L9 (42 slides)
> Errata applied: Box 9.5 corrections for $\lim_{k\to\infty} P_\pi^k$ (see Section 9.3.2)

## Purpose and Context

This chapter introduces **policy gradient methods**, which represent a fundamental shift from value-based to **policy-based** methods. Instead of representing policies as tables, policies are represented as **parameterized functions** $\pi(a|s,\theta)$, and optimal policies are found by optimizing scalar metrics via gradient ascent.

**Position in the book**: Chapter 9 bridges value function approximation (Chapter 8) to actor-critic methods (Chapter 10). It is the first chapter in the book that is purely policy-based. The book's progression is: *value-based* (Chapters 2-8) -> *policy-based* (Chapter 9) -> *combined actor-critic* (Chapter 10).

**Key transitions in this chapter**:
- From **tabular policy** representation to **function** representation
- From **value-based** methods to **policy-based** methods
- From optimizing **every state value** to optimizing a **scalar metric**

**Three central questions answered**:
1. What metrics should be used to define optimal policies? (Section 9.2)
2. How to calculate the gradients of the metrics? (Section 9.3)
3. How to use experience samples to calculate the gradients? (Section 9.4)

---

## 9.1 Policy Representation: From Table to Function

### Tabular Representation (Review)

Previously in the book, policies were represented by tables. Each entry is indexed by a state and an action:

| | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| $s_1$ | $\pi(a_1|s_1)$ | $\pi(a_2|s_1)$ | $\pi(a_3|s_1)$ | $\pi(a_4|s_1)$ | $\pi(a_5|s_1)$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $s_9$ | $\pi(a_1|s_9)$ | $\pi(a_2|s_9)$ | $\pi(a_3|s_9)$ | $\pi(a_4|s_9)$ | $\pi(a_5|s_9)$ |

### Function Representation

Policies can be represented by **parameterized functions**:

$$\pi(a|s, \theta)$$

where $\theta \in \mathbb{R}^m$ is a parameter vector. Alternative notations include $\pi_\theta(a|s)$, $\pi_\theta(a,s)$, or $\pi(a,s,\theta)$.

**Function structures** (Figure 9.2 in the text):
- **(a)** Input: $(s, a)$, Output: $\pi(a|s,\theta)$ (scalar probability for one action)
- **(b)** Input: $s$, Output: $[\pi(a_1|s,\theta), \ldots, \pi(a_{|\mathcal{A}|}|s,\theta)]$ (probability vector for all actions)

The function may be, for example, a **neural network** whose input is $s$, output is the probability of each action, and parameters are $\theta$.

**Advantage**: When the state space is large, the tabular representation is inefficient in terms of storage and generalization. The function representation is more efficient for large state/action spaces and has stronger generalization abilities.

### Three Key Differences Between Tabular and Function Representations

| Aspect | Tabular Representation | Function Representation |
|---|---|---|
| **Defining optimal policies** | Optimal if it maximizes *every* state value | Optimal if it maximizes certain *scalar metrics* |
| **Updating policies** | Directly change table entries | Change the parameter vector $\theta$ |
| **Retrieving action probabilities** | Look up the corresponding table entry | Calculate $\pi(a|s,\theta)$ given the function structure and parameter |

### The Basic Idea of Policy Gradient

Suppose $J(\theta)$ is a scalar metric. Optimal policies are obtained by optimizing this metric via gradient ascent:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)$$

where $\nabla_\theta J$ is the gradient of $J$ with respect to $\theta$, $t$ is the time step, and $\alpha$ is the learning rate.

---

## 9.2 Metrics for Defining Optimal Policies

If a policy is represented by a function, there are two types of metrics for defining optimal policies: one based on **state values** and the other based on **immediate rewards**.

### Metric 1: Average State Value ($\bar{v}_\pi$)

**Definition**: The average state value (or simply **average value**) is:

$$\bar{v}_\pi = \sum_{s \in \mathcal{S}} d(s) v_\pi(s)$$

where $d(s)$ is the weight of state $s$, satisfying $d(s) \geq 0$ for all $s \in \mathcal{S}$ and $\sum_{s \in \mathcal{S}} d(s) = 1$.

**Interpretation**: Since $d(s)$ is a probability distribution, the metric can be written as:

$$\bar{v}_\pi = \mathbb{E}_{S \sim d}[v_\pi(S)]$$

**Vector form**: Let $v_\pi = [\ldots, v_\pi(s), \ldots]^T \in \mathbb{R}^{|\mathcal{S}|}$ and $d = [\ldots, d(s), \ldots]^T \in \mathbb{R}^{|\mathcal{S}|}$. Then:

$$\bar{v}_\pi = d^T v_\pi$$

#### How to Select the Distribution $d$?

**Case 1: $d$ is independent of the policy $\pi$.**
- Denote $d$ as $d_0$ and $\bar{v}_\pi$ as $\bar{v}_\pi^0$ to indicate independence from the policy.
- *Equal weighting*: $d_0(s) = 1/|\mathcal{S}|$ treats all states as equally important.
- *Single start state*: If we only care about a specific state $s_0$ (e.g., the agent always starts from $s_0$):
  $$d_0(s_0) = 1, \quad d_0(s \neq s_0) = 0$$
  In this case, $\bar{v}_\pi = v_\pi(s_0)$.
- **Advantage of this case**: The gradient is easier to calculate because $\nabla_\theta \bar{v}_\pi^0 = d_0^T \nabla_\theta v_\pi$ (the distribution does not depend on $\theta$).

**Case 2: $d$ depends on the policy $\pi$.**
- A common choice is $d = d_\pi$, the **stationary distribution** under $\pi$, satisfying:
  $$d_\pi^T P_\pi = d_\pi^T$$
  where $P_\pi$ is the state transition probability matrix.
- **Interpretation**: The stationary distribution reflects the long-term behavior of the MDP under $\pi$. Frequently visited states receive higher weight; rarely visited states receive lower weight.
- More information about stationary distributions is in Box 8.1 (Chapter 8).

#### Equivalent Expression via Discounted Return

A commonly seen metric in the literature is:

$$J(\theta) = \lim_{n \to \infty} \mathbb{E}\left[\sum_{t=0}^{n} \gamma^t R_{t+1}\right] = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right] \tag{9.1}$$

**Claim**: This metric equals $\bar{v}_\pi$. **Proof**:

$$\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right] = \sum_{s \in \mathcal{S}} d(s) \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s\right] = \sum_{s \in \mathcal{S}} d(s) v_\pi(s) = \bar{v}_\pi$$

The first equality is due to the **law of total expectation**. The second equality is by the **definition of state values**.

### Metric 2: Average Reward ($\bar{r}_\pi$)

**Definition**: The average one-step reward (or simply **average reward**) is:

$$\bar{r}_\pi \doteq \sum_{s \in \mathcal{S}} d_\pi(s) r_\pi(s) = \mathbb{E}_{S \sim d_\pi}[r_\pi(S)] \tag{9.2}$$

where $d_\pi$ is the stationary distribution and:

$$r_\pi(s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s, \theta) r(s, a) = \mathbb{E}_{A \sim \pi(s,\theta)}[r(s, A) | s] \tag{9.3}$$

is the expected immediate reward at state $s$. Here, $r(s, a) \doteq \mathbb{E}[R|s, a] = \sum_r r \, p(r|s, a)$.

**Vector form**: Let $r_\pi = [\ldots, r_\pi(s), \ldots]^T \in \mathbb{R}^{|\mathcal{S}|}$ and $d_\pi = [\ldots, d_\pi(s), \ldots]^T \in \mathbb{R}^{|\mathcal{S}|}$. Then:

$$\bar{r}_\pi = d_\pi^T r_\pi$$

**Key distinction from $\bar{v}_\pi$**: The average reward $\bar{r}_\pi$ always uses the stationary distribution $d_\pi$ (which depends on $\pi$), while $\bar{v}_\pi$ can use either a policy-independent or policy-dependent distribution.

#### Equivalent Expression via Cesaro Mean of Rewards

A commonly seen metric in the literature is:

$$J(\theta) = \lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1}\right] \tag{9.4}$$

**Claim**: This metric equals $\bar{r}_\pi$:

$$\lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1}\right] = \sum_{s \in \mathcal{S}} d_\pi(s) r_\pi(s) = \bar{r}_\pi \tag{9.5}$$

#### Proof of Equation (9.5) (Box 9.1)

**Step 1**: Show that for any starting state $s_0 \in \mathcal{S}$:

$$\bar{r}_\pi = \lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1} \mid S_0 = s_0\right] \tag{9.6}$$

First, note:

$$\lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1} \mid S_0 = s_0\right] = \lim_{n \to \infty} \frac{1}{n} \sum_{t=0}^{n-1} \mathbb{E}[R_{t+1} | S_0 = s_0] = \lim_{t \to \infty} \mathbb{E}[R_{t+1} | S_0 = s_0] \tag{9.7}$$

The last equality uses the **Cesaro mean** property: if $\{a_k\}_{k=1}^\infty$ converges with $\lim_{k \to \infty} a_k$ existing, then $\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^n a_k = \lim_{k \to \infty} a_k$.

By the law of total expectation:

$$\mathbb{E}[R_{t+1} | S_0 = s_0] = \sum_{s \in \mathcal{S}} \mathbb{E}[R_{t+1} | S_t = s] \, p^{(t)}(s|s_0) = \sum_{s \in \mathcal{S}} r_\pi(s) \, p^{(t)}(s|s_0)$$

where $p^{(t)}(s|s_0)$ is the probability of transitioning from $s_0$ to $s$ in exactly $t$ steps. The second equality uses the **Markov memoryless property**. Since $\lim_{t \to \infty} p^{(t)}(s|s_0) = d_\pi(s)$ by definition of the stationary distribution (the starting state $s_0$ does not matter):

$$\lim_{t \to \infty} \mathbb{E}[R_{t+1} | S_0 = s_0] = \sum_{s \in \mathcal{S}} r_\pi(s) d_\pi(s) = \bar{r}_\pi$$

**Step 2**: For an arbitrary state distribution $d$, by the law of total expectation:

$$\lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1}\right] = \sum_{s \in \mathcal{S}} d(s) \lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1} \mid S_0 = s\right] = \sum_{s \in \mathcal{S}} d(s) \bar{r}_\pi = \bar{r}_\pi$$

### Summary of the Two Metrics

| Metric | Expression 1 | Expression 2 | Expression 3 |
|---|---|---|---|
| $\bar{v}_\pi$ | $\sum_{s \in \mathcal{S}} d(s) v_\pi(s)$ | $\mathbb{E}_{S \sim d}[v_\pi(S)]$ | $\lim_{n \to \infty} \mathbb{E}\left[\sum_{t=0}^{n} \gamma^t R_{t+1}\right]$ |
| $\bar{r}_\pi$ | $\sum_{s \in \mathcal{S}} d_\pi(s) r_\pi(s)$ | $\mathbb{E}_{S \sim d_\pi}[r_\pi(S)]$ | $\lim_{n \to \infty} \frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1}\right]$ |

### Key Remarks About the Metrics

1. **All metrics are functions of $\theta$**: Since $\pi$ is parameterized by $\theta$, different values of $\theta$ generate different metric values. The goal is to search for optimal $\theta$ to maximize the metric.

2. **Discounted vs. undiscounted**: Metrics can be defined with $\gamma \in (0,1)$ (discounted) or $\gamma = 1$ (undiscounted). The undiscounted case is nontrivial. The text primarily considers the discounted case.

3. **Equivalence of $\bar{r}_\pi$ and $\bar{v}_\pi$**: In the discounted case where $\gamma < 1$:

$$\bar{r}_\pi = (1 - \gamma) \bar{v}_\pi$$

This means they can be **simultaneously maximized**. The proof is given in Lemma 9.1 below.

4. **Notation convention**: $\bar{v}_\pi$ specifically refers to the case where the state distribution is the stationary distribution $d_\pi$; $\bar{v}_\pi^0$ refers to the case where $d_0$ is independent of $\pi$.

---

## 9.3 Gradients of the Metrics

The derivation of gradients is the **most complicated part** of the policy gradient method, because we must distinguish:
- Different metrics: $\bar{v}_\pi^0$, $\bar{v}_\pi$, $\bar{r}_\pi$
- Discounted ($\gamma < 1$) vs. undiscounted ($\gamma = 1$) cases

### Theorem 9.1 (Policy Gradient Theorem) -- Master Result

**Statement**: The gradient of $J(\theta)$ is:

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta) \, q_\pi(s, a) \tag{9.8}$$

where $\eta$ is a state distribution and $\nabla_\theta \pi$ is the gradient of $\pi$ with respect to $\theta$. Moreover, (9.8) has a compact form expressed in terms of expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi(S,\theta)}\left[\nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right] \tag{9.9}$$

where $\ln$ is the natural logarithm.

**Important remarks**:
- Theorem 9.1 is a **summary** of results in Theorems 9.2, 9.3, and 9.5. These three theorems address different scenarios involving different metrics and discounted/undiscounted cases.
- $J(\theta)$ could be $\bar{v}_\pi^0$, $\bar{v}_\pi$, or $\bar{r}_\pi$.
- The "$=$" in (9.8) may denote strict equality, approximation, or proportional to, depending on the specific scenario.
- The distribution $\eta$ varies across scenarios.
- For many readers, it is sufficient to be familiar with Theorem 9.1 without knowing the proof.

### Why Can (9.8) Be Written as (9.9)? -- The Log-Derivative Trick

The gradient of $\ln \pi(a|s,\theta)$ is:

$$\nabla_\theta \ln \pi(a|s, \theta) = \frac{\nabla_\theta \pi(a|s, \theta)}{\pi(a|s, \theta)}$$

Therefore:

$$\nabla_\theta \pi(a|s, \theta) = \pi(a|s, \theta) \nabla_\theta \ln \pi(a|s, \theta) \tag{9.11}$$

Substituting (9.11) into (9.8):

$$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \pi(a|s,\theta) \nabla_\theta \ln \pi(a|s,\theta) \, q_\pi(s,a) = \mathbb{E}_{S \sim \eta, A \sim \pi(S,\theta)}\left[\nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right]$$

**Why is expression (9.9) useful?** Because it is an expectation, and we can use samples to approximate it via stochastic gradient methods:

$$\nabla_\theta J \approx \nabla_\theta \ln \pi(a|s, \theta) \, q_\pi(s, a)$$

where $s, a$ are samples. This is the core idea of **stochastic gradient ascent** for policy optimization.

### Requirement: $\pi(a|s,\theta) > 0$ for All $(s,a)$

The log-derivative trick requires $\pi(a|s,\theta) > 0$ for all $(s,a)$ to ensure $\ln \pi(a|s,\theta)$ is well-defined. This is achieved using the **softmax function**:

$$\pi(a|s, \theta) = \frac{e^{h(s, a, \theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s, a', \theta)}}, \quad a \in \mathcal{A} \tag{9.12}$$

where $h(s, a, \theta)$ is a preference function for selecting action $a$ at state $s$.

**Properties of the softmax policy**:
- $\pi(a|s,\theta) \in (0, 1)$ for all $a$
- $\sum_{a \in \mathcal{A}} \pi(a|s,\theta) = 1$ for all $s$
- The policy is **stochastic** and hence **exploratory** (since $\pi(a|s,\theta) > 0$ for all $a$)
- Can be realized by a neural network with a **softmax output layer** (see Figure 9.2(b))
- The action is not deterministically chosen; instead it is **sampled** from the policy's probability distribution

**Slide insight**: There also exist **deterministic policy gradient (DPG)** methods, which are studied in the next chapter (Chapter 10, actor-critic methods).

---

### 9.3.1 Derivation of the Gradients in the Discounted Case

In the discounted case where $\gamma \in (0, 1)$, the state value and action value are:

$$v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t = s]$$

$$q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t = s, A_t = a]$$

It holds that $v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s,\theta) q_\pi(s,a)$ and the state value satisfies the Bellman equation.

#### Lemma 9.1 (Equivalence Between $\bar{v}_\pi$ and $\bar{r}_\pi$)

**Statement**: In the discounted case where $\gamma \in (0, 1)$:

$$\bar{r}_\pi = (1 - \gamma) \bar{v}_\pi \tag{9.13}$$

**Proof**: Note that $\bar{v}_\pi = d_\pi^T v_\pi$ and $\bar{r}_\pi = d_\pi^T r_\pi$, where $v_\pi$ and $r_\pi$ satisfy the Bellman equation $v_\pi = r_\pi + \gamma P_\pi v_\pi$. Multiplying $d_\pi^T$ on both sides:

$$\bar{v}_\pi = \bar{r}_\pi + \gamma d_\pi^T P_\pi v_\pi = \bar{r}_\pi + \gamma d_\pi^T v_\pi = \bar{r}_\pi + \gamma \bar{v}_\pi$$

where we used $d_\pi^T P_\pi = d_\pi^T$ (definition of stationary distribution). Rearranging gives $(1 - \gamma) \bar{v}_\pi = \bar{r}_\pi$.

#### Lemma 9.2 (Gradient of $v_\pi(s)$)

**Statement**: In the discounted case, for any $s \in \mathcal{S}$:

$$\nabla_\theta v_\pi(s) = \sum_{s' \in \mathcal{S}} \text{Pr}_\pi(s'|s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s', \theta) \, q_\pi(s', a) \tag{9.14}$$

where:

$$\text{Pr}_\pi(s'|s) \doteq \sum_{k=0}^{\infty} \gamma^k [P_\pi^k]_{ss'} = \left[(I_n - \gamma P_\pi)^{-1}\right]_{ss'}$$

is the **discounted total probability** of transitioning from $s$ to $s'$ under policy $\pi$. Here, $[P_\pi^k]_{ss'}$ is the probability of transitioning from $s$ to $s'$ using exactly $k$ steps under $\pi$.

**Proof (Box 9.2)**: For any $s \in \mathcal{S}$:

$$\nabla_\theta v_\pi(s) = \nabla_\theta \left[\sum_{a \in \mathcal{A}} \pi(a|s,\theta) q_\pi(s,a)\right] = \sum_{a \in \mathcal{A}} \left[\nabla_\theta \pi(a|s,\theta) q_\pi(s,a) + \pi(a|s,\theta) \nabla_\theta q_\pi(s,a)\right] \tag{9.15}$$

where $q_\pi(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s')$. Since $r(s,a)$ is independent of $\theta$:

$$\nabla_\theta q_\pi(s,a) = \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \nabla_\theta v_\pi(s')$$

Substituting back:

$$\nabla_\theta v_\pi(s) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a) + \gamma \sum_{s' \in \mathcal{S}} \left[\sum_{a \in \mathcal{A}} \pi(a|s,\theta) p(s'|s,a)\right] \nabla_\theta v_\pi(s') \tag{9.16}$$

Define $u(s) \doteq \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)$. Since $\sum_{a} \pi(a|s,\theta) p(s'|s,a) = [P_\pi]_{ss'}$, equation (9.16) becomes the matrix-vector equation:

$$\nabla_\theta v_\pi = u + \gamma (P_\pi \otimes I_m) \nabla_\theta v_\pi$$

where $n = |\mathcal{S}|$, $m$ is the dimension of $\theta$, and $\otimes$ is the Kronecker product (which appears because $\nabla_\theta v_\pi(s)$ is a vector). Solving this linear equation:

$$\nabla_\theta v_\pi = (I_{nm} - \gamma P_\pi \otimes I_m)^{-1} u = \left((I_n - \gamma P_\pi)^{-1} \otimes I_m\right) u \tag{9.17}$$

For any state $s$:

$$\nabla_\theta v_\pi(s) = \sum_{s' \in \mathcal{S}} \left[(I_n - \gamma P_\pi)^{-1}\right]_{ss'} u(s') = \sum_{s' \in \mathcal{S}} \left[(I_n - \gamma P_\pi)^{-1}\right]_{ss'} \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s',\theta) q_\pi(s',a) \tag{9.18}$$

**Probabilistic interpretation** of $[(I_n - \gamma P_\pi)^{-1}]_{ss'}$:

Since $(I_n - \gamma P_\pi)^{-1} = I + \gamma P_\pi + \gamma^2 P_\pi^2 + \cdots$, we have:

$$\left[(I_n - \gamma P_\pi)^{-1}\right]_{ss'} = [I]_{ss'} + \gamma [P_\pi]_{ss'} + \gamma^2 [P_\pi^2]_{ss'} + \cdots = \sum_{k=0}^{\infty} \gamma^k [P_\pi^k]_{ss'}$$

This is the **discounted total probability** of transitioning from $s$ to $s'$ using any number of steps.

#### Theorem 9.2 (Gradient of $\bar{v}_\pi^0$ in the Discounted Case)

**Statement**: In the discounted case where $\gamma \in (0,1)$, the gradient of $\bar{v}_\pi^0 = d_0^T v_\pi$ is:

$$\nabla_\theta \bar{v}_\pi^0 = \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right]$$

where $S \sim \rho_\pi$ and $A \sim \pi(S, \theta)$. The state distribution $\rho_\pi$ is:

$$\rho_\pi(s) = \sum_{s' \in \mathcal{S}} d_0(s') \text{Pr}_\pi(s|s'), \quad s \in \mathcal{S} \tag{9.19}$$

where $\text{Pr}_\pi(s|s') = \sum_{k=0}^{\infty} \gamma^k [P_\pi^k]_{s's} = [(I - \gamma P_\pi)^{-1}]_{s's}$ is the discounted total probability of transitioning from $s'$ to $s$.

**Proof (Box 9.3)**: Since $d_0(s)$ is independent of $\pi$:

$$\nabla_\theta \bar{v}_\pi^0 = \sum_{s \in \mathcal{S}} d_0(s) \nabla_\theta v_\pi(s)$$

Substituting Lemma 9.2:

$$= \sum_{s \in \mathcal{S}} d_0(s) \sum_{s' \in \mathcal{S}} \text{Pr}_\pi(s'|s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s',\theta) q_\pi(s',a)$$

$$= \sum_{s' \in \mathcal{S}} \underbrace{\left(\sum_{s \in \mathcal{S}} d_0(s) \text{Pr}_\pi(s'|s)\right)}_{\rho_\pi(s')} \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s',\theta) q_\pi(s',a)$$

$$= \sum_{s \in \mathcal{S}} \rho_\pi(s) \sum_{a \in \mathcal{A}} \pi(a|s,\theta) \nabla_\theta \ln \pi(a|s,\theta) q_\pi(s,a) = \mathbb{E}[\nabla_\theta \ln \pi(A|S,\theta) q_\pi(S,A)]$$

#### Theorem 9.3 (Gradients of $\bar{r}_\pi$ and $\bar{v}_\pi$ in the Discounted Case)

**Statement**: In the discounted case where $\gamma \in (0,1)$, the gradients of $\bar{r}_\pi$ and $\bar{v}_\pi$ are:

$$\nabla_\theta \bar{r}_\pi = (1 - \gamma) \nabla_\theta \bar{v}_\pi \approx \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a) = \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta) q_\pi(S, A)\right]$$

where $S \sim d_\pi$ and $A \sim \pi(S, \theta)$. The **approximation is more accurate when $\gamma$ is closer to 1**.

**Proof (Box 9.4)**: From the definition:

$$\nabla_\theta \bar{v}_\pi = \nabla_\theta \sum_{s \in \mathcal{S}} d_\pi(s) v_\pi(s) = \sum_{s \in \mathcal{S}} \nabla_\theta d_\pi(s) v_\pi(s) + \sum_{s \in \mathcal{S}} d_\pi(s) \nabla_\theta v_\pi(s) \tag{9.20}$$

For the **second term**, substituting (9.17):

$$\sum_{s \in \mathcal{S}} d_\pi(s) \nabla_\theta v_\pi(s) = (d_\pi^T \otimes I_m) \nabla_\theta v_\pi = (d_\pi^T \otimes I_m)\left((I_n - \gamma P_\pi)^{-1} \otimes I_m\right) u = \left(d_\pi^T (I_n - \gamma P_\pi)^{-1}\right) \otimes I_m \, u \tag{9.21}$$

A key identity is:

$$d_\pi^T (I_n - \gamma P_\pi)^{-1} = \frac{1}{1 - \gamma} d_\pi^T$$

(verified by multiplying $(I_n - \gamma P_\pi)$ on both sides and using $d_\pi^T P_\pi = d_\pi^T$).

Therefore, the second term of (9.20) becomes:

$$\frac{1}{1 - \gamma} \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)$$

Since the second term contains $\frac{1}{1-\gamma}$, when $\gamma \to 1$ this term **dominates** and the first term (involving $\nabla_\theta d_\pi$) becomes **negligible**. Therefore:

$$\nabla_\theta \bar{v}_\pi \approx \frac{1}{1 - \gamma} \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)$$

Using $\bar{r}_\pi = (1 - \gamma) \bar{v}_\pi$:

$$\nabla_\theta \bar{r}_\pi = (1 - \gamma) \nabla_\theta \bar{v}_\pi \approx \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a) = \mathbb{E}[\nabla_\theta \ln \pi(A|S,\theta) q_\pi(S,A)]$$

**Note**: The approximation requires that the first term of (9.20) does not go to infinity when $\gamma \to 1$. See [66, Section 4] for details.

---

### 9.3.2 Derivation of the Gradients in the Undiscounted Case

#### Why Study the Undiscounted Case?

The definition of the average reward $\bar{r}_\pi$ is valid for both discounted and undiscounted cases. While the gradient of $\bar{r}_\pi$ in the discounted case is an **approximation**, its gradient in the undiscounted case is **strictly valid** and more elegant.

#### State Values and the Poisson Equation

In the undiscounted case ($\gamma = 1$), the standard sum $\mathbb{E}[R_{t+1} + R_{t+2} + \cdots | S_t = s]$ may diverge. Therefore, state and action values are redefined using **differential values**:

$$v_\pi(s) \doteq \mathbb{E}[(R_{t+1} - \bar{r}_\pi) + (R_{t+2} - \bar{r}_\pi) + (R_{t+3} - \bar{r}_\pi) + \cdots | S_t = s]$$

$$q_\pi(s, a) \doteq \mathbb{E}[(R_{t+1} - \bar{r}_\pi) + (R_{t+2} - \bar{r}_\pi) + (R_{t+3} - \bar{r}_\pi) + \cdots | S_t = s, A_t = a]$$

These are also called **differential rewards** or **bias** in the literature.

The state value satisfies a Bellman-like equation:

$$v_\pi(s) = \sum_a \pi(a|s,\theta) \left[\sum_r p(r|s,a)(r - \bar{r}_\pi) + \sum_{s'} p(s'|s,a) v_\pi(s')\right] \tag{9.22}$$

Since $v_\pi(s) = \sum_a \pi(a|s,\theta) q_\pi(s,a)$, we have $q_\pi(s,a) = \sum_r p(r|s,a)(r - \bar{r}_\pi) + \sum_{s'} p(s'|s,a) v_\pi(s')$.

The **matrix-vector form** of (9.22) is:

$$v_\pi = r_\pi - \bar{r}_\pi \mathbf{1}_n + P_\pi v_\pi \tag{9.23}$$

where $\mathbf{1}_n = [1, \ldots, 1]^T \in \mathbb{R}^n$. Equation (9.23) is called the **Poisson equation**.

#### Theorem 9.4 (Solution of the Poisson Equation)

**Statement**: Let:

$$v_\pi^* = (I_n - P_\pi + \mathbf{1}_n d_\pi^T)^{-1} r_\pi \tag{9.24}$$

Then $v_\pi^*$ is a solution of the Poisson equation (9.23). Moreover, **any** solution of the Poisson equation has the form:

$$v_\pi = v_\pi^* + c \mathbf{1}_n$$

where $c \in \mathbb{R}$.

**Key insight**: The solution of the Poisson equation is **not unique** -- it is determined up to an additive constant.

#### Proof (Box 9.5, with errata corrections applied)

**Step 1**: Show that $v_\pi^*$ in (9.24) is a solution.

Let $A \doteq I_n - P_\pi + \mathbf{1}_n d_\pi^T$. Then $v_\pi^* = A^{-1} r_\pi$. Substituting into (9.23): $A^{-1} r_\pi = r_\pi - \mathbf{1}_n d_\pi^T r_\pi + P_\pi A^{-1} r_\pi$. This gives $(-A^{-1} + I_n - \mathbf{1}_n d_\pi^T + P_\pi) A^{-1} r_\pi = 0$, i.e., $(-I_n + A - \mathbf{1}_n d_\pi^T A + P_\pi) A^{-1} r_\pi = 0$. The bracket evaluates to $-I_n + (I_n - P_\pi + \mathbf{1}_n d_\pi^T) - \mathbf{1}_n d_\pi^T (I_n - P_\pi + \mathbf{1}_n d_\pi^T) + P_\pi = 0$.

**Step 2**: General expression of solutions.

Substituting $\bar{r}_\pi = d_\pi^T r_\pi$ into (9.23) gives $(I_n - P_\pi) v_\pi = (I_n - \mathbf{1}_n d_\pi^T) r_\pi$. Since $I_n - P_\pi$ is **singular** (because $(I_n - P_\pi)\mathbf{1}_n = 0$ for any $\pi$), the solution is not unique. If $v_\pi^*$ is a solution, then $v_\pi^* + x$ is also a solution for any $x \in \text{Null}(I_n - P_\pi)$. When $P_\pi$ is irreducible, $\text{Null}(I_n - P_\pi) = \text{span}\{\mathbf{1}_n\}$, so any solution has the form $v_\pi^* + c\mathbf{1}_n$.

**Step 3**: Show that $A = I_n - P_\pi + \mathbf{1}_n d_\pi^T$ is invertible. This is proven in Lemma 9.3.

#### Lemma 9.3 (Invertibility of $I_n - P_\pi + \mathbf{1}_n d_\pi^T$)

**Statement**: The matrix $I_n - P_\pi + \mathbf{1}_n d_\pi^T$ is invertible and its inverse is:

$$(I_n - (P_\pi - \mathbf{1}_n d_\pi^T))^{-1} = \sum_{k=1}^{\infty} (P_\pi^k - \mathbf{1}_n d_\pi^T) + I_n$$

**Proof**: The approach is to show $\lim_{k \to \infty} (P_\pi - \mathbf{1}_n d_\pi^T)^k = 0$, which implies the spectral radius $\rho(P_\pi - \mathbf{1}_n d_\pi^T) < 1$ and hence $I_n - (P_\pi - \mathbf{1}_n d_\pi^T)$ is invertible.

First, prove by induction that:

$$(P_\pi - \mathbf{1}_n d_\pi^T)^k = P_\pi^k - \mathbf{1}_n d_\pi^T, \quad k \geq 1 \tag{9.27}$$

*Base case ($k=1$)*: trivially true.

*Case $k=2$*:
$(P_\pi - \mathbf{1}_n d_\pi^T)^2 = P_\pi^2 - P_\pi \mathbf{1}_n d_\pi^T - \mathbf{1}_n d_\pi^T P_\pi + \mathbf{1}_n d_\pi^T \mathbf{1}_n d_\pi^T = P_\pi^2 - \mathbf{1}_n d_\pi^T$

using $P_\pi \mathbf{1}_n = \mathbf{1}_n$, $d_\pi^T P_\pi = d_\pi^T$, and $d_\pi^T \mathbf{1}_n = 1$.

**[ERRATA CORRECTION]** Since $d_\pi$ is the stationary distribution, the correct limit is:

$$\lim_{k \to \infty} P_\pi^k = \mathbf{1}_n d_\pi^T$$

(The original text had $d_\pi^T \mathbf{1}_n$ on the right-hand side, which is incorrect -- it should be $\mathbf{1}_n d_\pi^T$, an $n \times n$ matrix where each row equals $d_\pi^T$.)

Therefore, from (9.27):

$$\lim_{k \to \infty} (P_\pi - \mathbf{1}_n d_\pi^T)^k = \lim_{k \to \infty} P_\pi^k - \mathbf{1}_n d_\pi^T = 0$$

(The original text had $d_\pi^T \mathbf{1}_n$ in this expression as well; the corrected version uses $\mathbf{1}_n d_\pi^T$.)

Since $\rho(P_\pi - \mathbf{1}_n d_\pi^T) < 1$, the inverse is:

$$(I_n - (P_\pi - \mathbf{1}_n d_\pi^T))^{-1} = \sum_{k=0}^{\infty} (P_\pi - \mathbf{1}_n d_\pi^T)^k = I_n + \sum_{k=1}^{\infty} (P_\pi^k - \mathbf{1}_n d_\pi^T)$$

**Note**: The text observes that the result $(I_n - P_\pi + \mathbf{1}_n d_\pi^T)^{-1} = \sum_{k=0}^{\infty} (P_\pi^k - \mathbf{1}_n d_\pi^T)$ given in reference [66] is inaccurate because $\sum_{k=0}^{\infty}(P_\pi^k - \mathbf{1}_n d_\pi^T)$ is singular (since $\sum_{k=0}^{\infty}(P_\pi^k - \mathbf{1}_n d_\pi^T) \mathbf{1}_n = 0$). Lemma 9.3 corrects this inaccuracy.

#### Uniqueness of $\bar{r}_\pi$ Despite Non-Uniqueness of $v_\pi$

Although $v_\pi$ is not unique in the undiscounted case, $\bar{r}_\pi$ is unique:

$$\bar{r}_\pi \mathbf{1}_n = r_\pi + (P_\pi - I_n) v_\pi = r_\pi + (P_\pi - I_n)(v_\pi^* + c\mathbf{1}_n) = r_\pi + (P_\pi - I_n) v_\pi^*$$

The undetermined constant $c$ is canceled. Therefore, we can calculate the gradient of $\bar{r}_\pi$ in the undiscounted case. Since $v_\pi$ is not unique, $\bar{v}_\pi$ is also not unique, and its gradient in the undiscounted case is not studied.

#### Theorem 9.5 (Gradient of $\bar{r}_\pi$ in the Undiscounted Case)

**Statement**: In the undiscounted case, the gradient of the average reward $\bar{r}_\pi$ is:

$$\nabla_\theta \bar{r}_\pi = \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta) \, q_\pi(s, a) = \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta) \, q_\pi(S, A)\right] \tag{9.28}$$

where $S \sim d_\pi$ and $A \sim \pi(S, \theta)$.

**Key advantage over the discounted case**: Equation (9.28) is **strictly valid** (not an approximation) and $S$ obeys the stationary distribution $d_\pi$.

**Proof (Box 9.6)**: Starting from $v_\pi(s) = \sum_a \pi(a|s,\theta) q_\pi(s,a)$:

$$\nabla_\theta v_\pi(s) = \sum_{a \in \mathcal{A}} \left[\nabla_\theta \pi(a|s,\theta) q_\pi(s,a) + \pi(a|s,\theta) \nabla_\theta q_\pi(s,a)\right] \tag{9.29}$$

Since $q_\pi(s,a) = r(s,a) - \bar{r}_\pi + \sum_{s'} p(s'|s,a) v_\pi(s')$ and $r(s,a)$ is independent of $\theta$:

$$\nabla_\theta q_\pi(s,a) = -\nabla_\theta \bar{r}_\pi + \sum_{s'} p(s'|s,a) \nabla_\theta v_\pi(s')$$

Substituting back, with $u(s) \doteq \sum_a \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)$:

$$\nabla_\theta v_\pi(s) = u(s) - \nabla_\theta \bar{r}_\pi + \sum_{s'} [P_\pi]_{ss'} \nabla_\theta v_\pi(s') \tag{9.30}$$

In matrix-vector form:

$$\nabla_\theta v_\pi = u - \mathbf{1}_n \otimes \nabla_\theta \bar{r}_\pi + (P_\pi \otimes I_m) \nabla_\theta v_\pi$$

Rearranging: $\mathbf{1}_n \otimes \nabla_\theta \bar{r}_\pi = u + (P_\pi \otimes I_m) \nabla_\theta v_\pi - \nabla_\theta v_\pi$.

Multiplying $d_\pi^T \otimes I_m$ on both sides:

$$(d_\pi^T \mathbf{1}_n) \otimes \nabla_\theta \bar{r}_\pi = d_\pi^T \otimes I_m \, u + (d_\pi^T P_\pi) \otimes I_m \nabla_\theta v_\pi - d_\pi^T \otimes I_m \nabla_\theta v_\pi = d_\pi^T \otimes I_m \, u$$

since $d_\pi^T P_\pi = d_\pi^T$ (canceling the last two terms) and $d_\pi^T \mathbf{1}_n = 1$. Therefore:

$$\nabla_\theta \bar{r}_\pi = \sum_{s \in \mathcal{S}} d_\pi(s) u(s) = \sum_{s \in \mathcal{S}} d_\pi(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)$$

### Summary of Gradient Results Across Scenarios

| Scenario | Metric $J(\theta)$ | Gradient $\nabla_\theta J(\theta)$ | Distribution $\eta$ | Equality type |
|---|---|---|---|---|
| Discounted, $d_0$ indep. of $\pi$ | $\bar{v}_\pi^0$ | $\sum_s \eta(s) \sum_a \nabla_\theta \pi \, q_\pi$ | $\rho_\pi$ (discounted total prob.) | Exact (Thm 9.2) |
| Discounted, $d = d_\pi$ | $\bar{r}_\pi = (1-\gamma)\bar{v}_\pi$ | $\sum_s \eta(s) \sum_a \nabla_\theta \pi \, q_\pi$ | $d_\pi$ (stationary dist.) | Approximate (Thm 9.3) |
| Undiscounted | $\bar{r}_\pi$ | $\sum_s \eta(s) \sum_a \nabla_\theta \pi \, q_\pi$ | $d_\pi$ (stationary dist.) | Exact (Thm 9.5) |

All three cases share the same **form** of the gradient, which is why they are unified in Theorem 9.1.

---

## 9.4 Monte Carlo Policy Gradient (REINFORCE)

### From True Gradient to Stochastic Gradient

**Step 1**: The gradient-ascent algorithm for maximizing $J(\theta)$ is:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t) = \theta_t + \alpha \, \mathbb{E}\left[\nabla_\theta \ln \pi(A|S, \theta_t) \, q_\pi(S, A)\right] \tag{9.31}$$

where $\alpha > 0$ is a constant learning rate.

**Step 2**: Since the true gradient is unknown, replace it with a **stochastic gradient**:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \, q_t(s_t, a_t) \tag{9.32}$$

where $q_t(s_t, a_t)$ is an approximation of $q_\pi(s_t, a_t)$.

**Step 3**: If $q_t(s_t, a_t)$ is obtained by **Monte Carlo estimation**, the algorithm is called **REINFORCE** or **Monte Carlo policy gradient** -- one of the earliest and simplest policy gradient algorithms.

### Interpretation of the Algorithm

Since $\nabla_\theta \ln \pi(a_t|s_t, \theta_t) = \frac{\nabla_\theta \pi(a_t|s_t, \theta_t)}{\pi(a_t|s_t, \theta_t)}$, we can rewrite (9.32) as:

$$\theta_{t+1} = \theta_t + \alpha \underbrace{\frac{q_t(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}}_{\beta_t} \nabla_\theta \pi(a_t|s_t, \theta_t)$$

which can be written concisely as:

$$\theta_{t+1} = \theta_t + \alpha \beta_t \nabla_\theta \pi(a_t|s_t, \theta_t) \tag{9.33}$$

#### Interpretation 1: Direction of Policy Update

If $\alpha$ is sufficiently small, by Taylor expansion:

$$\pi(a_t|s_t, \theta_{t+1}) \approx \pi(a_t|s_t, \theta_t) + (\nabla_\theta \pi(a_t|s_t, \theta_t))^T (\theta_{t+1} - \theta_t) = \pi(a_t|s_t, \theta_t) + \alpha \beta_t \|\nabla_\theta \pi(a_t|s_t, \theta_t)\|^2$$

Therefore:
- If $\beta_t \geq 0$: the probability of choosing $(s_t, a_t)$ is **enhanced**: $\pi(a_t|s_t, \theta_{t+1}) \geq \pi(a_t|s_t, \theta_t)$. The greater $\beta_t$ is, the stronger the enhancement.
- If $\beta_t < 0$: the probability of choosing $(s_t, a_t)$ **decreases**: $\pi(a_t|s_t, \theta_{t+1}) < \pi(a_t|s_t, \theta_t)$.

#### Interpretation 2: Exploration-Exploitation Balance

The coefficient $\beta_t = \frac{q_t(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}$ naturally balances exploration and exploitation:

- **Exploitation**: $\beta_t$ is proportional to $q_t(s_t, a_t)$. If the action value is large, then $\pi(a_t|s_t, \theta_t)$ is enhanced. The algorithm **exploits** actions with greater values.

- **Exploration**: $\beta_t$ is inversely proportional to $\pi(a_t|s_t, \theta_t)$ (when $q_t(s_t, a_t) > 0$). If the probability of selecting $a_t$ is small, then $\pi(a_t|s_t, \theta_t)$ is enhanced. The algorithm **explores** actions with low probabilities.

### Sampling Requirements

Since (9.32) uses samples to approximate the true gradient in (9.31), the sampling strategy matters:

- **How to sample $S$?** $S$ in $\mathbb{E}[\nabla_\theta \ln \pi(A|S,\theta_t) q_\pi(S,A)]$ should obey the distribution $\eta$, which is either $d_\pi$ or $\rho_\pi$. Both represent the long-term behavior under $\pi$. In practice, people usually do not worry about sampling $S$ explicitly.

- **How to sample $A$?** $A$ should obey $\pi(A|S, \theta)$. Therefore, $a_t$ should be sampled following $\pi(a|s_t, \theta_t)$. This means **policy gradient methods are on-policy**.

**Practical note**: The ideal sampling methods are not strictly followed in practice due to low sample efficiency. A more sample-efficient implementation generates an entire episode first, then updates $\theta$ using every experience sample in the episode (Algorithm 9.1).

### Algorithm 9.1: REINFORCE (Policy Gradient by Monte Carlo)

**Initialization**: Initial parameter $\theta$; $\gamma \in (0,1)$; $\alpha > 0$.

**Goal**: Learn an optimal policy for maximizing $J(\theta)$.

```
For each episode, do:
    Generate an episode {s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T} following pi(theta)
    For t = 0, 1, ..., T-1:
        Value update:  q_t(s_t, a_t) = sum_{k=t+1}^{T} gamma^{k-t-1} r_k
        Policy update: theta <- theta + alpha * grad_theta(ln pi(a_t|s_t, theta)) * q_t(s_t, a_t)
```

**Key details**:
- The value update uses **Monte Carlo estimation**: the return from time $t$ onwards is $q_t(s_t, a_t) = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$
- The episode is generated first, then $\theta$ is updated multiple times (once per time step in the episode)
- The algorithm is **on-policy**: actions are sampled from the current policy $\pi(\theta)$

**Forward reference**: Many other policy gradient algorithms, including **actor-critic methods** (Chapter 10), can be obtained by extending REINFORCE.

---

## 9.5 Summary

This chapter introduced the policy gradient method, which is the foundation of many modern RL algorithms. Key points:

1. **Policy gradient methods are policy-based** -- a major shift from all previous value-based methods in the book.

2. **The basic idea is simple**: Select a scalar metric $J(\theta)$, derive its gradient $\nabla_\theta J(\theta)$, and optimize via gradient ascent.

3. **The most complicated part** is deriving the gradients, which requires distinguishing various scenarios (different metrics, discounted/undiscounted cases). Fortunately, all gradients share a similar form summarized in Theorem 9.1.

4. **The REINFORCE algorithm** (9.32) must be properly understood as it is the foundation of many advanced algorithms. Its concise form (9.33) reveals the exploration-exploitation mechanism.

5. **Next step**: In Chapter 10, the REINFORCE algorithm is extended to **actor-critic methods**.

---

## 9.6 Q&A -- Important Clarifications

### Q: What is the basic idea of the policy gradient method?
**A**: Define an appropriate scalar metric, derive its gradient, and use gradient-ascent methods to optimize the metric. The most important theoretical result is the policy gradient given in Theorem 9.1.

### Q: What is the most complicated part of the policy gradient method?
**A**: The derivation of the gradients is the most complicated part because we must distinguish numerous different scenarios. The mathematical derivation in each scenario is nontrivial. It is sufficient for many readers to be familiar with Theorem 9.1 without knowing the proof.

### Q: What metrics should be used?
**A**: Three common metrics: $\bar{v}_\pi$, $\bar{v}_\pi^0$, and $\bar{r}_\pi$. Since they all lead to similar policy gradients, any can be adopted. The expressions in (9.1) and (9.4) are often encountered in the literature.

### Q: Why is a natural logarithm function contained in the policy gradient?
**A**: The log function is introduced to express the gradient as an **expected value**. This enables approximating the true gradient with a stochastic one (using samples), which is the core of practical policy gradient algorithms.

### Q: Why study undiscounted cases when deriving the policy gradient?
**A**: The definition of $\bar{r}_\pi$ is valid for both discounted and undiscounted cases. While the gradient of $\bar{r}_\pi$ in the discounted case is an approximation, its gradient in the undiscounted case is more elegant and strictly valid.

### Q: What does the policy gradient algorithm do mathematically?
**A**: Examine its concise expression (9.33): it is a gradient-ascent algorithm for updating $\pi(a_t|s_t, \theta_t)$. When a sample $(s_t, a_t)$ is available, the policy is updated so that $\pi(a_t|s_t, \theta_{t+1}) \geq \pi(a_t|s_t, \theta_t)$ or $\pi(a_t|s_t, \theta_{t+1}) < \pi(a_t|s_t, \theta_t)$ depending on the sign of $\beta_t$.

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| Parameterized policy | $\pi(a|s,\theta)$, $\theta \in \mathbb{R}^m$ | 9.1 |
| Softmax policy | $\pi(a|s,\theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a'} e^{h(s,a',\theta)}}$ | 9.3 |
| Preference function | $h(s,a,\theta)$ | 9.3 |
| Average state value | $\bar{v}_\pi = \sum_s d(s) v_\pi(s) = d^T v_\pi$ | 9.2 |
| Average value (policy-independent $d_0$) | $\bar{v}_\pi^0 = d_0^T v_\pi$ | 9.2 |
| Average reward | $\bar{r}_\pi = \sum_s d_\pi(s) r_\pi(s) = d_\pi^T r_\pi$ | 9.2 |
| Expected immediate reward | $r_\pi(s) = \sum_a \pi(a|s,\theta) r(s,a)$ | 9.2 |
| Stationary distribution | $d_\pi^T P_\pi = d_\pi^T$ | 9.2 |
| Discounted total probability | $\text{Pr}_\pi(s'|s) = \sum_{k=0}^\infty \gamma^k [P_\pi^k]_{ss'} = [(I_n - \gamma P_\pi)^{-1}]_{ss'}$ | 9.3.1 |
| Discounted state distribution | $\rho_\pi(s) = \sum_{s'} d_0(s') \text{Pr}_\pi(s|s')$ | 9.3.1 |
| Policy gradient theorem | $\nabla_\theta J = \sum_s \eta(s) \sum_a \nabla_\theta \pi \, q_\pi$ | 9.3 |
| Log-derivative trick | $\nabla_\theta \pi = \pi \nabla_\theta \ln \pi$ | 9.3 |
| Compact gradient form | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \ln \pi(A|S,\theta) q_\pi(S,A)]$ | 9.3 |
| Differential state value (undiscounted) | $v_\pi(s) = \mathbb{E}[\sum_k (R_{t+k+1} - \bar{r}_\pi) | S_t = s]$ | 9.3.2 |
| Poisson equation | $v_\pi = r_\pi - \bar{r}_\pi \mathbf{1}_n + P_\pi v_\pi$ | 9.3.2 |
| REINFORCE algorithm | $\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t,\theta_t) q_t(s_t,a_t)$ | 9.4 |
| Concise REINFORCE form | $\theta_{t+1} = \theta_t + \alpha \beta_t \nabla_\theta \pi(a_t|s_t,\theta_t)$ | 9.4 |
| Update coefficient | $\beta_t = q_t(s_t,a_t) / \pi(a_t|s_t,\theta_t)$ | 9.4 |
| Cesaro mean | $\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^n a_k = \lim_{k \to \infty} a_k$ | 9.2 (Box 9.1) |
| Kronecker product in gradient derivation | $\nabla_\theta v_\pi = u + \gamma(P_\pi \otimes I_m) \nabla_\theta v_\pi$ | 9.3.1 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| Parameterized policy $\pi(a|s,\theta)$ | Ch 10 (actor-critic methods) |
| Policy gradient theorem (Thm 9.1) | Ch 10 (actor-critic gradient updates) |
| REINFORCE algorithm (Alg 9.1) | Ch 10 (extended to actor-critic by replacing MC with TD for value estimation) |
| $\beta_t$ interpretation (exploration-exploitation) | Ch 10 (baseline subtraction in actor-critic) |
| Average reward $\bar{r}_\pi$ and average value $\bar{v}_\pi$ | Ch 10 (choice of objective in actor-critic) |
| Softmax policy / neural network policy | Ch 10 (actor network) |
| Stationary distribution $d_\pi$ (from Ch 8, Box 8.1) | Used throughout Ch 9 in metric definitions |
| Bellman equation (from Ch 2) | Used in Lemma 9.1, 9.2 proofs |
| Stochastic approximation theory (from Ch 6) | Justifies convergence of stochastic gradient updates |
| State/action values $v_\pi$, $q_\pi$ (from Ch 2, 3) | Core quantities in gradient expressions |
| Poisson equation (undiscounted values) | Ch 10 (differential value functions in continuing tasks) |
| Deterministic policy gradient (DPG) | Ch 10 (actor-critic with deterministic policies) |
