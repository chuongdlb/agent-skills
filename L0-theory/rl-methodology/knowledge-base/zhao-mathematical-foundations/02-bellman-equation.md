---
chapter: 2
title: State Values and Bellman Equation
key_topics: [state value, state-value function, return, discounted return, bootstrapping, Bellman equation, elementwise form, matrix-vector form, policy evaluation, closed-form solution, iterative solution, state transition matrix, stochastic matrix, action value, action-value function, Gershgorin circle theorem, contraction mapping]
depends_on: [1]
required_by: [3, 4, 5, 7, 8, 9, 10]
---

# Chapter 2: State Values and Bellman Equation

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 2, pp. 15-34
> Supplemented by: Lecture slides L2 (52 slides)
> Errata: Section 2.7.1 -- Gershgorin radius corrected (absolute value was missing, negative sign removed); see Section 2.7.1 below

## Purpose and Context

This chapter introduces two core concepts and one fundamental tool:
- **Core concept 1**: The **state value**, defined as the expected (average) return an agent can obtain starting from a state under a given policy. State values serve as the metric for evaluating whether a policy is good or not.
- **Core concept 2**: The **action value**, defined as the expected return an agent can obtain starting from a state, taking a specific action, and then following the policy.
- **Fundamental tool**: The **Bellman equation**, a set of linear equations that describe the relationships between the values of all states. Solving the Bellman equation yields the state values, a process called **policy evaluation**.

**Position in the book**: Chapter 2 builds directly on the MDP framework from Chapter 1. The Bellman equation is the foundation for the Bellman optimality equation (Chapter 3), value/policy iteration (Chapter 4), and is used throughout all subsequent chapters.

**Key insight from slides**: The chapter has two motivating questions: (1) *Why are returns important?* -- because they can evaluate policies; (2) *How to calculate returns?* -- via the Bellman equation, which introduces the idea of bootstrapping.

---

## 2.1 Motivating Example 1: Why Are Returns Important?

Returns play a fundamental role in RL because they can evaluate whether a policy is good or not. This is demonstrated by comparing three policies on a simple four-state chain.

### Setup

Consider four states $s_1, s_2, s_3, s_4$ where $s_2$ is in a forbidden area and $s_3, s_4$ are target-like states. Three policies differ only at $s_1$:

- **Policy 1** (deterministic): At $s_1$, go down to $s_3$ (avoids forbidden area). Rewards: $r = 0$ from $s_1$, $r = 1$ from $s_3$, $r = 1$ from $s_4$, etc.
- **Policy 2** (deterministic): At $s_1$, go right to $s_2$ (enters forbidden area). Rewards: $r = -1$ from $s_1$, $r = 1$ from $s_2$ onward.
- **Policy 3** (stochastic): At $s_1$, go right with probability 0.5 or down with probability 0.5.

### Returns Calculation

Starting from $s_1$ with discount rate $\gamma \in (0,1)$:

**Policy 1**:
$$\text{return}_1 = 0 + \gamma \cdot 1 + \gamma^2 \cdot 1 + \cdots = \gamma(1 + \gamma + \gamma^2 + \cdots) = \frac{\gamma}{1-\gamma}$$

**Policy 2**:
$$\text{return}_2 = -1 + \gamma \cdot 1 + \gamma^2 \cdot 1 + \cdots = -1 + \frac{\gamma}{1-\gamma}$$

**Policy 3** (average of two possible trajectories):
$$\text{return}_3 = 0.5\left(-1 + \frac{\gamma}{1-\gamma}\right) + 0.5\left(\frac{\gamma}{1-\gamma}\right) = -0.5 + \frac{\gamma}{1-\gamma}$$

### Conclusion

$$\text{return}_1 > \text{return}_3 > \text{return}_2 \quad \text{for any } \gamma \in (0,1)$$

Policy 1 is the best (greatest return) and Policy 2 is the worst (smallest return). This mathematical conclusion matches the intuition that avoiding the forbidden area is better.

**Important note**: $\text{return}_3$ is not strictly a single return but an expected value (average of returns). As formalized later, $\text{return}_3$ is actually a **state value**.

---

## 2.2 Motivating Example 2: How to Calculate Returns?

### Method 1: By Definition (Direct Calculation)

Consider four states in a cycle $s_1 \to s_2 \to s_3 \to s_4 \to s_1 \to \cdots$ with rewards $r_1, r_2, r_3, r_4$ respectively. Let $v_i$ denote the return starting from $s_i$:

$$v_1 = r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots$$
$$v_2 = r_2 + \gamma r_3 + \gamma^2 r_4 + \cdots$$
$$v_3 = r_3 + \gamma r_4 + \gamma^2 r_1 + \cdots$$
$$v_4 = r_4 + \gamma r_1 + \gamma^2 r_2 + \cdots$$

### Method 2: Bootstrapping

By factoring out $\gamma$ from the tail of each expression:

$$v_1 = r_1 + \gamma v_2$$
$$v_2 = r_2 + \gamma v_3$$
$$v_3 = r_3 + \gamma v_4$$
$$v_4 = r_4 + \gamma v_1$$

**Bootstrapping**: The values rely on each other. $v_1$ relies on $v_2$, $v_2$ on $v_3$, $v_3$ on $v_4$, and $v_4$ on $v_1$. This means calculating unknown values from other unknown values. While seemingly circular, it is straightforward from a linear algebra perspective.

### Matrix-Vector Form

The bootstrapping equations can be written as:

$$\begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \end{bmatrix} = \begin{bmatrix} r_1 \\ r_2 \\ r_3 \\ r_4 \end{bmatrix} + \gamma \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \end{bmatrix}$$

Compactly: $v = r + \gamma P v$, which gives $v = (I - \gamma P)^{-1} r$.

This is the **Bellman equation** for this simple deterministic example. It demonstrates the core idea: the return obtained starting from one state depends on those obtained starting from other states.

**Slide insight**: The matrix-vector form makes it clear how to solve for the state values -- it is just a system of linear equations.

---

## 2.3 State Values

### Motivation

Returns can evaluate policies, but they are inapplicable to **stochastic** systems because starting from one state may lead to different returns. The concept of **state value** resolves this by taking the expectation (average) of all possible returns.

### Notation

At time $t$, the agent is in state $S_t$ and takes action $A_t$ following policy $\pi$. The next state is $S_{t+1}$ and the immediate reward is $R_{t+1}$:

$$S_t \xrightarrow{A_t} S_{t+1}, R_{t+1}$$

All of $S_t, S_{t+1}, A_t, R_{t+1}$ are **random variables**. Moreover, $S_t, S_{t+1} \in \mathcal{S}$, $A_t \in \mathcal{A}(S_t)$, and $R_{t+1} \in \mathcal{R}(S_t, A_t)$.

A trajectory starting from time $t$:

$$S_t \xrightarrow{A_t} S_{t+1}, R_{t+1} \xrightarrow{A_{t+1}} S_{t+2}, R_{t+2} \xrightarrow{A_{t+2}} S_{t+3}, R_{t+3} \cdots$$

The **discounted return** along the trajectory:

$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

where $\gamma \in (0,1)$ is the discount rate. $G_t$ is a random variable since $R_{t+1}, R_{t+2}, \ldots$ are random variables.

### Definition: State-Value Function

$$v_\pi(s) \doteq \mathbb{E}[G_t | S_t = s]$$

$v_\pi(s)$ is called the **state-value function** or simply the **state value** of $s$.

### Key Properties

| Property | Explanation |
|---|---|
| $v_\pi(s)$ depends on $s$ | It is a conditional expectation conditioned on starting from $S_t = s$ |
| $v_\pi(s)$ depends on $\pi$ | Trajectories are generated by following policy $\pi$; different policies yield different state values |
| $v_\pi(s)$ does **not** depend on $t$ | The value is determined once the policy is given (stationary model assumption) |

### Relationship Between State Values and Returns

| System type | Relationship |
|---|---|
| Both policy and model **deterministic** | Starting from a state always leads to the same trajectory. Return = state value |
| Policy or model is **stochastic** | Starting from same state may generate different trajectories with different returns. State value = **mean** of these returns |

**Key insight**: It is more formal to use state values (rather than raw returns) to evaluate policies. Policies that generate greater state values are better. State values are therefore a **core concept** in RL.

---

## 2.4 Bellman Equation (Derivation)

The Bellman equation is a mathematical tool for analyzing state values. It is a set of linear equations describing the relationships between the values of all states.

### Derivation

**Step 1**: Rewrite $G_t$ recursively:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = R_{t+1} + \gamma G_{t+1}$$

where $G_{t+1} = R_{t+2} + \gamma R_{t+3} + \cdots$

**Step 2**: Apply the definition of state value:

$$v_\pi(s) = \mathbb{E}[G_t | S_t = s] = \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s]$$
$$= \underbrace{\mathbb{E}[R_{t+1} | S_t = s]}_{\text{immediate reward term}} + \gamma \underbrace{\mathbb{E}[G_{t+1} | S_t = s]}_{\text{future reward term}}$$

### Term 1: Mean of Immediate Rewards

Using the law of total expectation:

$$\mathbb{E}[R_{t+1} | S_t = s] = \sum_{a \in \mathcal{A}} \pi(a|s) \mathbb{E}[R_{t+1} | S_t = s, A_t = a] = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a) \, r$$

### Term 2: Mean of Future Rewards

$$\mathbb{E}[G_{t+1} | S_t = s] = \sum_{s' \in \mathcal{S}} \mathbb{E}[G_{t+1} | S_t = s, S_{t+1} = s'] \, p(s'|s)$$

By the **Markov property**: $\mathbb{E}[G_{t+1} | S_t = s, S_{t+1} = s'] = \mathbb{E}[G_{t+1} | S_{t+1} = s'] = v_\pi(s')$

Therefore:

$$\mathbb{E}[G_{t+1} | S_t = s] = \sum_{s' \in \mathcal{S}} v_\pi(s') \, p(s'|s) = \sum_{s' \in \mathcal{S}} v_\pi(s') \sum_{a \in \mathcal{A}} p(s'|s,a) \, \pi(a|s)$$

### The Bellman Equation (Elementwise Form)

Combining Terms 1 and 2:

$$\boxed{v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ \sum_{r \in \mathcal{R}} p(r|s,a) \, r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \, v_\pi(s') \right], \quad \forall s \in \mathcal{S}}$$

This is the **Bellman equation**. It is valid for every state $s$, so it represents a **set** of $|\mathcal{S}|$ equations.

### Structure of the Bellman Equation

| Symbol | Role |
|---|---|
| $v_\pi(s)$ and $v_\pi(s')$ | Unknown state values to be calculated (bootstrapping!) |
| $\pi(a \mid s)$ | Given policy. Solving for state values = **policy evaluation** |
| $p(r \mid s,a)$ and $p(s' \mid s,a)$ | System model (dynamics). Known in model-based setting; unknown in model-free setting |

**Slide insight**: The equation has two clear parts -- the immediate reward term and the future reward term. Every state has an equation like this.

### Equivalent Expressions of the Bellman Equation

**Expression 2** (used in Sutton & Barto): Using joint probability $p(s',r|s,a)$:

$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s',r|s,a) \left[ r + \gamma v_\pi(s') \right]$$

**Expression 3** (when reward depends only on next state $s'$): Writing $r(s')$ and using $p(r(s')|s,a) = p(s'|s,a)$:

$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a) \left[ r(s') + \gamma v_\pi(s') \right]$$

---

## 2.5 Examples for Illustrating the Bellman Equation

### Example 1: Deterministic Policy

**Setup**: Four states $s_1, s_2, s_3, s_4$. Policy: $s_1 \to s_3$, $s_2 \to s_4$, $s_3 \to s_4$, $s_4 \to s_4$. Rewards: $r_1 = 0$, $r_2 = 1$, $r_3 = 1$, $r_4 = 1$.

**Writing the Bellman equations for each state**:

For $s_1$: $\pi(a_3|s_1) = 1$, $p(s_3|s_1,a_3) = 1$, $p(r=0|s_1,a_3) = 1$:
$$v_\pi(s_1) = 0 + \gamma v_\pi(s_3)$$

Similarly:
$$v_\pi(s_2) = 1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_3) = 1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_4) = 1 + \gamma v_\pi(s_4)$$

**Solving** (from last equation backward):
$$v_\pi(s_4) = \frac{1}{1-\gamma}, \quad v_\pi(s_3) = \frac{1}{1-\gamma}, \quad v_\pi(s_2) = \frac{1}{1-\gamma}, \quad v_\pi(s_1) = \frac{\gamma}{1-\gamma}$$

**With $\gamma = 0.9$**:
$$v_\pi(s_4) = 10, \quad v_\pi(s_3) = 10, \quad v_\pi(s_2) = 10, \quad v_\pi(s_1) = 9$$

### Example 2: Stochastic Policy

**Setup**: Same four states. Policy at $s_1$: go right to $s_2$ with probability 0.5, go down to $s_3$ with probability 0.5. All other states as before.

**Bellman equations**:
$$v_\pi(s_1) = 0.5[0 + \gamma v_\pi(s_3)] + 0.5[-1 + \gamma v_\pi(s_2)]$$
$$v_\pi(s_2) = 1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_3) = 1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_4) = 1 + \gamma v_\pi(s_4)$$

**Solving**:
$$v_\pi(s_4) = \frac{1}{1-\gamma}, \quad v_\pi(s_3) = \frac{1}{1-\gamma}, \quad v_\pi(s_2) = \frac{1}{1-\gamma}$$
$$v_\pi(s_1) = -0.5 + \frac{\gamma}{1-\gamma}$$

**With $\gamma = 0.9$**:
$$v_\pi(s_4) = 10, \quad v_\pi(s_3) = 10, \quad v_\pi(s_2) = 10, \quad v_\pi(s_1) = 8.5$$

### Comparing the Two Policies

For all $i = 1,2,3,4$: $v_{\pi_1}(s_i) \geq v_{\pi_2}(s_i)$

The deterministic policy (Example 1) is better because it has greater state values. This matches the intuition that the first policy avoids the forbidden area from $s_1$.

---

## 2.6 Matrix-Vector Form of the Bellman Equation

The elementwise Bellman equation is valid for every state. Combining all equations yields a concise matrix-vector form.

### Reformulation

First, rewrite the elementwise Bellman equation as:

$$v_\pi(s) = r_\pi(s) + \gamma \sum_{s' \in \mathcal{S}} p_\pi(s'|s) \, v_\pi(s')$$

where:

$$r_\pi(s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a) \, r$$

$$p_\pi(s'|s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \, p(s'|s,a)$$

Here $r_\pi(s)$ is the **mean of immediate rewards** at state $s$ under policy $\pi$, and $p_\pi(s'|s)$ is the **transition probability** from $s$ to $s'$ under policy $\pi$.

### Indexing States

Suppose states are indexed as $s_i$ with $i = 1, \ldots, n$ where $n = |\mathcal{S}|$. For state $s_i$:

$$v_\pi(s_i) = r_\pi(s_i) + \gamma \sum_{s_j \in \mathcal{S}} p_\pi(s_j|s_i) \, v_\pi(s_j)$$

### The Matrix-Vector Form

Define:
- $v_\pi = [v_\pi(s_1), \ldots, v_\pi(s_n)]^T \in \mathbb{R}^n$
- $r_\pi = [r_\pi(s_1), \ldots, r_\pi(s_n)]^T \in \mathbb{R}^n$
- $P_\pi \in \mathbb{R}^{n \times n}$ with $[P_\pi]_{ij} = p_\pi(s_j|s_i)$ (the **state transition matrix**)

Then:

$$\boxed{v_\pi = r_\pi + \gamma P_\pi v_\pi}$$

where $v_\pi$ is the unknown to be solved, and $r_\pi$, $P_\pi$ are known.

### Properties of the State Transition Matrix $P_\pi$

| Property | Mathematical statement | Meaning |
|---|---|---|
| Nonnegative | $P_\pi \geq 0$ | All entries are $\geq 0$ |
| Stochastic (row-stochastic) | $P_\pi \mathbf{1} = \mathbf{1}$ | Each row sums to 1 |

Here $\mathbf{1} = [1, \ldots, 1]^T$ and $\geq$ denotes elementwise comparison.

### Worked Example (Stochastic Policy, 4 States)

For the stochastic policy from Example 2 (Section 2.5):

$$\begin{bmatrix} v_\pi(s_1) \\ v_\pi(s_2) \\ v_\pi(s_3) \\ v_\pi(s_4) \end{bmatrix} = \begin{bmatrix} 0.5(0) + 0.5(-1) \\ 1 \\ 1 \\ 1 \end{bmatrix} + \gamma \begin{bmatrix} 0 & 0.5 & 0.5 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} v_\pi(s_1) \\ v_\pi(s_2) \\ v_\pi(s_3) \\ v_\pi(s_4) \end{bmatrix}$$

One can verify $P_\pi \mathbf{1} = \mathbf{1}$ (each row sums to 1).

---

## 2.7 Solving State Values from the Bellman Equation

Calculating state values of a given policy is the fundamental problem of **policy evaluation**. Two methods are presented.

### 2.7.1 Closed-Form Solution

Since $v_\pi = r_\pi + \gamma P_\pi v_\pi$ is a linear equation, its closed-form solution is:

$$\boxed{v_\pi = (I - \gamma P_\pi)^{-1} r_\pi}$$

#### Properties of $(I - \gamma P_\pi)^{-1}$

**Property 1: $I - \gamma P_\pi$ is invertible.**

*Proof*: By the **Gershgorin circle theorem**, every eigenvalue of $I - \gamma P_\pi$ lies within at least one Gershgorin disc. The $i$-th disc has:
- Center: $[I - \gamma P_\pi]_{ii} = 1 - \gamma p_\pi(s_i|s_i)$
- Radius: $\sum_{j \neq i} |[I - \gamma P_\pi]_{ij}| = \sum_{j \neq i} \gamma p_\pi(s_j|s_i)$

> **Errata correction**: The original text wrote the radius as $\sum_{j \neq i} [I - \gamma P_\pi]_{ij} = -\sum_{j \neq i} \gamma p_\pi(s_j|s_i)$. The corrected version uses absolute values: $\sum_{j \neq i} |[I - \gamma P_\pi]_{ij}| = \sum_{j \neq i} \gamma p_\pi(s_j|s_i)$. The off-diagonal entries of $I - \gamma P_\pi$ are $-\gamma p_\pi(s_j|s_i) \leq 0$, so taking absolute values removes the negative sign.

Since $\gamma < 1$, the radius is strictly less than the magnitude of the center:

$$\sum_{j \neq i} \gamma p_\pi(s_j|s_i) < 1 - \gamma p_\pi(s_i|s_i)$$

This holds because $\sum_{j \neq i} p_\pi(s_j|s_i) = 1 - p_\pi(s_i|s_i)$, so $\gamma(1 - p_\pi(s_i|s_i)) < 1 - \gamma p_\pi(s_i|s_i)$, which simplifies to $\gamma < 1$.

Therefore, no Gershgorin disc contains the origin, and hence no eigenvalue of $I - \gamma P_\pi$ is zero. The matrix is invertible.

**Property 2: $(I - \gamma P_\pi)^{-1} \geq I$.**

Every element of $(I - \gamma P_\pi)^{-1}$ is nonnegative and no less than the corresponding element of the identity matrix. This follows from the **Neumann series**:

$$(I - \gamma P_\pi)^{-1} = I + \gamma P_\pi + \gamma^2 P_\pi^2 + \cdots \geq I \geq 0$$

since $P_\pi$ has nonnegative entries.

**Property 3: Monotonicity.**

For any vector $r \geq 0$: $(I - \gamma P_\pi)^{-1} r \geq r \geq 0$.

More generally, if $r_1 \geq r_2$, then $(I - \gamma P_\pi)^{-1} r_1 \geq (I - \gamma P_\pi)^{-1} r_2$.

*Proof*: From Property 2, $[(I - \gamma P_\pi)^{-1} - I] r \geq 0$.

### 2.7.2 Iterative Solution

The closed-form solution involves matrix inversion, which is impractical for large state spaces. Instead, the Bellman equation can be solved iteratively:

$$\boxed{v_{k+1} = r_\pi + \gamma P_\pi v_k, \quad k = 0, 1, 2, \ldots}$$

Starting from an arbitrary initial guess $v_0 \in \mathbb{R}^n$, this generates a sequence $\{v_0, v_1, v_2, \ldots\}$ satisfying:

$$v_k \to v_\pi = (I - \gamma P_\pi)^{-1} r_\pi \quad \text{as } k \to \infty$$

#### Convergence Proof

Define the error $\delta_k = v_k - v_\pi$. Substituting $v_{k+1} = \delta_{k+1} + v_\pi$ and $v_k = \delta_k + v_\pi$ into $v_{k+1} = r_\pi + \gamma P_\pi v_k$:

$$\delta_{k+1} + v_\pi = r_\pi + \gamma P_\pi(\delta_k + v_\pi)$$

Since $v_\pi = r_\pi + \gamma P_\pi v_\pi$, this simplifies to:

$$\delta_{k+1} = \gamma P_\pi \delta_k$$

By induction:

$$\delta_{k+1} = \gamma^{k+1} P_\pi^{k+1} \delta_0$$

Since $P_\pi$ is a stochastic matrix, $0 \leq P_\pi^k \leq \mathbf{1}$ (every entry of $P_\pi^k$ is between 0 and 1), because $P_\pi^k \mathbf{1} = \mathbf{1}$. Since $\gamma < 1$, we have $\gamma^k \to 0$, and hence $\delta_{k+1} = \gamma^{k+1} P_\pi^{k+1} \delta_0 \to 0$ as $k \to \infty$.

### 2.7.3 Illustrative Examples (5x5 Grid World)

The examples use a 5x5 grid world with rewards: $r_{\text{boundary}} = r_{\text{forbidden}} = -1$, $r_{\text{target}} = +1$, and $\gamma = 0.9$.

#### Two "Good" Policies

Both policies yield the same state values despite differing at the top two states in the fourth column:

| Row\Col | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 3.5 | 3.9 | 4.3 | 4.8 | 5.3 |
| 2 | 3.1 | 3.5 | 4.8 | 5.3 | 5.9 |
| 3 | 2.8 | 2.5 | 10.0 | 5.9 | 6.6 |
| 4 | 2.5 | 10.0 | 10.0 | 10.0 | 7.3 |
| 5 | 2.3 | 9.0 | 10.0 | 9.0 | 8.1 |

**Key observation**: Different policies can have the same state values.

#### Two "Bad" Policies

**Bad policy 1**:

| Row\Col | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | -6.6 | -7.3 | -8.1 | -9.0 | -10.0 |
| 2 | -8.5 | -8.3 | -8.1 | -9.0 | -10.0 |
| 3 | -7.5 | -8.3 | -8.1 | -9.0 | -10.0 |
| 4 | -7.5 | -7.2 | -9.1 | -9.0 | -10.0 |
| 5 | -7.6 | -7.3 | -8.1 | -9.0 | -10.0 |

**Bad policy 2**:

| Row\Col | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | -10.0 | -10.0 |
| 2 | -9.0 | -10.0 | -0.4 | -0.5 | -10.0 |
| 3 | -10.0 | -0.5 | 0.5 | -0.5 | 0.0 |
| 4 | 0.0 | -1.0 | -0.5 | -0.5 | -10.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The state values of the "bad" policies are negative and much smaller than those of the "good" policies, confirming the intuition that these policies are indeed bad.

---

## 2.8 From State Value to Action Value

### Motivation

While state values tell us how good it is to *be* in a state, **action values** tell us how good it is to *take a specific action* at a state. Action values are crucial for finding optimal policies (as will become clear in Chapter 3).

### Definition: Action-Value Function

$$\boxed{q_\pi(s, a) \doteq \mathbb{E}[G_t | S_t = s, A_t = a]}$$

The action value is the expected return obtained after taking action $a$ in state $s$ and then following policy $\pi$. Note: $q_\pi(s,a)$ depends on the state-action pair $(s,a)$, not on the action alone. More rigorously, it could be called a "state-action value," but conventionally it is called an "action value."

### Relationship Between State Values and Action Values

**Direction 1: State values from action values**

From the properties of conditional expectation:

$$\boxed{v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \, q_\pi(s, a)}$$

A state value is the **weighted average** (expectation) of the action values for that state, weighted by the policy.

**Direction 2: Action values from state values**

Comparing the Bellman equation with the above:

$$\boxed{q_\pi(s, a) = \sum_{r \in \mathcal{R}} p(r|s,a) \, r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \, v_\pi(s')}$$

The action value consists of:
- **Immediate reward**: $\sum_{r} p(r|s,a) \, r$ (mean of immediate rewards for taking action $a$ at state $s$)
- **Future reward**: $\gamma \sum_{s'} p(s'|s,a) \, v_\pi(s')$ (discounted mean of future rewards from the next state)

These two equations are **two sides of the same coin**: the first shows how to obtain state values from action values; the second shows how to obtain action values from state values.

**Slide insight**: We can first calculate all state values and then calculate action values. Alternatively, we can directly calculate action values with or without models.

### 2.8.1 Illustrative Example (Action Values)

Consider the stochastic policy from Figure 2.8 (same as Example 2). At state $s_1$, the policy selects $a_2$ (right) or $a_3$ (down) each with probability 0.5.

**Action values for actions selected by the policy**:

$$q_\pi(s_1, a_2) = -1 + \gamma v_\pi(s_2)$$
$$q_\pi(s_1, a_3) = 0 + \gamma v_\pi(s_3)$$

**Action values for actions NOT selected by the policy**:

A common mistake is to set $q_\pi(s_1, a_1) = q_\pi(s_1, a_4) = q_\pi(s_1, a_5) = 0$. **This is wrong.**

Even though the policy does not select $a_1, a_4, a_5$, they still have well-defined action values:

- $a_1$ (up): Agent bounces back to $s_1$ with reward $-1$, then follows $\pi$ from $s_1$:
  $$q_\pi(s_1, a_1) = -1 + \gamma v_\pi(s_1)$$

- $a_4$ (left): Agent bounces back to $s_1$ with reward $-1$:
  $$q_\pi(s_1, a_4) = -1 + \gamma v_\pi(s_1)$$

- $a_5$ (stay): Agent stays at $s_1$ with reward $0$:
  $$q_\pi(s_1, a_5) = 0 + \gamma v_\pi(s_1)$$

**Why care about unselected actions?** Although a given policy does not select some actions, this does not mean those actions are bad. The given policy might be suboptimal and miss the best action. To find optimal policies, we must keep exploring **all** actions.

**Verification**: The state value can be recovered from action values:
$$v_\pi(s_1) = 0.5 \, q_\pi(s_1, a_2) + 0.5 \, q_\pi(s_1, a_3) = 0.5[0 + \gamma v_\pi(s_3)] + 0.5[-1 + \gamma v_\pi(s_2)]$$

### 2.8.2 Bellman Equation in Terms of Action Values

Substituting $v_\pi(s') = \sum_{a'} \pi(a'|s') q_\pi(s',a')$ into the action value expression yields the **Bellman equation for action values**:

$$q_\pi(s, a) = \sum_{r \in \mathcal{R}} p(r|s,a) \, r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \sum_{a' \in \mathcal{A}(s')} \pi(a'|s') \, q_\pi(s', a')$$

This is valid for every state-action pair. In matrix-vector form:

$$q_\pi = \tilde{r} + \gamma P \Pi \, q_\pi$$

where:
- $q_\pi$: action value vector indexed by state-action pairs; $[q_\pi]_{(s,a)} = q_\pi(s,a)$
- $\tilde{r}$: immediate reward vector indexed by state-action pairs; $[\tilde{r}]_{(s,a)} = \sum_{r} p(r|s,a) \, r$
- $P$: probability transition matrix; rows indexed by $(s,a)$, columns by $s'$; $[P]_{(s,a),s'} = p(s'|s,a)$
- $\Pi$: block diagonal matrix; each block is a $1 \times |\mathcal{A}|$ vector; $\Pi_{s',(s',a')} = \pi(a'|s')$, all other entries zero

**Key feature**: $\tilde{r}$ and $P$ are **independent of the policy** (determined by the system model only). The policy is embedded solely in $\Pi$. This Bellman equation for action values is also a contraction mapping with a unique solution that can be iteratively solved.

---

## 2.9 Summary

The chapter's key results in order of importance:

1. **State value**: $v_\pi(s) = \mathbb{E}[G_t | S_t = s]$ -- the expected return starting from state $s$ under policy $\pi$.

2. **Action value**: $q_\pi(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$ -- the expected return starting from state $s$, taking action $a$, then following $\pi$.

3. **Bellman equation (elementwise)**:
$$v_\pi(s) = \sum_a \pi(a|s) \left[ \sum_r p(r|s,a) \, r + \gamma \sum_{s'} p(s'|s,a) \, v_\pi(s') \right] = \sum_a \pi(a|s) \, q_\pi(s,a)$$

4. **Bellman equation (matrix-vector)**: $v_\pi = r_\pi + \gamma P_\pi v_\pi$

5. **Solving the Bellman equation**:
   - Closed-form: $v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$
   - Iterative: $v_{k+1} = r_\pi + \gamma P_\pi v_k$, converges for any $v_0$

6. **Policy evaluation**: The process of solving the Bellman equation to obtain state values of a given policy.

7. **Bootstrapping**: The fundamental idea that the value of one state relies on the values of other states.

The Bellman equation is not restricted to RL -- it widely exists in control theory and operations research. In different fields, it may have different expressions. This book studies it under discrete MDPs.

---

## 2.10 Q&A -- Important Clarifications

### Q: What is the relationship between state values and returns?
**A**: The value of a state is the **mean** of the returns that can be obtained if the agent starts from that state. When everything is deterministic, the state value equals the return.

### Q: Why do we care about state values?
**A**: State values evaluate policies. Optimal policies are defined based on state values (formalized in Chapter 3).

### Q: Why do we care about the Bellman equation?
**A**: The Bellman equation describes the relationships among the values of all states. It is the tool for analyzing state values.

### Q: Why is solving the Bellman equation called policy evaluation?
**A**: Solving the Bellman equation yields state values. Since state values evaluate a policy, solving the Bellman equation = evaluating the corresponding policy.

### Q: Why study the matrix-vector form?
**A**: The Bellman equation is a set of linear equations for all states. To solve for state values, we must put all equations together. The matrix-vector form is the concise expression of these equations.

### Q: What is the relationship between state values and action values?
**A**: On one hand, a state value is the weighted average of the action values for that state. On the other hand, an action value relies on the values of the next states the agent may transition to after taking the action.

### Q: Why care about actions a given policy cannot select?
**A**: Although a given policy does not select some actions, those actions may still be good. The given policy might be suboptimal. To find better policies, we must keep exploring all actions.

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| Discounted return | $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ | 2.3 |
| State-value function (state value) | $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]$ | 2.3 |
| Bootstrapping | values depending on each other | 2.2, 2.4 |
| Bellman equation (elementwise) | $v_\pi(s) = \sum_a \pi(a \mid s)[\sum_r p(r \mid s,a)r + \gamma \sum_{s'} p(s' \mid s,a) v_\pi(s')]$ | 2.4 |
| Mean immediate reward | $r_\pi(s) = \sum_a \pi(a \mid s) \sum_r p(r \mid s,a) r$ | 2.6 |
| Transition probability under policy | $p_\pi(s' \mid s) = \sum_a \pi(a \mid s) p(s' \mid s,a)$ | 2.6 |
| State transition matrix | $P_\pi \in \mathbb{R}^{n \times n}$, $[P_\pi]_{ij} = p_\pi(s_j \mid s_i)$ | 2.6 |
| Stochastic matrix | $P_\pi \geq 0$, $P_\pi \mathbf{1} = \mathbf{1}$ | 2.6 |
| Bellman equation (matrix-vector) | $v_\pi = r_\pi + \gamma P_\pi v_\pi$ | 2.6 |
| Policy evaluation | solving for state values of a given policy | 2.7 |
| Closed-form solution | $v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$ | 2.7.1 |
| Neumann series expansion | $(I - \gamma P_\pi)^{-1} = I + \gamma P_\pi + \gamma^2 P_\pi^2 + \cdots$ | 2.7.1 |
| Gershgorin circle theorem | used to prove $I - \gamma P_\pi$ invertible | 2.7.1 |
| Iterative solution | $v_{k+1} = r_\pi + \gamma P_\pi v_k$ | 2.7.2 |
| Action-value function (action value) | $q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$ | 2.8 |
| State value from action values | $v_\pi(s) = \sum_a \pi(a \mid s) q_\pi(s,a)$ | 2.8 |
| Action value from state values | $q_\pi(s,a) = \sum_r p(r \mid s,a)r + \gamma \sum_{s'} p(s' \mid s,a) v_\pi(s')$ | 2.8 |
| Bellman equation for action values | $q_\pi = \tilde{r} + \gamma P \Pi q_\pi$ | 2.8.2 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| State value $v_\pi(s)$ | Ch 3 (optimal state value, Bellman optimality equation), Ch 4 (policy/value iteration), Ch 5 (Monte Carlo estimation), Ch 7 (TD learning), Ch 8-10 (function approximation) |
| Bellman equation $v_\pi = r_\pi + \gamma P_\pi v_\pi$ | Ch 3 (Bellman optimality equation), Ch 4 (policy iteration uses policy evaluation as a subroutine) |
| Policy evaluation (solving Bellman equation) | Ch 4 (policy iteration alternates evaluation and improvement), Ch 5 (Monte Carlo policy evaluation), Ch 7 (TD-based policy evaluation) |
| Action value $q_\pi(s,a)$ | Ch 3 (optimal action values define optimal policies), Ch 4 (action values used for policy improvement), Ch 7 (Q-learning), Ch 8 (DQN), Ch 9-10 (policy gradient, actor-critic) |
| Iterative solution $v_{k+1} = r_\pi + \gamma P_\pi v_k$ | Ch 4 (basis for value iteration), Ch 6 (stochastic approximation generalizes this) |
| $(I - \gamma P_\pi)^{-1}$ properties (monotonicity, nonnegativity) | Ch 3 (proving properties of optimal values), Ch 4 (convergence proofs) |
| Bootstrapping idea | Ch 7 (TD methods bootstrap), Ch 8 (value function approximation) |
| Bellman equation for action values $q_\pi = \tilde{r} + \gamma P \Pi q_\pi$ | Ch 3 (Bellman optimality equation for $q^*$), Ch 7 (SARSA, Q-learning) |
| Action values of unselected actions | Ch 3 (greedy policy improvement requires comparing all action values), Ch 4 (policy improvement step) |
