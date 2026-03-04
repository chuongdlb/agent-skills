---
chapter: 4
title: Value Iteration and Policy Iteration
key_topics: [value iteration, policy iteration, truncated policy iteration, dynamic programming, policy update, value update, policy evaluation, policy improvement, greedy policy, contraction mapping theorem application, Bellman optimality equation solving, generalized policy iteration, model-based vs model-free]
depends_on: [1, 2, 3]
required_by: [5, 7, 8, 9, 10]
---

# Chapter 4: Value Iteration and Policy Iteration

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 4, pp. 57-76
> Supplemented by: Lecture slides L4 (37 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter presents the **first algorithms that can find optimal policies**. Three closely related algorithms are introduced:

1. **Value iteration** -- exactly the algorithm suggested by the contraction mapping theorem (Theorem 3.3) for solving the Bellman optimality equation.
2. **Policy iteration** -- whose idea of alternating between policy evaluation and policy improvement is widely used throughout reinforcement learning.
3. **Truncated policy iteration** -- a unified algorithm that includes value iteration and policy iteration as two extreme special cases.

All three are **dynamic programming algorithms** that require the system model (i.e., $p(r|s,a)$ and $p(s'|s,a)$ are known). They are important foundations for the model-free reinforcement learning algorithms in subsequent chapters. For example, Monte Carlo algorithms (Chapter 5) are obtained by extending policy iteration.

**Position in the book**: Chapter 4 is the last chapter in the "with model" portion of the fundamental tools. Starting from Chapter 5, the book transitions to model-free methods.

---

## 4.1 Value Iteration

### 4.1.0 Core Idea

Value iteration is exactly the algorithm suggested by the **contraction mapping theorem** (Theorem 3.3 from Chapter 3) for solving the Bellman optimality equation. The algorithm is:

$$v_{k+1} = \max_{\pi \in \Pi} (r_\pi + \gamma P_\pi v_k), \quad k = 0, 1, 2, \ldots$$

It is guaranteed by Theorem 3.3 that $v_k$ and $\pi_k$ converge to the optimal state value $v^*$ and an optimal policy $\pi^*$ as $k \to \infty$, respectively. The initial guess $v_0$ can be arbitrary.

### Two Steps per Iteration (Matrix-Vector Form)

Each iteration decomposes into two steps:

**Step 1: Policy update.** Find a policy that solves:
$$\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$$
where $v_k$ is obtained in the previous iteration.

**Step 2: Value update.** Calculate a new value:
$$v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k \tag{4.1}$$
where $v_{k+1}$ will be used in the next iteration.

**Key insight from slides**: The matrix-vector form is useful for theoretical analysis, while the elementwise form is necessary for implementation.

### 4.1.1 Elementwise Form and Implementation

Consider time step $k$ and a state $s$.

**Step 1: Policy update (elementwise form).**

The elementwise form of $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$ is:

$$\pi_{k+1}(s) = \arg\max_\pi \sum_a \pi(a|s) \underbrace{\left( \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right)}_{q_k(s,a)}, \quad s \in \mathcal{S}$$

From Section 3.3.1, the optimal policy solving this optimization problem is:

$$\pi_{k+1}(a|s) = \begin{cases} 1, & a = a^*_k(s), \\ 0, & a \neq a^*_k(s), \end{cases} \tag{4.2}$$

where $a^*_k(s) = \arg\max_a q_k(s,a)$. If $\arg\max_a q_k(s,a)$ has multiple solutions, any of them can be selected without affecting convergence. Since $\pi_{k+1}$ selects the action with the greatest $q_k(s,a)$, such a policy is called **greedy**.

**Step 2: Value update (elementwise form).**

The elementwise form of $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$ is:

$$v_{k+1}(s) = \sum_a \pi_{k+1}(a|s) \underbrace{\left( \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right)}_{q_k(s,a)}, \quad s \in \mathcal{S}$$

Since $\pi_{k+1}$ is greedy, this simplifies to:

$$v_{k+1}(s) = \max_a q_k(s,a)$$

**Procedure summary (from slides):**
$$v_k(s) \;\to\; q_k(s,a) \;\to\; \text{greedy policy } \pi_{k+1}(s) \;\to\; \text{new value } v_{k+1}(s) = \max_a q_k(s,a)$$

### Algorithm 4.1: Value Iteration Algorithm (Pseudocode)

```
Initialization: The probability models p(r|s,a) and p(s'|s,a) for all (s,a) are known.
               Initial guess v_0.
Goal: Search for the optimal state value and an optimal policy for solving the
      Bellman optimality equation.

While v_k has not converged (i.e., ||v_k - v_{k-1}|| > threshold), for the kth iteration, do:
    For every state s in S, do:
        For every action a in A(s), do:
            q-value:  q_k(s,a) = sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a)*v_k(s')
        Maximum action value:  a*_k(s) = argmax_a q_k(s,a)
        Policy update:  pi_{k+1}(a|s) = 1 if a = a*_k, and pi_{k+1}(a|s) = 0 otherwise
        Value update:   v_{k+1}(s) = max_a q_k(s,a)
```

### Important Note: Is $v_k$ a State Value?

**No.** Although $v_k$ eventually converges to the optimal state value $v^*$, it is not ensured to satisfy the Bellman equation of any policy. Specifically, $v_k \neq r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$ and $v_k \neq r_{\pi_k} + \gamma P_{\pi_k} v_k$ in general. It is merely an intermediate value generated by the algorithm. Since $v_k$ is not a state value, $q_k$ is not an action value.

### 4.1.2 Illustrative Example

**Setup**: A 2x2 grid world with states $s_1, s_2, s_3, s_4$. State $s_3$ is a forbidden area. State $s_4$ is the target. Reward settings: $r_{\text{boundary}} = r_{\text{forbidden}} = -1$, $r_{\text{target}} = 1$. Discount rate $\gamma = 0.9$.

**Q-table expressions** (Table 4.1):

| q-table | $a_1$ (up) | $a_2$ (right) | $a_3$ (down) | $a_4$ (left) | $a_5$ (stay) |
|---|---|---|---|---|---|
| $s_1$ | $-1 + \gamma v(s_1)$ | $-1 + \gamma v(s_2)$ | $0 + \gamma v(s_3)$ | $-1 + \gamma v(s_1)$ | $0 + \gamma v(s_1)$ |
| $s_2$ | $-1 + \gamma v(s_2)$ | $-1 + \gamma v(s_2)$ | $1 + \gamma v(s_4)$ | $0 + \gamma v(s_1)$ | $-1 + \gamma v(s_2)$ |
| $s_3$ | $0 + \gamma v(s_1)$ | $1 + \gamma v(s_4)$ | $-1 + \gamma v(s_3)$ | $-1 + \gamma v(s_3)$ | $0 + \gamma v(s_3)$ |
| $s_4$ | $-1 + \gamma v(s_2)$ | $-1 + \gamma v(s_4)$ | $-1 + \gamma v(s_4)$ | $0 + \gamma v(s_3)$ | $1 + \gamma v(s_4)$ |

#### Iteration k = 0

Initial values: $v_0(s_1) = v_0(s_2) = v_0(s_3) = v_0(s_4) = 0$.

**Q-values** (Table 4.2, substituting $v_0 = 0$):

| q-table | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| $s_1$ | $-1$ | $-1$ | $0$ | $-1$ | $0$ |
| $s_2$ | $-1$ | $-1$ | $1$ | $0$ | $-1$ |
| $s_3$ | $0$ | $1$ | $-1$ | $-1$ | $0$ |
| $s_4$ | $-1$ | $-1$ | $-1$ | $0$ | $1$ |

**Policy update** (select actions with greatest q-values):
$$\pi_1(a_5|s_1) = 1, \quad \pi_1(a_3|s_2) = 1, \quad \pi_1(a_2|s_3) = 1, \quad \pi_1(a_5|s_4) = 1$$

This policy is **not optimal** because it selects "stay" at $s_1$. (Note: $q_0(s_1, a_5) = q_0(s_1, a_3) = 0$, so either action could be selected.)

**Value update**:
$$v_1(s_1) = 0, \quad v_1(s_2) = 1, \quad v_1(s_3) = 1, \quad v_1(s_4) = 1$$

#### Iteration k = 1

Substituting $v_1$ into Table 4.1:

| q-table | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| $s_1$ | $-1 + 0.9 \cdot 0$ | $-1 + 0.9 \cdot 1$ | $0 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 0$ | $0 + 0.9 \cdot 0$ |
| $s_2$ | $-1 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 1$ | $1 + 0.9 \cdot 1$ | $0 + 0.9 \cdot 0$ | $-1 + 0.9 \cdot 1$ |
| $s_3$ | $0 + 0.9 \cdot 0$ | $1 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 1$ | $0 + 0.9 \cdot 1$ |
| $s_4$ | $-1 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 1$ | $-1 + 0.9 \cdot 1$ | $0 + 0.9 \cdot 1$ | $1 + 0.9 \cdot 1$ |

Evaluated q-values:

| q-table | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| $s_1$ | $-1$ | $-0.1$ | $0.9$ | $-1$ | $0$ |
| $s_2$ | $-0.1$ | $-0.1$ | $1.9$ | $0$ | $-0.1$ |
| $s_3$ | $0$ | $1.9$ | $-0.1$ | $-0.1$ | $0.9$ |
| $s_4$ | $-0.1$ | $-0.1$ | $-0.1$ | $0.9$ | $1.9$ |

**Policy update**:
$$\pi_2(a_3|s_1) = 1, \quad \pi_2(a_3|s_2) = 1, \quad \pi_2(a_2|s_3) = 1, \quad \pi_2(a_5|s_4) = 1$$

**This policy is already optimal!**

**Value update**:
$$v_2(s_1) = 0.9, \quad v_2(s_2) = 1.9, \quad v_2(s_3) = 1.9, \quad v_2(s_4) = 1.9$$

#### k = 2, 3, 4, ...

Continue iterating until $\|v_{k+1} - v_k\|$ is smaller than a pre-specified threshold. In this simple example, only two iterations were needed to find an optimal policy.

---

## 4.2 Policy Iteration

### 4.2.0 Overview

Policy iteration is another important algorithm for finding optimal policies. Unlike value iteration, policy iteration is **not** for directly solving the Bellman optimality equation. However, it has an intimate relationship with value iteration and its idea is widely utilized in reinforcement learning algorithms.

### 4.2.1 Algorithm Analysis

Policy iteration is an iterative algorithm. Each iteration has two steps.

**Step 1: Policy evaluation (PE).** Evaluate the current policy $\pi_k$ by calculating its state value. Solve the Bellman equation:

$$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k} \tag{4.3}$$

where $\pi_k$ is the policy from the last iteration and $v_{\pi_k}$ is the state value to be calculated.

**Step 2: Policy improvement (PI).** Improve the policy using the calculated state value:

$$\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$$

The algorithm generates a sequence:

$$\pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \xrightarrow{PE} v_{\pi_2} \xrightarrow{PI} \cdots$$

### Three Key Questions

#### Q1: How to calculate $v_{\pi_k}$ in the policy evaluation step?

Two methods (from Chapter 2):

**Method 1: Closed-form solution:**
$$v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$$

This is useful for theoretical analysis but inefficient to implement since it requires computing a matrix inverse.

**Method 2: Iterative solution:**
$$v^{(j+1)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j)}_{\pi_k}, \quad j = 0, 1, 2, \ldots \tag{4.4}$$

where $v^{(j)}_{\pi_k}$ denotes the $j$th estimate of $v_{\pi_k}$. Starting from any initial guess $v^{(0)}_{\pi_k}$, it is ensured that $v^{(j)}_{\pi_k} \to v_{\pi_k}$ as $j \to \infty$.

**Key insight**: Policy iteration is an iterative algorithm with **another iterative algorithm** (4.4) embedded in the policy evaluation step. In theory, this embedded algorithm requires an infinite number of steps ($j \to \infty$) to converge to the true state value $v_{\pi_k}$. In practice, the iterative process terminates when $\|v^{(j+1)}_{\pi_k} - v^{(j)}_{\pi_k}\|$ is less than a threshold or $j$ exceeds a maximum value.

**Important**: Even if we do not run infinitely many iterations and only obtain an imprecise value of $v_{\pi_k}$, this does **not** cause problems. The reason is explained by truncated policy iteration (Section 4.3).

#### Q2: Why is the new policy $\pi_{k+1}$ better than $\pi_k$?

### Lemma 4.1 (Policy Improvement)

**Statement**: If $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$, then $v_{\pi_{k+1}} \geq v_{\pi_k}$.

Here, $v_{\pi_{k+1}} \geq v_{\pi_k}$ means $v_{\pi_{k+1}}(s) \geq v_{\pi_k}(s)$ for all $s$.

**Proof (Box 4.1):**

Since $v_{\pi_{k+1}}$ and $v_{\pi_k}$ are state values, they satisfy the Bellman equations:
$$v_{\pi_{k+1}} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}$$
$$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$$

Since $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$, we know that:
$$r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k} \geq r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$$

It then follows that:
$$v_{\pi_k} - v_{\pi_{k+1}} = (r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}) - (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}})$$
$$\leq (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}})$$
$$= \gamma P_{\pi_{k+1}} (v_{\pi_k} - v_{\pi_{k+1}})$$

Therefore:
$$v_{\pi_k} - v_{\pi_{k+1}} \leq \gamma^2 P^2_{\pi_{k+1}} (v_{\pi_k} - v_{\pi_{k+1}}) \leq \cdots \leq \gamma^n P^n_{\pi_{k+1}} (v_{\pi_k} - v_{\pi_{k+1}})$$
$$\leq \lim_{n \to \infty} \gamma^n P^n_{\pi_{k+1}} (v_{\pi_k} - v_{\pi_{k+1}}) = 0$$

The limit follows from $\gamma^n \to 0$ as $n \to \infty$ and $P^n_{\pi_{k+1}}$ being a nonnegative stochastic matrix (rows sum to 1) for any $n$.

Therefore $v_{\pi_k} - v_{\pi_{k+1}} \leq 0$, i.e., $v_{\pi_{k+1}} \geq v_{\pi_k}$. $\blacksquare$

#### Q3: Why does the algorithm converge to an optimal policy?

Since every iteration improves the policy (Lemma 4.1):

$$v_{\pi_0} \leq v_{\pi_1} \leq v_{\pi_2} \leq \cdots \leq v_{\pi_k} \leq \cdots \leq v^*$$

Since $v_{\pi_k}$ is nondecreasing and always bounded above by $v^*$, the **monotone convergence theorem** (Appendix C) guarantees that $v_{\pi_k}$ converges to some constant value $v_\infty$ as $k \to \infty$.

### Theorem 4.1 (Convergence of Policy Iteration)

**Statement**: The state value sequence $\{v_{\pi_k}\}_{k=0}^\infty$ generated by the policy iteration algorithm converges to the optimal state value $v^*$. As a result, the policy sequence $\{\pi_k\}_{k=0}^\infty$ converges to an optimal policy.

**Proof (Box 4.2):**

The idea is to show that policy iteration converges **faster** than value iteration.

Introduce another sequence $\{v_k\}_{k=0}^\infty$ generated by:
$$v_{k+1} = f(v_k) = \max_\pi (r_\pi + \gamma P_\pi v_k)$$

This is exactly the value iteration algorithm. We already know $v_k \to v^*$ for any initial $v_0$.

For $k=0$, we can always find $v_0$ such that $v_{\pi_0} \geq v_0$ for any $\pi_0$.

**Induction**: Show $v_k \leq v_{\pi_k} \leq v^*$ for all $k$.

For $k \geq 0$, suppose $v_{\pi_k} \geq v_k$. For $k+1$:

$$v_{\pi_{k+1}} - v_{k+1} = (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_{k+1}}) - \max_\pi (r_\pi + \gamma P_\pi v_k)$$
$$\geq (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - \max_\pi (r_\pi + \gamma P_\pi v_k)$$

(because $v_{\pi_{k+1}} \geq v_{\pi_k}$ by Lemma 4.1 and $P_{\pi_{k+1}} \geq 0$)

Let $\pi'_k = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$. Then:

$$= (r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_{\pi_k}) - (r_{\pi'_k} + \gamma P_{\pi'_k} v_k)$$
$$\geq (r_{\pi'_k} + \gamma P_{\pi'_k} v_{\pi_k}) - (r_{\pi'_k} + \gamma P_{\pi'_k} v_k)$$

(because $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$)

$$= \gamma P_{\pi'_k} (v_{\pi_k} - v_k) \geq 0$$

Since $v_{\pi_k} - v_k \geq 0$ and $P_{\pi'_k}$ is nonnegative, we have $v_{\pi_{k+1}} - v_{k+1} \geq 0$.

By induction, $v_k \leq v_{\pi_k} \leq v^*$ for all $k \geq 0$. Since $v_k \to v^*$, it follows that $v_{\pi_k} \to v^*$. $\blacksquare$

**Key insight from the proof**: If both algorithms start from the same initial guess, policy iteration converges **faster** than value iteration due to the additional iterations embedded in the policy evaluation step.

### 4.2.2 Elementwise Form and Implementation

**Policy evaluation (elementwise form):**

$$v^{(j+1)}_{\pi_k}(s) = \sum_a \pi_k(a|s) \left( \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v^{(j)}_{\pi_k}(s') \right), \quad s \in \mathcal{S}$$

Stop when $j$ is sufficiently large or $\|v^{(j+1)}_{\pi_k} - v^{(j)}_{\pi_k}\|$ is sufficiently small.

**Policy improvement (elementwise form):**

$$\pi_{k+1}(s) = \arg\max_\pi \sum_a \pi(a|s) \underbrace{\left( \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s') \right)}_{q_{\pi_k}(s,a)}, \quad s \in \mathcal{S}$$

where $q_{\pi_k}(s,a)$ is the **action value under policy $\pi_k$**. Let $a^*_k(s) = \arg\max_a q_{\pi_k}(s,a)$. Then the greedy policy is:

$$\pi_{k+1}(a|s) = \begin{cases} 1, & a = a^*_k(s), \\ 0, & a \neq a^*_k(s). \end{cases}$$

### Algorithm 4.2: Policy Iteration Algorithm (Pseudocode)

```
Initialization: The system model p(r|s,a) and p(s'|s,a) for all (s,a) are known.
               Initial guess pi_0.
Goal: Search for the optimal state value and an optimal policy.

While v_{pi_k} has not converged, for the kth iteration, do:

    Policy evaluation:
        Initialization: an arbitrary initial guess v^(0)_{pi_k}
        While v^(j)_{pi_k} has not converged, for the jth iteration, do:
            For every state s in S, do:
                v^(j+1)_{pi_k}(s) = sum_a pi_k(a|s) [sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a)*v^(j)_{pi_k}(s')]

    Policy improvement:
        For every state s in S, do:
            For every action a in A, do:
                q_{pi_k}(s,a) = sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a)*v_{pi_k}(s')
            a*_k(s) = argmax_a q_{pi_k}(s,a)
            pi_{k+1}(a|s) = 1 if a = a*_k, and pi_{k+1}(a|s) = 0 otherwise
```

### 4.2.3 Illustrative Examples

#### Simple Example (Two-State Grid)

**Setup**: Two states $s_1, s_2$. Three actions: $\mathcal{A} = \{a_\ell, a_0, a_r\}$ (go left, stay, go right). State $s_2$ is the target. Reward settings: $r_{\text{boundary}} = -1$, $r_{\text{target}} = 1$. Discount rate $\gamma = 0.9$.

**Q-table expression** (Table 4.4):

| $q_{\pi_k}(s,a)$ | $a_\ell$ | $a_0$ | $a_r$ |
|---|---|---|---|
| $s_1$ | $-1 + \gamma v_{\pi_k}(s_1)$ | $0 + \gamma v_{\pi_k}(s_1)$ | $1 + \gamma v_{\pi_k}(s_2)$ |
| $s_2$ | $0 + \gamma v_{\pi_k}(s_1)$ | $1 + \gamma v_{\pi_k}(s_2)$ | $-1 + \gamma v_{\pi_k}(s_2)$ |

##### Iteration k = 0

**Step 1: Policy evaluation.** Start with initial policy $\pi_0$: $\pi_0(a_\ell|s_1) = 1$ (go left at $s_1$), $\pi_0(a_\ell|s_2) = 1$ (go left at $s_2$). This policy is bad -- it moves away from the target.

The Bellman equation becomes:
$$v_{\pi_0}(s_1) = -1 + \gamma v_{\pi_0}(s_1)$$
$$v_{\pi_0}(s_2) = 0 + \gamma v_{\pi_0}(s_1)$$

**Closed-form solution**: Solving the first equation: $v_{\pi_0}(s_1)(1 - \gamma) = -1$, so:
$$v_{\pi_0}(s_1) = \frac{-1}{1-0.9} = -10, \quad v_{\pi_0}(s_2) = 0.9 \times (-10) = -9$$

**Iterative solution** (starting from $v^{(0)}_{\pi_0}(s_1) = v^{(0)}_{\pi_0}(s_2) = 0$):

| $j$ | $v^{(j)}_{\pi_0}(s_1)$ | $v^{(j)}_{\pi_0}(s_2)$ |
|---|---|---|
| 0 | 0 | 0 |
| 1 | $-1 + 0.9(0) = -1$ | $0 + 0.9(0) = 0$ |
| 2 | $-1 + 0.9(-1) = -1.9$ | $0 + 0.9(-1) = -0.9$ |
| 3 | $-1 + 0.9(-1.9) = -2.71$ | $0 + 0.9(-1.9) = -1.71$ |
| $\vdots$ | $\to -10$ | $\to -9$ |

**Step 2: Policy improvement.** Substituting $v_{\pi_0}(s_1) = -10$, $v_{\pi_0}(s_2) = -9$ into the q-table:

| $q_{\pi_0}(s,a)$ | $a_\ell$ | $a_0$ | $a_r$ |
|---|---|---|---|
| $s_1$ | $-10$ | $-9$ | $-7.1$ |
| $s_2$ | $-9$ | $-7.1$ | $-9.1$ |

Selecting the greatest q-value for each state:
$$\pi_1(a_r|s_1) = 1, \quad \pi_1(a_0|s_2) = 1$$

**This policy is optimal after just one iteration!** The agent moves right at $s_1$ to reach the target, and stays at $s_2$ (the target).

#### Complicated Example (5x5 Grid)

**Setup**: 5x5 grid with forbidden areas and a target area. Reward settings: $r_{\text{boundary}} = -1$, $r_{\text{forbidden}} = -10$, $r_{\text{target}} = 1$. Discount rate $\gamma = 0.9$.

The policy iteration algorithm converges to the optimal policy when starting from a random initial policy. **Two interesting phenomena** are observed:

1. **Spatial pattern of policy evolution**: States closer to the target find optimal policies **earlier** than those farther away. Only after close states find trajectories to the target can farther states find trajectories passing through those close states.

2. **Spatial distribution of state values**: States closer to the target have **greater** state values. Agents starting from farther states must travel many steps to obtain positive reward, and such rewards are severely discounted.

---

## 4.3 Truncated Policy Iteration

### 4.3.1 Comparing Value Iteration and Policy Iteration

The two algorithms are compared side by side.

**Policy iteration** (starts from $\pi_0$):
- Step 1 (PE): Given $\pi_k$, solve $v_{\pi_k}$ from $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$
- Step 2 (PI): Given $v_{\pi_k}$, solve $\pi_{k+1}$ from $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$

**Value iteration** (starts from $v_0$):
- Step 1 (PU): Given $v_k$, solve $\pi_{k+1}$ from $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$
- Step 2 (VU): Given $\pi_{k+1}$, solve $v_{k+1}$ from $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$

**Flow comparison:**

$$\text{Policy iteration: } \pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \xrightarrow{PE} v_{\pi_2} \xrightarrow{PI} \cdots$$

$$\text{Value iteration: } v_0 \xrightarrow{PU} \pi'_1 \xrightarrow{VU} v_1 \xrightarrow{PU} \pi'_2 \xrightarrow{VU} v_2 \xrightarrow{PU} \cdots$$

### Detailed Step-by-Step Comparison (Table 4.6)

Let both algorithms start from the same initial condition: $v_0 = v_{\pi_0}$.

| Step | Policy Iteration | Value Iteration | Comment |
|---|---|---|---|
| 1) Policy | $\pi_0$ | N/A | |
| 2) Value | $v_{\pi_0} = r_{\pi_0} + \gamma P_{\pi_0} v_{\pi_0}$ | $v_0 \doteq v_{\pi_0}$ | |
| 3) Policy | $\pi_1 = \arg\max_\pi(r_\pi + \gamma P_\pi v_{\pi_0})$ | $\pi_1 = \arg\max_\pi(r_\pi + \gamma P_\pi v_0)$ | **Same policies** |
| 4) Value | $v_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}$ | $v_1 = r_{\pi_1} + \gamma P_{\pi_1} v_0$ | $v_{\pi_1} \geq v_1$ since $v_{\pi_1} \geq v_{\pi_0}$ |
| 5) Policy | $\pi_2 = \arg\max_\pi(r_\pi + \gamma P_\pi v_{\pi_1})$ | $\pi'_2 = \arg\max_\pi(r_\pi + \gamma P_\pi v_1)$ | May differ |

**Key observation**: The first three steps produce the same results. They diverge at Step 4:
- **Policy iteration** solves $v_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}$, requiring an **infinite number** of iterations
- **Value iteration** computes $v_1 = r_{\pi_1} + \gamma P_{\pi_1} v_0$, which is a **single-step** calculation

### The Unifying View

Writing out the iterative process for solving $v_{\pi_1}$ with initial guess $v^{(0)}_{\pi_1} = v_0$:

$$v^{(0)}_{\pi_1} = v_0$$
$$\text{value iteration} \leftarrow v_1 \leftarrow v^{(1)}_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v^{(0)}_{\pi_1}$$
$$v^{(2)}_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v^{(1)}_{\pi_1}$$
$$\vdots$$
$$\text{truncated policy iteration} \leftarrow \bar{v}_1 \leftarrow v^{(j)}_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v^{(j-1)}_{\pi_1}$$
$$\vdots$$
$$\text{policy iteration} \leftarrow v_{\pi_1} \leftarrow v^{(\infty)}_{\pi_1} = r_{\pi_1} + \gamma P_{\pi_1} v^{(\infty)}_{\pi_1}$$

**Three cases:**
- **1 iteration** ($j_{\text{truncate}} = 1$): This is **value iteration**
- **Finite $j$ iterations** ($j_{\text{truncate}} = j$): This is **truncated policy iteration**
- **Infinite iterations** ($j_{\text{truncate}} = \infty$): This is **policy iteration**

Value iteration and policy iteration are **two extreme cases** of truncated policy iteration.

**Important caveat**: This comparison is based on the condition that $v^{(0)}_{\pi_1} = v_0 = v_{\pi_0}$. The two algorithms cannot be directly compared without this condition.

### 4.3.2 Truncated Policy Iteration Algorithm

Truncated policy iteration is the same as policy iteration except that it runs only a **finite** number of iterations ($j_{\text{truncate}}$) in the policy evaluation step.

**Note**: $v_k$ and $v^{(j)}_k$ in the algorithm are **not** state values. They are approximations of the true state values because only a finite number of iterations are executed.

### Algorithm 4.3: Truncated Policy Iteration Algorithm (Pseudocode)

```
Initialization: The probability models p(r|s,a) and p(s'|s,a) for all (s,a) are known.
               Initial guess pi_0.
Goal: Search for the optimal state value and an optimal policy.

While v_k has not converged, for the kth iteration, do:

    Policy evaluation:
        Initialization: select initial guess v^(0)_k = v_{k-1}.
                       Set maximum number of iterations to j_truncate.
        While j < j_truncate, do:
            For every state s in S, do:
                v^(j+1)_k(s) = sum_a pi_k(a|s) [sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a)*v^(j)_k(s')]
        Set v_k = v^(j_truncate)_k

    Policy improvement:
        For every state s in S, do:
            For every action a in A(s), do:
                q_k(s,a) = sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a)*v_k(s')
            a*_k(s) = argmax_a q_k(s,a)
            pi_{k+1}(a|s) = 1 if a = a*_k, and pi_{k+1}(a|s) = 0 otherwise
```

**Key design choice**: The initial guess for the policy evaluation step is $v^{(0)}_k = v_{k-1}$ (the value from the previous outer iteration), not an arbitrary value.

### Convergence Analysis

**Will truncation undermine convergence?** No. Intuitively, truncated policy iteration lies between value iteration and policy iteration:
- It converges **faster** than value iteration because it computes more than one iteration during policy evaluation.
- It converges **slower** than policy iteration because it only computes a finite number of iterations.

This intuition is formalized by the following proposition.

### Proposition 4.1 (Value Improvement)

**Statement**: Consider the iterative algorithm in the policy evaluation step:

$$v^{(j+1)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j)}_{\pi_k}, \quad j = 0, 1, 2, \ldots$$

If the initial guess is selected as $v^{(0)}_{\pi_k} = v_{\pi_{k-1}}$, it holds that:

$$v^{(j+1)}_{\pi_k} \geq v^{(j)}_{\pi_k} \quad \text{for every } j = 0, 1, 2, \ldots$$

**Proof (Box 4.3):**

First, since $v^{(j)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j-1)}_{\pi_k}$ and $v^{(j+1)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j)}_{\pi_k}$, we have:

$$v^{(j+1)}_{\pi_k} - v^{(j)}_{\pi_k} = \gamma P_{\pi_k} (v^{(j)}_{\pi_k} - v^{(j-1)}_{\pi_k}) = \cdots = \gamma^j P^j_{\pi_k} (v^{(1)}_{\pi_k} - v^{(0)}_{\pi_k}) \tag{4.5}$$

Second, since $v^{(0)}_{\pi_k} = v_{\pi_{k-1}}$:

$$v^{(1)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(0)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_{k-1}} \geq r_{\pi_{k-1}} + \gamma P_{\pi_{k-1}} v_{\pi_{k-1}} = v_{\pi_{k-1}} = v^{(0)}_{\pi_k}$$

The inequality holds because $\pi_k = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_{k-1}})$.

Substituting $v^{(1)}_{\pi_k} \geq v^{(0)}_{\pi_k}$ into (4.5) yields $v^{(j+1)}_{\pi_k} \geq v^{(j)}_{\pi_k}$. $\blacksquare$

**Practical note**: Proposition 4.1 requires the assumption $v^{(0)}_{\pi_k} = v_{\pi_{k-1}}$, but $v_{\pi_{k-1}}$ is unavailable in practice -- only $v_{k-1}$ (the truncated approximation) is available. Nevertheless, the proposition provides insight into why truncated policy iteration converges.

### Advantages of Truncated Policy Iteration

- **Compared to policy iteration**: Only requires a finite number of iterations in PE, so it is more **computationally efficient**.
- **Compared to value iteration**: Running a few more iterations in PE speeds up the overall convergence rate.
- **Practical guideline**: Run a few iterations but not too many. A few iterations speed up overall convergence, but too many iterations do not significantly speed up convergence further.

---

## 4.4 Summary

All three algorithms find optimal policies. Each iteration has two steps: one updates the value and the other updates the policy.

| Property | Value Iteration | Policy Iteration | Truncated Policy Iteration |
|---|---|---|---|
| **Starts from** | Arbitrary $v_0$ | Arbitrary $\pi_0$ | Arbitrary $\pi_0$ |
| **Value step** | One-step update: $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$ | Solve Bellman eq.: $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$ (infinite iterations) | Finite $j_{\text{truncate}}$ iterations in PE |
| **Policy step** | $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$ | $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$ | $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$ |
| **Is $v_k$ a state value?** | No | Yes | No |
| **Is $q_k$ an action value?** | No | Yes | No |
| **Convergence guarantee** | Yes (contraction mapping theorem) | Yes (Theorem 4.1) | Yes |
| **$j_{\text{truncate}}$** | 1 | $\infty$ | Finite $j$ |

The idea of interaction between value and policy updates widely exists in RL algorithms. This idea is called **generalized policy iteration** (GPI).

**Model requirement**: All three algorithms require the system model. Starting in Chapter 5, model-free RL algorithms are introduced that extend the algorithms from this chapter.

---

## 4.5 Q&A -- Important Clarifications

### Q: Is the value iteration algorithm guaranteed to find optimal policies?
**A**: Yes. Value iteration is exactly the algorithm suggested by the contraction mapping theorem for solving the Bellman optimality equation (Chapter 3). Convergence is guaranteed by the contraction mapping theorem.

### Q: Are the intermediate values generated by value iteration state values?
**A**: No. These values are not guaranteed to satisfy the Bellman equation of any policy.

### Q: What steps are included in the policy iteration algorithm?
**A**: Each iteration contains two steps: *policy evaluation* (solve the Bellman equation to obtain the state value of the current policy) and *policy improvement* (update the policy so the new policy has greater state values).

### Q: Is another iterative algorithm embedded in policy iteration?
**A**: Yes. The policy evaluation step requires an iterative algorithm to solve the Bellman equation of the current policy.

### Q: Are the intermediate values generated by policy iteration state values?
**A**: Yes. These values are solutions of the Bellman equation of the current policy.

### Q: Is policy iteration guaranteed to find optimal policies?
**A**: Yes. Theorem 4.1 provides a rigorous proof of convergence.

### Q: What is the relationship between truncated policy iteration and policy iteration?
**A**: Truncated policy iteration is obtained from policy iteration by executing only a finite number of iterations during the policy evaluation step.

### Q: What is the relationship between truncated policy iteration and value iteration?
**A**: Value iteration is an extreme case of truncated policy iteration where a single iteration ($j_{\text{truncate}} = 1$) is run during the policy evaluation step.

### Q: Are the intermediate values generated by truncated policy iteration state values?
**A**: No. Only running an infinite number of iterations in the PE step yields true state values. A finite number yields approximations.

### Q: How many iterations should we run in the PE step of truncated policy iteration?
**A**: A few iterations but not too many. A few iterations speed up overall convergence, but too many iterations do not significantly speed up the convergence rate further.

### Q: What is generalized policy iteration?
**A**: Generalized policy iteration (GPI) is not a specific algorithm but the general idea of **interaction between value and policy updates**. This idea is rooted in the policy iteration algorithm. Most RL algorithms in this book fall into the scope of GPI.

### Q: What are model-based and model-free reinforcement learning?
**A**: The algorithms in this chapter are usually called **dynamic programming** algorithms (not RL algorithms) because they require the system model. RL algorithms are classified into:
- **Model-based**: Uses data to *estimate* the system model and uses this model during learning. ("Model-based" does **not** mean the system model is given.)
- **Model-free**: Does not involve model estimation during learning.

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| Value iteration algorithm | $v_{k+1} = \max_\pi (r_\pi + \gamma P_\pi v_k)$ | 4.1 |
| Policy update (in VI) | $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_k)$ | 4.1 |
| Value update (in VI) | $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$ | 4.1 |
| Greedy policy | $\pi_{k+1}(a|s) = 1$ if $a = a^*_k(s)$, 0 otherwise | 4.1.1 |
| Action value (intermediate) | $q_k(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s')$ | 4.1.1 |
| Policy iteration algorithm | PE + PI alternation | 4.2 |
| Policy evaluation (PE) | $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$ | 4.2.1 |
| Policy improvement (PI) | $\pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k})$ | 4.2.1 |
| Policy improvement lemma | $v_{\pi_{k+1}} \geq v_{\pi_k}$ (Lemma 4.1) | 4.2.1 |
| Convergence of policy iteration | $v_{\pi_k} \to v^*$ (Theorem 4.1) | 4.2.1 |
| Closed-form PE solution | $v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$ | 4.2.1 |
| Iterative PE solution | $v^{(j+1)}_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v^{(j)}_{\pi_k}$ | 4.2.1 |
| Truncated policy iteration | PE with finite $j_{\text{truncate}}$ iterations | 4.3 |
| Value improvement property | $v^{(j+1)}_{\pi_k} \geq v^{(j)}_{\pi_k}$ (Proposition 4.1) | 4.3.2 |
| Generalized policy iteration (GPI) | Interaction between value and policy updates | 4.4 |
| Dynamic programming algorithms | Model-based algorithms of Ch. 4 | 4.4 |
| Model-based RL | Estimates model from data, uses model in learning | 4.5 |
| Model-free RL | Does not estimate/use model during learning | 4.5 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| Value iteration (contraction mapping application) | Ch 3 (Theorem 3.3 provides convergence guarantee) |
| Policy evaluation (solving Bellman equation) | Ch 2 (iterative solution method from Section 2.7) |
| Policy iteration (PE + PI framework) | Ch 5 (Monte Carlo methods extend policy iteration to model-free) |
| Greedy policy improvement | Ch 7 (TD methods), Ch 8 (value function approximation), Ch 9 (policy gradient) |
| Generalized policy iteration (GPI) idea | Ch 5, 7, 8, 9, 10 (most RL algorithms fall under GPI) |
| Model requirement | Ch 5 onward (transition to model-free algorithms) |
| Tabular representation of policies/values | Ch 8 (replaced by function approximation) |
| Dynamic programming as foundation | Ch 5 (Monte Carlo), Ch 7 (TD methods build on DP ideas) |
| State value vs. intermediate value distinction | Ch 5, 7 (understanding when estimates are true values vs. approximations) |
