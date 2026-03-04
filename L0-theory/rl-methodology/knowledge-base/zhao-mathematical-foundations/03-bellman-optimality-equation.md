---
chapter: 3
title: Optimal State Values and Bellman Optimality Equation
key_topics: [optimal policy, optimal state value, policy improvement, action value, Bellman optimality equation, BOE elementwise form, BOE matrix-vector form, fixed point, contraction mapping, contraction mapping theorem, value iteration, greedy policy, existence of optimal policy, uniqueness of optimal state value, non-uniqueness of optimal policy, deterministic optimal policy, discount rate impact, reward invariance, affine transformation of rewards, meaningless detours]
depends_on: [1, 2]
required_by: [4, 5, 7, 8, 9, 10]
---

# Chapter 3: Optimal State Values and Bellman Optimality Equation

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 3, pp. 35-55
> Supplemented by: Lecture slides L3 (46 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter introduces two central elements of reinforcement learning: the **core concept** of optimal state values (which define optimal policies) and the **core tool** of the Bellman optimality equation (BOE), from which optimal state values and policies can be solved. The chapter is slightly more mathematically intensive than previous chapters but provides clear answers to fundamental questions about the existence, uniqueness, and computation of optimal policies.

**Position in the book**: Chapter 3 bridges the Bellman equation (Chapter 2) and the value iteration / policy iteration algorithms (Chapter 4). Specifically:
- **Chapter 2** introduced the Bellman equation for *any given* policy.
- **Chapter 3** introduces the Bellman optimality equation, which is a *special* Bellman equation whose corresponding policy is optimal.
- **Chapter 4** will introduce the **value iteration** algorithm, which is exactly the iterative algorithm for solving the BOE as derived in this chapter.

---

## 3.1 Motivating Example: How to Improve Policies?

### Setup

Consider a simple grid world with four states $s_1, s_2, s_3, s_4$:
- Orange cell (forbidden): $s_2$
- Blue cell (target): $s_3, s_4$
- The given policy selects $a_2$ (rightward) at $s_1$, leading through the forbidden area

The policy is not good because it selects $a_2$ (rightward) at state $s_1$, causing the agent to enter the forbidden area and receive $r = -1$.

### Step 1: Calculate State Values of the Given Policy

Using the Bellman equation (from Chapter 2) with $\gamma = 0.9$:

$$v_\pi(s_1) = -1 + \gamma v_\pi(s_2)$$
$$v_\pi(s_2) = +1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_3) = +1 + \gamma v_\pi(s_4)$$
$$v_\pi(s_4) = +1 + \gamma v_\pi(s_4)$$

Solving the last equation first: $v_\pi(s_4) = 1/(1 - 0.9) = 10$. Then:

$$v_\pi(s_4) = v_\pi(s_3) = v_\pi(s_2) = 10, \quad v_\pi(s_1) = 8$$

### Step 2: Calculate Action Values for State $s_1$

Using the action value formula $q_\pi(s, a) = r(s, a) + \gamma v_\pi(s')$:

| Action | Calculation | Action Value |
|---|---|---|
| $a_1$ (up) | $-1 + \gamma v_\pi(s_1) = -1 + 0.9 \times 8$ | $q_\pi(s_1, a_1) = 6.2$ |
| $a_2$ (right) | $-1 + \gamma v_\pi(s_2) = -1 + 0.9 \times 10$ | $q_\pi(s_1, a_2) = 8.0$ |
| $a_3$ (down) | $0 + \gamma v_\pi(s_3) = 0 + 0.9 \times 10$ | $q_\pi(s_1, a_3) = 9.0$ |
| $a_4$ (left) | $-1 + \gamma v_\pi(s_1) = -1 + 0.9 \times 8$ | $q_\pi(s_1, a_4) = 6.2$ |
| $a_5$ (stay) | $0 + \gamma v_\pi(s_1) = 0 + 0.9 \times 8$ | $q_\pi(s_1, a_5) = 7.2$ |

### Step 3: Improve the Policy

Action $a_3$ has the **greatest action value**: $q_\pi(s_1, a_3) = 9.0 \geq q_\pi(s_1, a_i)$ for all $i \neq 3$.

Therefore, updating the policy to select $a_3$ at $s_1$ yields a better policy.

### Key Insight (from slides)

- **Intuition**: Actions with greater action values are better.
- **Mathematics**: The formal justification that selecting actions with the greatest action value always improves the policy is nontrivial and is addressed in the remainder of this chapter and in Chapter 4.

### Open Questions Raised

1. If the policy is not good for *multiple* states, will selecting the action with the greatest action value *at every state simultaneously* still generate a better policy?
2. Do optimal policies always exist?
3. What does an optimal policy look like?

---

## 3.2 Optimal State Values and Optimal Policies

### Partial Ordering of Policies

Two policies $\pi_1$ and $\pi_2$ can be compared via their state values:

If $v_{\pi_1}(s) \geq v_{\pi_2}(s)$ for all $s \in \mathcal{S}$, then $\pi_1$ is said to be **better** than $\pi_2$.

### Definition 3.1 (Optimal Policy and Optimal State Value)

> **Definition**: A policy $\pi^*$ is **optimal** if $v_{\pi^*}(s) \geq v_\pi(s)$ for all $s \in \mathcal{S}$ and for any other policy $\pi$. The state values of $\pi^*$ are the **optimal state values**.

**Interpretation**: An optimal policy has the greatest state value for *every* state compared to *all* other policies.

### Fundamental Questions About Optimal Policies

| Question | Category |
|---|---|
| Does the optimal policy exist? | Existence |
| Is the optimal policy unique? | Uniqueness |
| Is the optimal policy stochastic or deterministic? | Stochasticity |
| How to obtain the optimal policy and optimal state values? | Algorithm |

**Important note (from slides)**: These questions *must* be answered clearly. For example, if optimal policies do not exist, there is no need to design algorithms to find them.

**Important note (from Q&A)**: This specific definition of optimality is valid only for **tabular** reinforcement learning. When values or policies are approximated by functions (Chapters 8 and 9), different metrics must be used.

---

## 3.3 Bellman Optimality Equation (BOE)

The BOE is the tool for analyzing optimal policies and optimal state values. By solving this equation, we can obtain both.

### 3.3.1 Elementwise Form of the BOE

For every $s \in \mathcal{S}$:

$$v(s) = \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \left( \sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v(s') \right)$$

$$= \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a|s) \, q(s,a) \tag{3.1}$$

where:
- $v(s)$, $v(s')$ are **unknown variables** to be solved
- $q(s,a) \doteq \sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v(s')$
- $p(r|s,a)$, $p(s'|s,a)$, $r$, $\gamma$ are **known**
- $\pi(s)$ denotes a policy for state $s$, and $\Pi(s)$ is the set of all possible policies for state $s$

**Key observation (from slides)**: The BOE is "tricky yet elegant."
- **Elegant**: It describes the optimal policy and optimal state value in a compact equation.
- **Tricky**: The maximization on the right-hand side makes it a *nonlinear* equation, which may not be straightforward to solve.

**Two unknowns, one equation**: The BOE has two unknown variables, $v(s)$ and $\pi(a|s)$. These can be solved **one by one** (first $\pi$, then $v$), as demonstrated below.

### Solving the Maximization on the Right-Hand Side

#### Example 3.1 (Two unknowns from one equation)

Consider $x = \max_{y \in \mathbb{R}} (2x - 1 - y^2)$.

- **Step 1** (solve $y$): Regardless of $x$, $\max_y(2x - 1 - y^2) = 2x - 1$, achieved at $y = 0$.
- **Step 2** (solve $x$): With $y = 0$, the equation becomes $x = 2x - 1$, giving $x = 1$.
- **Solution**: $y = 0$, $x = 1$.

#### Example 3.2 (Maximizing a weighted sum under probability constraints)

Given $q_1, q_2, q_3 \in \mathbb{R}$, find $c_1^*, c_2^*, c_3^*$ to maximize:

$$\sum_{i=1}^3 c_i q_i = c_1 q_1 + c_2 q_2 + c_3 q_3$$

subject to $c_1 + c_2 + c_3 = 1$ and $c_1, c_2, c_3 \geq 0$.

**Solution**: Without loss of generality, suppose $q_3 \geq q_1, q_2$. Then $c_3^* = 1$ and $c_1^* = c_2^* = 0$.

**Proof**:
$$q_3 = (c_1 + c_2 + c_3) q_3 = c_1 q_3 + c_2 q_3 + c_3 q_3 \geq c_1 q_1 + c_2 q_2 + c_3 q_3$$

#### Applying to the BOE

Since $\sum_a \pi(a|s) = 1$:

$$\sum_{a \in \mathcal{A}} \pi(a|s) \, q(s,a) \leq \sum_{a \in \mathcal{A}} \pi(a|s) \max_{a \in \mathcal{A}} q(s,a) = \max_{a \in \mathcal{A}} q(s,a)$$

Equality is achieved when:

$$\pi(a|s) = \begin{cases} 1, & a = a^* \\ 0, & a \neq a^* \end{cases}$$

where $a^* = \arg\max_a q(s,a)$.

**Conclusion**: The optimal policy $\pi(s)$ selects the action with the greatest value of $q(s,a)$. The maximization $\max_\pi \sum_a \pi(a|s) q(s,a) = \max_{a \in \mathcal{A}} q(s,a)$.

### 3.3.2 Matrix-Vector Form of the BOE

Combining the elementwise equations for all states:

$$\mathbf{v} = \max_{\pi \in \Pi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}) \tag{3.2}$$

where $\mathbf{v} \in \mathbb{R}^{|\mathcal{S}|}$ and $\max_\pi$ is performed **elementwise** (each state $s$ independently optimizes its own policy $\pi(s)$).

The structures of $\mathbf{r}_\pi$ and $\mathbf{P}_\pi$ are the same as in the normal Bellman equation (Chapter 2):

$$[\mathbf{r}_\pi]_s \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a) r$$

$$[\mathbf{P}_\pi]_{s,s'} = p(s'|s) \doteq \sum_{a \in \mathcal{A}} \pi(a|s) \, p(s'|s,a)$$

**Elementwise max interpretation (from slides)**:
$$\max_\pi \begin{pmatrix} * \\ \vdots \\ * \end{pmatrix} = \begin{pmatrix} \max_{\pi(s_1)} * \\ \vdots \\ \max_{\pi(s_n)} * \end{pmatrix}$$

### Compact Form: $v = f(v)$

Define the right-hand side as a function of $\mathbf{v}$:

$$f(\mathbf{v}) \doteq \max_{\pi \in \Pi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}) \tag{3.3}$$

Then the BOE is:

$$\mathbf{v} = f(\mathbf{v})$$

This is a **nonlinear equation** (due to the $\max$ operator). The remainder of the section shows how to solve it.

### 3.3.3 Contraction Mapping Theorem

The contraction mapping theorem is the key mathematical tool for analyzing the BOE. It is also known as the **fixed-point theorem** or **Banach fixed-point theorem**.

#### Fixed Point

A point $x^*$ is a **fixed point** of $f$ if:

$$f(x^*) = x^*$$

**Interpretation**: The map of $x^*$ is itself -- it is "fixed" under the mapping.

#### Contraction Mapping

A function $f : \mathbb{R}^d \to \mathbb{R}^d$ is a **contraction mapping** (or contractive function) if there exists $\gamma \in (0, 1)$ such that:

$$\|f(x_1) - f(x_2)\| \leq \gamma \|x_1 - x_2\|$$

for any $x_1, x_2 \in \mathbb{R}^d$. Here $\|\cdot\|$ denotes a vector or matrix norm.

**Key requirement (from slides)**: $\gamma$ must be *strictly* less than 1 so that limits such as $\gamma^k \to 0$ as $k \to \infty$ hold.

#### Example 3.3 (Three examples of contraction mappings and fixed points)

**Example (a)**: $x = f(x) = 0.5x$, $x \in \mathbb{R}$.
- Fixed point: $x = 0$ (since $0 = 0.5 \cdot 0$).
- Contraction: $|0.5x_1 - 0.5x_2| = 0.5|x_1 - x_2| \leq \gamma|x_1 - x_2|$ for any $\gamma \in [0.5, 1)$.

**Example (b)**: $x = f(x) = Ax$, where $x \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$, $\|A\| \leq \gamma < 1$.
- Fixed point: $x = 0$ (since $0 = A \cdot 0$).
- Contraction: $\|Ax_1 - Ax_2\| = \|A(x_1 - x_2)\| \leq \|A\|\|x_1 - x_2\| \leq \gamma\|x_1 - x_2\|$.

**Example (c)**: $x = f(x) = 0.5\sin x$, $x \in \mathbb{R}$.
- Fixed point: $x = 0$ (since $0 = 0.5 \sin 0$).
- Contraction: By the mean value theorem, $\left|\frac{0.5\sin x_1 - 0.5\sin x_2}{x_1 - x_2}\right| = |0.5\cos x_3| \leq 0.5$ for some $x_3 \in [x_1, x_2]$. Hence $|0.5\sin x_1 - 0.5\sin x_2| \leq 0.5|x_1 - x_2|$.

#### Theorem 3.1 (Contraction Mapping Theorem)

> **Theorem**: For any equation of the form $x = f(x)$ where $x$ and $f(x)$ are real vectors, if $f$ is a contraction mapping, then the following properties hold:
>
> 1. **Existence**: There exists a fixed point $x^*$ satisfying $f(x^*) = x^*$.
> 2. **Uniqueness**: The fixed point $x^*$ is unique.
> 3. **Algorithm**: The iterative process $x_{k+1} = f(x_k)$, $k = 0, 1, 2, \ldots$, converges $x_k \to x^*$ as $k \to \infty$ for **any** initial guess $x_0$. Moreover, the convergence rate is **exponentially fast**.

**Significance**: The contraction mapping theorem not only tells whether a solution exists but also suggests a numerical algorithm for finding it.

#### Example 3.4 (Iterative algorithms from contraction mapping theorem)

For the three contraction mappings from Example 3.3, the unique fixed point $x^* = 0$ can be iteratively computed by:

$$x_{k+1} = 0.5 x_k, \qquad x_{k+1} = A x_k, \qquad x_{k+1} = 0.5 \sin x_k$$

given any initial guess $x_0$.

#### Proof of the Contraction Mapping Theorem (Box 3.1)

The proof has four parts:

**Part 1: Convergence of $\{x_k\}$ (via Cauchy sequences)**

A sequence $x_1, x_2, \ldots$ is **Cauchy** if for any small $\varepsilon > 0$, there exists $N$ such that $\|x_m - x_n\| < \varepsilon$ for all $m, n > N$. Cauchy sequences are guaranteed to converge to a limit.

*Important subtlety*: Having $\|x_{n+1} - x_n\| \to 0$ is **not sufficient** for a Cauchy sequence (counterexample: $x_n = \sqrt{n}$, where $x_{n+1} - x_n \to 0$ but $x_n$ diverges).

From the contraction property, repeatedly applying gives:

$$\|x_{k+1} - x_k\| \leq \gamma^k \|x_1 - x_0\|$$

For $m > n$:

$$\|x_m - x_n\| \leq \|x_m - x_{m-1}\| + \cdots + \|x_{n+1} - x_n\| \leq \gamma^n(1 + \gamma + \gamma^2 + \cdots)\|x_1 - x_0\| = \frac{\gamma^n}{1-\gamma}\|x_1 - x_0\| \tag{3.4}$$

Since $\gamma < 1$, for any $\varepsilon$ we can find $N$ such that $\|x_m - x_n\| < \varepsilon$ for all $m, n > N$. The sequence is Cauchy and converges to a limit $x^* = \lim_{k \to \infty} x_k$.

**Part 2: The limit is a fixed point**

Since $\|f(x_k) - x_k\| = \|x_{k+1} - x_k\| \leq \gamma^k\|x_1 - x_0\| \to 0$, we have $f(x^*) = x^*$ at the limit.

**Part 3: Uniqueness**

Suppose another fixed point $x'$ exists with $f(x') = x'$. Then:

$$\|x' - x^*\| = \|f(x') - f(x^*)\| \leq \gamma\|x' - x^*\|$$

Since $\gamma < 1$, this holds only if $\|x' - x^*\| = 0$, so $x' = x^*$.

**Part 4: Exponential convergence rate**

From (3.4), since $m$ can be arbitrarily large:

$$\|x^* - x_n\| = \lim_{m \to \infty}\|x_m - x_n\| \leq \frac{\gamma^n}{1 - \gamma}\|x_1 - x_0\|$$

Since $\gamma < 1$, the error converges to zero **exponentially fast** as $n \to \infty$.

### 3.3.4 Contraction Property of the Right-Hand Side of the BOE

#### Theorem 3.2 (Contraction Property of $f(\mathbf{v})$)

> **Theorem**: The function $f(\mathbf{v})$ on the right-hand side of the BOE in (3.3) is a contraction mapping. In particular, for any $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{|\mathcal{S}|}$:
>
> $$\|f(\mathbf{v}_1) - f(\mathbf{v}_2)\|_\infty \leq \gamma \|\mathbf{v}_1 - \mathbf{v}_2\|_\infty$$
>
> where $\gamma \in (0,1)$ is the discount rate, and $\|\cdot\|_\infty$ is the **maximum norm** (the maximum absolute value of the elements of a vector).

**Key insight (from slides)**: The contraction factor is exactly the discount rate $\gamma$! This connects the discount rate to convergence speed.

#### Proof of Theorem 3.2 (Box 3.2)

Consider $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{|\mathcal{S}|}$. Let $\pi_1^* = \arg\max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_1)$ and $\pi_2^* = \arg\max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_2)$.

**Step 1**: By the definition of max:

$$f(\mathbf{v}_1) = \mathbf{r}_{\pi_1^*} + \gamma \mathbf{P}_{\pi_1^*} \mathbf{v}_1 \geq \mathbf{r}_{\pi_2^*} + \gamma \mathbf{P}_{\pi_2^*} \mathbf{v}_1$$
$$f(\mathbf{v}_2) = \mathbf{r}_{\pi_2^*} + \gamma \mathbf{P}_{\pi_2^*} \mathbf{v}_2 \geq \mathbf{r}_{\pi_1^*} + \gamma \mathbf{P}_{\pi_1^*} \mathbf{v}_2$$

**Step 2**: Upper bound on $f(\mathbf{v}_1) - f(\mathbf{v}_2)$:

$$f(\mathbf{v}_1) - f(\mathbf{v}_2) \leq (\mathbf{r}_{\pi_1^*} + \gamma \mathbf{P}_{\pi_1^*}\mathbf{v}_1) - (\mathbf{r}_{\pi_1^*} + \gamma \mathbf{P}_{\pi_1^*}\mathbf{v}_2) = \gamma \mathbf{P}_{\pi_1^*}(\mathbf{v}_1 - \mathbf{v}_2)$$

Similarly: $f(\mathbf{v}_2) - f(\mathbf{v}_1) \leq \gamma \mathbf{P}_{\pi_2^*}(\mathbf{v}_2 - \mathbf{v}_1)$

Therefore:

$$\gamma \mathbf{P}_{\pi_2^*}(\mathbf{v}_1 - \mathbf{v}_2) \leq f(\mathbf{v}_1) - f(\mathbf{v}_2) \leq \gamma \mathbf{P}_{\pi_1^*}(\mathbf{v}_1 - \mathbf{v}_2)$$

**Step 3**: Define $\mathbf{z} = \max\left(|\gamma \mathbf{P}_{\pi_2^*}(\mathbf{v}_1 - \mathbf{v}_2)|, |\gamma \mathbf{P}_{\pi_1^*}(\mathbf{v}_1 - \mathbf{v}_2)|\right) \in \mathbb{R}^{|\mathcal{S}|}$ (elementwise), so $|f(\mathbf{v}_1) - f(\mathbf{v}_2)| \leq \mathbf{z}$ and $\|f(\mathbf{v}_1) - f(\mathbf{v}_2)\|_\infty \leq \|\mathbf{z}\|_\infty$.

**Step 4**: For the $i$-th entry, let $\mathbf{p}_i^T$ and $\mathbf{q}_i^T$ be the $i$-th rows of $\mathbf{P}_{\pi_1^*}$ and $\mathbf{P}_{\pi_2^*}$:

$$z_i = \max\{\gamma|\mathbf{p}_i^T(\mathbf{v}_1 - \mathbf{v}_2)|, \gamma|\mathbf{q}_i^T(\mathbf{v}_1 - \mathbf{v}_2)|\}$$

Since $\mathbf{p}_i$ has all nonnegative elements summing to 1:

$$|\mathbf{p}_i^T(\mathbf{v}_1 - \mathbf{v}_2)| \leq \mathbf{p}_i^T|\mathbf{v}_1 - \mathbf{v}_2| \leq \|\mathbf{v}_1 - \mathbf{v}_2\|_\infty$$

Similarly for $\mathbf{q}_i$. Therefore $z_i \leq \gamma\|\mathbf{v}_1 - \mathbf{v}_2\|_\infty$, giving $\|\mathbf{z}\|_\infty \leq \gamma\|\mathbf{v}_1 - \mathbf{v}_2\|_\infty$, which concludes the proof.

---

## 3.4 Solving an Optimal Policy from the BOE

### Solving $v^*$

If $v^*$ is a solution of the BOE, then $v^* = f(v^*)$, so $v^*$ is a fixed point. By the contraction mapping theorem (since $f$ is a contraction mapping by Theorem 3.2):

#### Theorem 3.3 (Existence, Uniqueness, and Algorithm)

> **Theorem**: For the BOE $\mathbf{v} = f(\mathbf{v}) = \max_{\pi \in \Pi}(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v})$:
>
> 1. There **always exists** a unique solution $\mathbf{v}^*$.
> 2. The solution can be computed iteratively by:
>
> $$\mathbf{v}_{k+1} = f(\mathbf{v}_k) = \max_{\pi \in \Pi}(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_k), \quad k = 0, 1, 2, \ldots$$
>
> 3. $\mathbf{v}_k$ converges to $\mathbf{v}^*$ **exponentially fast** as $k \to \infty$ for **any** initial guess $\mathbf{v}_0$.
> 4. The convergence rate is determined by $\gamma$.

**Important answers**:
- **Existence of $v^*$**: The solution of the BOE always exists.
- **Uniqueness of $v^*$**: The solution $v^*$ is always unique.
- **Algorithm for solving $v^*$**: This iterative algorithm is called **value iteration**. Its detailed implementation is given in Chapter 4.

**Note (from slides)**: The convergence rate is determined by $\gamma$. Smaller $\gamma$ leads to faster convergence.

### Solving $\pi^*$

Once $v^*$ is obtained, the optimal policy is:

$$\pi^* = \arg\max_{\pi \in \Pi}(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*) \tag{3.6}$$

Substituting (3.6) into the BOE gives:

$$\mathbf{v}^* = \mathbf{r}_{\pi^*} + \gamma \mathbf{P}_{\pi^*} \mathbf{v}^*$$

Therefore, $\mathbf{v}^* = v_{\pi^*}$ is the state value of $\pi^*$, and **the BOE is a special Bellman equation whose corresponding policy is $\pi^*$**.

### Optimality of the Solution

#### Theorem 3.4 (Optimality of $v^*$ and $\pi^*$)

> **Theorem**: The solution $\mathbf{v}^*$ is the optimal state value, and $\pi^*$ is an optimal policy. That is, for any policy $\pi$:
>
> $$\mathbf{v}^* = v_{\pi^*} \geq v_\pi$$
>
> where $v_\pi$ is the state value of $\pi$, and $\geq$ is an elementwise comparison.

**This is why we study the BOE**: Its solution corresponds to optimal state values and optimal policies.

#### Proof of Theorem 3.4 (Box 3.3)

For any policy $\pi$: $v_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi v_\pi$.

Since $\mathbf{v}^* = \max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*) \geq \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*$:

$$\mathbf{v}^* - v_\pi \geq (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*) - (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi v_\pi) = \gamma \mathbf{P}_\pi(\mathbf{v}^* - v_\pi)$$

Repeatedly applying: $\mathbf{v}^* - v_\pi \geq \gamma^n \mathbf{P}_\pi^n(\mathbf{v}^* - v_\pi)$.

Taking the limit: $\mathbf{v}^* - v_\pi \geq \lim_{n \to \infty} \gamma^n \mathbf{P}_\pi^n(\mathbf{v}^* - v_\pi) = 0$

(since $\gamma < 1$ and $\mathbf{P}_\pi^n$ is a nonnegative matrix with $\mathbf{P}_\pi^n \mathbf{1} = \mathbf{1}$, all elements $\leq 1$).

Therefore $\mathbf{v}^* \geq v_\pi$ for any $\pi$.

### The Greedy Optimal Policy

#### Theorem 3.5 (Greedy Optimal Policy)

> **Theorem**: For any $s \in \mathcal{S}$, the deterministic greedy policy
>
> $$\pi^*(a|s) = \begin{cases} 1, & a = a^*(s) \\ 0, & a \neq a^*(s) \end{cases} \tag{3.7}$$
>
> is an optimal policy for solving the BOE. Here,
>
> $$a^*(s) = \arg\max_a q^*(s, a)$$
>
> where
>
> $$q^*(s, a) \doteq \sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v^*(s')$$

**Proof (Box 3.4)**: The elementwise form of $\pi^* = \arg\max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*)$ is:

$$\pi^*(s) = \arg\max_{\pi \in \Pi} \sum_{a \in \mathcal{A}} \pi(a|s) \underbrace{\left(\sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v^*(s')\right)}_{q^*(s,a)}, \quad s \in \mathcal{S}$$

Since $\sum_a \pi(a|s) q^*(s,a)$ is maximized when $\pi(s)$ assigns all probability to the action with the greatest $q^*(s,a)$, the greedy deterministic policy is optimal.

The policy is called **greedy** because it selects the action with the greatest $q^*(s,a)$.

### Properties of Optimal Policies

#### Uniqueness of Optimal Policies

Although $v^*$ is unique, the optimal policy may **not** be unique. Multiple or even infinitely many optimal policies can share the same optimal state values.

**Example**: Two different policies (one going right then down, another going down then right to reach the target) can both be optimal, yielding the same state values.

#### Stochasticity of Optimal Policies

An optimal policy can be either **deterministic** or **stochastic**. However, it is guaranteed that **there always exists a deterministic optimal policy** (Theorem 3.5).

**Example**: A stochastic policy that moves right with probability 0.5 and down with probability 0.5 can also be optimal when both actions lead to equally good outcomes.

---

## 3.5 Factors That Influence Optimal Policies

From the BOE, the optimal state value and optimal policy are determined by three factors:

1. **Immediate reward** $r$
2. **Discount rate** $\gamma$
3. **System model** $p(s'|s,a)$, $p(r|s,a)$

Since the system model is fixed, we examine how $r$ and $\gamma$ affect the optimal policy.

### Baseline Example

**Parameters**: 5x5 grid world with $r_{\text{boundary}} = r_{\text{forbidden}} = -1$, $r_{\text{target}} = 1$, $r_{\text{other}} = 0$, $\gamma = 0.9$.

**Optimal State Values** (Figure 3.4(a)):

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|---|
| Row 1 | 5.8 | 5.6 | 6.2 | 6.5 | 5.8 |
| Row 2 | 6.5 | 7.2 | 8.0 | 7.2 | 6.5 |
| Row 3 | 7.2 | 8.0 | 10.0 | 8.0 | 7.2 |
| Row 4 | 8.0 | 10.0 | 10.0 | 10.0 | 8.0 |
| Row 5 | 7.2 | 9.0 | 10.0 | 9.0 | 8.1 |

**Observation**: The agent is **not afraid** of passing through forbidden areas to reach the target. Starting from (row 4, col 1), the agent passes through forbidden areas rather than traveling the long way around. The cumulative reward of the shorter path through forbidden areas is greater than the longer safe path.

**Insight (from slides)**: The optimal policy "dares to take risks" -- entering forbidden areas -- because the far-sighted discount rate $\gamma = 0.9$ makes future rewards significant.

### Impact of the Discount Rate

#### $\gamma = 0.5$ (Figure 3.4(b))

**Optimal State Values**:

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|---|
| Row 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Row 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| Row 3 | 0.0 | 0.0 | 2.0 | 0.1 | 0.1 |
| Row 4 | 0.0 | 2.0 | 2.0 | 2.0 | 0.2 |
| Row 5 | 0.0 | 1.0 | 2.0 | 1.0 | 0.5 |

**Observation**: The agent becomes **short-sighted** and does not dare to take risks. It travels the long distance to reach the target while avoiding all forbidden areas. The state values are much lower overall (most are 0.0).

**Insight (from slides)**: The optimal policy becomes "short-sighted" -- it avoids all forbidden areas because it does not value distant future rewards enough to justify the immediate penalty of entering forbidden cells.

#### $\gamma = 0$ (Figure 3.4(c))

**Optimal State Values**:

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|---|
| Row 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Row 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Row 3 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| Row 4 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| Row 5 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |

**Observation**: The agent is **extremely short-sighted** and simply selects the action with the greatest **immediate reward** instead of the greatest total reward. Only states directly adjacent to the target have nonzero values. The agent **cannot reach the target** from distant states.

**Insight (from slides)**: At $\gamma = 0$, the agent only considers immediate rewards. This makes it impossible to learn long-horizon tasks.

#### Spatial Distribution Pattern

Across all examples, states closer to the target have **greater** state values, while those farther away have **lower** values. This is explained by the discount rate: a state requiring a longer trajectory to reach the target has a smaller state value due to heavier discounting.

### Impact of Reward Values

#### Increased Punishment: $r_{\text{forbidden}} = -10$ (Figure 3.4(d))

**Optimal State Values**:

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|---|
| Row 1 | 3.5 | 3.9 | 4.3 | 4.8 | 5.3 |
| Row 2 | 3.1 | 3.5 | 4.8 | 5.3 | 5.9 |
| Row 3 | 2.8 | 2.5 | 10.0 | 5.9 | 6.6 |
| Row 4 | 2.5 | 10.0 | 10.0 | 10.0 | 7.3 |
| Row 5 | 2.3 | 9.0 | 10.0 | 9.0 | 8.1 |

**Observation**: The optimal policy now avoids all forbidden areas, even with $\gamma = 0.9$. The heavy punishment makes entering forbidden cells too costly even when accounting for future rewards.

### Theorem 3.6 (Optimal Policy Invariance Under Affine Transformations)

> **Theorem**: Consider an MDP with optimal state value $\mathbf{v}^*$ satisfying $\mathbf{v}^* = \max_{\pi \in \Pi}(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*)$. If every reward $r \in \mathcal{R}$ is changed by an affine transformation to $\alpha r + \beta$, where $\alpha, \beta \in \mathbb{R}$ and $\alpha > 0$, then the corresponding optimal state value $\mathbf{v}'$ is:
>
> $$\mathbf{v}' = \alpha \mathbf{v}^* + \frac{\beta}{1 - \gamma} \mathbf{1} \tag{3.8}$$
>
> where $\gamma \in (0,1)$ is the discount rate and $\mathbf{1} = [1, \ldots, 1]^T$. Consequently, the optimal policy derived from $\mathbf{v}'$ is **invariant** to the affine transformation of reward values.

**Interpretation**: What matters is the **relative** reward values, not their absolute values. Scaling all rewards or adding a constant to all rewards does not change the optimal policy.

**Example (from slides)**: Changing rewards from $(r_{\text{boundary}}, r_{\text{forbidden}}, r_{\text{target}}, r_{\text{other}}) = (-1, -1, 1, 0)$ to $(0, 0, 2, 1)$ (adding 1 to all) yields the **same optimal policy**.

#### Proof of Theorem 3.6 (Box 3.5)

For any policy $\pi$, define $r_\pi(s) = \sum_a \pi(a|s) \sum_r p(r|s,a) r$. If $r \to \alpha r + \beta$, then $r_\pi(s) \to \alpha r_\pi(s) + \beta$, hence $\mathbf{r}_\pi \to \alpha \mathbf{r}_\pi + \beta\mathbf{1}$.

The new BOE becomes: $\mathbf{v}' = \max_{\pi \in \Pi}(\alpha \mathbf{r}_\pi + \beta\mathbf{1} + \gamma \mathbf{P}_\pi \mathbf{v}')$.

Substituting $\mathbf{v}' = \alpha \mathbf{v}^* + c\mathbf{1}$ with $c = \beta/(1-\gamma)$:

$$\alpha \mathbf{v}^* + c\mathbf{1} = \max_{\pi \in \Pi}(\alpha \mathbf{r}_\pi + \beta\mathbf{1} + \gamma \mathbf{P}_\pi(\alpha \mathbf{v}^* + c\mathbf{1})) = \max_{\pi \in \Pi}(\alpha \mathbf{r}_\pi + \beta\mathbf{1} + \alpha\gamma \mathbf{P}_\pi \mathbf{v}^* + c\gamma\mathbf{1})$$

where $\mathbf{P}_\pi \mathbf{1} = \mathbf{1}$ was used. Rearranging:

$$\alpha \mathbf{v}^* = \max_{\pi \in \Pi}(\alpha \mathbf{r}_\pi + \alpha\gamma \mathbf{P}_\pi \mathbf{v}^*) + \beta\mathbf{1} + c\gamma\mathbf{1} - c\mathbf{1}$$

This requires $\beta\mathbf{1} + c\gamma\mathbf{1} - c\mathbf{1} = \mathbf{0}$, which holds when $c = \beta/(1-\gamma)$. Therefore $\mathbf{v}' = \alpha \mathbf{v}^* + c\mathbf{1}$ is the unique solution.

Since $\mathbf{v}'$ is an affine transformation of $\mathbf{v}^*$, the relative relationships between action values remain the same. Hence $\arg\max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}')$ equals $\arg\max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}^*)$.

### Avoiding Meaningless Detours

**Question**: Since $r_{\text{other}} = 0$ (no punishment for ordinary steps), will the optimal policy take meaningless detours? Should we set $r_{\text{other}}$ to be negative?

**Answer**: No. The discount rate automatically prevents meaningless detours.

#### Worked Example (Figure 3.5)

Consider a 2x2 grid where the bottom-right cell is the target. Two policies for state $s_2$:

**Policy (a)** (optimal): Agent moves directly down at $s_2$. Trajectory: $s_2 \to s_4$.
$$\text{return} = 1 + \gamma \cdot 1 + \gamma^2 \cdot 1 + \cdots = \frac{1}{1-\gamma} = 10 \quad (\gamma = 0.9)$$

State value at $s_2$: $v(s_2) = 10$.

**Policy (b)** (non-optimal): Agent moves left at $s_2$, taking a detour. Trajectory: $s_2 \to s_1 \to s_3 \to s_4$.
$$\text{return} = 0 + \gamma \cdot 0 + \gamma^2 \cdot 1 + \gamma^3 \cdot 1 + \cdots = \frac{\gamma^2}{1-\gamma} = 8.1 \quad (\gamma = 0.9)$$

State value at $s_2$: $v(s_2) = 8.1$.

**Conclusion**: The shorter trajectory yields a greater discounted return. The discount rate acts as an implicit punishment for detours.

#### Common Misunderstanding

A beginner may think adding a negative reward (e.g., $-1$) to every step is necessary to encourage the agent to reach the target quickly. This is wrong for two reasons:

1. Adding the same reward to all steps is an **affine transformation**, which does not change the optimal policy (Theorem 3.6).
2. The **discount rate** already encourages shortest paths: meaningless detours increase trajectory length and reduce the discounted return.

---

## 3.6 Summary

| Aspect | Key Result |
|---|---|
| Core concept | Optimal state values and optimal policies |
| Core tool | Bellman optimality equation (BOE) |
| BOE nature | Nonlinear equation with contraction property |
| Analysis method | Contraction mapping theorem |
| Existence | Optimal $v^*$ always exists |
| Uniqueness of $v^*$ | Unique |
| Uniqueness of $\pi^*$ | Not unique (multiple optimal policies possible) |
| Stochasticity of $\pi^*$ | Can be stochastic or deterministic; a deterministic one always exists |
| Algorithm | Iterative: $v_{k+1} = f(v_k)$ (value iteration, detailed in Ch. 4) |
| BOE as Bellman eq. | BOE is a special Bellman equation whose policy is optimal |
| Factors | Reward values, discount rate, system model |
| Reward invariance | Optimal policy invariant to affine transformations of rewards |
| Detours | Discount rate prevents meaningless detours |

---

## 3.7 Q&A -- Important Clarifications

### Q: What is the definition of optimal policies?
**A**: A policy is optimal if its corresponding state values are greater than or equal to those of any other policy for every state. This definition is valid only for **tabular** reinforcement learning. For function approximation (Chapters 8-9), different metrics are needed.

### Q: Why is the Bellman optimality equation important?
**A**: It characterizes both optimal policies and optimal state values. Solving this equation yields an optimal policy and the corresponding optimal state value.

### Q: Is the Bellman optimality equation a Bellman equation?
**A**: Yes. The BOE is a special Bellman equation whose corresponding policy is optimal.

### Q: Is the solution of the BOE unique?
**A**: The BOE has two unknown variables. The **value solution** (optimal state value $v^*$) is unique. The **policy solution** (optimal policy $\pi^*$) may not be unique.

### Q: What is the key property of the BOE for analyzing its solution?
**A**: The right-hand side of the BOE is a **contraction mapping**. This allows applying the contraction mapping theorem to establish existence, uniqueness, and an iterative algorithm.

### Q: Do optimal policies exist?
**A**: Yes. Optimal policies always exist, as shown by the analysis of the BOE.

### Q: Are optimal policies unique?
**A**: No. There may exist multiple or infinitely many optimal policies that share the same optimal state values.

### Q: Are optimal policies stochastic or deterministic?
**A**: An optimal policy can be either. A key fact is that there **always** exists a deterministic greedy optimal policy (Theorem 3.5).

### Q: How to obtain an optimal policy?
**A**: Solve the BOE using the iterative algorithm from Theorem 3.3. The detailed implementation (value iteration) is given in Chapter 4. All RL algorithms in this book aim to obtain optimal policies under different settings.

### Q: What is the general impact of reducing the discount rate?
**A**: The optimal policy becomes more **short-sighted**. The agent does not dare to take risks even though greater cumulative rewards may be available afterward.

### Q: What happens if the discount rate is zero?
**A**: The optimal policy becomes **extremely short-sighted**, selecting only the action with the greatest immediate reward. The agent cannot plan ahead and may fail to reach distant targets.

### Q: If we increase all rewards by the same amount, do state values and policy change?
**A**: The optimal **policy** does not change (affine invariance, Theorem 3.6). The optimal **state values** do increase, following $v' = v^* + \frac{\beta}{1-\gamma}\mathbf{1}$.

### Q: Should we add a negative reward to every step to avoid meaningless detours?
**A**: No. (1) Adding the same reward to every step is an affine transformation that does not change the optimal policy. (2) The discount rate already encourages reaching the target quickly because detours increase trajectory length and reduce the discounted return.

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| Optimal policy | $\pi^*$ | 3.2 |
| Optimal state value | $v^*(s)$ or $v_{\pi^*}(s)$ | 3.2 |
| Policy comparison (partial order) | $v_{\pi_1}(s) \geq v_{\pi_2}(s) \; \forall s$ | 3.2 |
| Bellman optimality equation (elementwise) | $v(s) = \max_\pi \sum_a \pi(a|s) q(s,a)$ | 3.3 |
| Bellman optimality equation (matrix-vector) | $\mathbf{v} = \max_\pi(\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v})$ | 3.3.2 |
| Bellman optimality equation (compact) | $\mathbf{v} = f(\mathbf{v})$ | 3.3.2 |
| Optimal action value | $q^*(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v^*(s')$ | 3.4 |
| Fixed point | $f(x^*) = x^*$ | 3.3.3 |
| Contraction mapping | $\|f(x_1) - f(x_2)\| \leq \gamma\|x_1 - x_2\|$ | 3.3.3 |
| Contraction mapping theorem | Existence + Uniqueness + Algorithm | 3.3.3 |
| Cauchy sequence | $\|x_m - x_n\| < \varepsilon$ for all $m,n > N$ | 3.3.3 (Box 3.1) |
| Contraction property of BOE | $\|f(v_1) - f(v_2)\|_\infty \leq \gamma\|v_1 - v_2\|_\infty$ | 3.3.4 |
| Maximum norm | $\|\cdot\|_\infty$ = max absolute value of elements | 3.3.4 |
| Value iteration (preview) | $v_{k+1} = f(v_k)$ | 3.4 |
| Greedy policy | $\pi^*(a|s) = 1$ if $a = \arg\max_a q^*(s,a)$, else 0 | 3.4 |
| Affine transformation of rewards | $r \to \alpha r + \beta$, $\alpha > 0$ | 3.5 |
| Optimal policy invariance | $\pi^*$ unchanged under affine reward transform | 3.5 |
| Transformed optimal state value | $v' = \alpha v^* + \frac{\beta}{1-\gamma}\mathbf{1}$ | 3.5 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| Bellman equation (from Ch. 2) | Used throughout Ch. 3 as the basis for the BOE |
| Optimal state value $v^*$ | Ch. 4 (value iteration computes $v^*$); Ch. 5 (Monte Carlo estimation); Ch. 7 (TD learning); Ch. 8 (function approximation of $v^*$) |
| Optimal policy $\pi^*$ | Ch. 4 (policy iteration); Ch. 9 (policy gradient methods); Ch. 10 (actor-critic) |
| BOE and contraction mapping | Ch. 4 (value iteration is exactly the iterative algorithm from Theorem 3.3) |
| Greedy policy (Theorem 3.5) | Ch. 4 (policy improvement step in policy iteration) |
| Affine invariance of rewards (Theorem 3.6) | Validates reward design choices throughout the book |
| Discount rate's role in preventing detours | Ch. 4 and onward (justifies $r_{\text{other}} = 0$ design) |
| Tabular optimality definition | Ch. 8-9 (replaced by function-approximation-based optimality metrics) |
| Contraction mapping theorem | Ch. 4 (convergence of value iteration); Ch. 6 (stochastic approximation convergence); Ch. 7 (TD convergence) |
| Action value $q^*(s,a)$ | Ch. 7 (Q-learning targets $q^*$); Ch. 8 (DQN approximates $q^*$) |
