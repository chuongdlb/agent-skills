---
chapter: 6
title: Stochastic Approximation
key_topics: [mean estimation, incremental algorithm, non-incremental algorithm, Robbins-Monro algorithm, stochastic approximation, root finding, convergence conditions, step size conditions, Dvoretzky theorem, quasimartingale, stochastic gradient descent, batch gradient descent, mini-batch gradient descent, convergence pattern, relative error]
depends_on: [1, 2, 3, 4, 5]
required_by: [7, 8, 9, 10]
---

# Chapter 6: Stochastic Approximation

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 6, pp. 101-124
> Supplemented by: Lecture slides L6 (61 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter does **not** introduce any specific reinforcement learning algorithms. Instead, it lays the **mathematical foundation** required to understand the temporal-difference (TD) algorithms of Chapter 7. Specifically, it fills a critical **knowledge gap**: all algorithms studied so far (Chapters 4-5) are **non-incremental**, but all algorithms from Chapter 7 onward are **incremental** (stochastic iterative).

**Why the gap matters**: Many readers encountering TD algorithms for the first time wonder how they were designed and why they work. The answer is that TD algorithms are **special cases of stochastic approximation algorithms**. This chapter introduces:
1. The **mean estimation** problem as a motivating example (non-incremental to incremental)
2. The **Robbins-Monro (RM) algorithm** -- a foundational stochastic approximation method
3. **Dvoretzky's theorem** -- a convergence tool for stochastic iterative algorithms
4. **Stochastic gradient descent (SGD)** -- a special case of the RM algorithm

**Key hierarchy** (from slides): Mean estimation is a special SGD algorithm; SGD is a special RM algorithm.

**Position in the book**: Chapter 6 is the last chapter in Part 1 (Fundamental Tools). It bridges the model-based methods of Chapters 2-4 and the model-free Monte Carlo methods of Chapter 5 to the incremental model-free methods of Chapters 7-10.

---

## 6.1 Motivating Example: Mean Estimation

### Problem Setup

Consider a random variable $X$ taking values from a finite set $\mathcal{X}$. The goal is to estimate $E[X]$. Given a sequence of i.i.d. samples $\{x_i\}_{i=1}^{n}$, the expected value can be approximated by:

$$E[X] \approx \bar{x} \doteq \frac{1}{n} \sum_{i=1}^{n} x_i$$

This is the basic idea of **Monte Carlo estimation** (Chapter 5). By the law of large numbers, $\bar{x} \to E[X]$ as $n \to \infty$.

**Why mean estimation matters for RL**: Many quantities in RL -- such as state values, action values, and policy gradients -- are defined as **expectations**. Hence mean estimation is a recurring fundamental operation.

### Two Methods to Calculate $\bar{x}$

**Method 1: Non-incremental (batch)**
- Collect **all** $n$ samples first, then compute the average.
- **Drawback**: If samples arrive one by one over time, we must wait until all are collected.

**Method 2: Incremental (iterative)**
- Update the estimate each time a new sample arrives.

### Deriving the Incremental Algorithm

Define:
$$w_{k+1} \doteq \frac{1}{k} \sum_{i=1}^{k} x_i, \quad k = 1, 2, \ldots$$

and hence:
$$w_k = \frac{1}{k-1} \sum_{i=1}^{k-1} x_i, \quad k = 2, 3, \ldots$$

Then $w_{k+1}$ can be expressed in terms of $w_k$:

$$w_{k+1} = \frac{1}{k} \sum_{i=1}^{k} x_i = \frac{1}{k}\left(\sum_{i=1}^{k-1} x_i + x_k\right) = \frac{1}{k}\left((k-1)w_k + x_k\right) = w_k - \frac{1}{k}(w_k - x_k)$$

This yields the **incremental mean estimation algorithm**:

$$\boxed{w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)}$$

### Verification (Worked Example with Actual Steps)

| Step | Computation | Result |
|---|---|---|
| $w_1$ | $= x_1$ | $x_1$ |
| $w_2$ | $= w_1 - \frac{1}{1}(w_1 - x_1) = x_1$ | $x_1$ |
| $w_3$ | $= w_2 - \frac{1}{2}(w_2 - x_2) = x_1 - \frac{1}{2}(x_1 - x_2)$ | $\frac{1}{2}(x_1 + x_2)$ |
| $w_4$ | $= w_3 - \frac{1}{3}(w_3 - x_3)$ | $\frac{1}{3}(x_1 + x_2 + x_3)$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $w_{k+1}$ | | $\frac{1}{k}\sum_{i=1}^{k} x_i$ |

### Advantages of the Incremental Algorithm

1. A mean estimate is available **immediately** after each new sample.
2. Early estimates are inaccurate (insufficient samples), but "better than nothing."
3. As more samples arrive, accuracy improves gradually ($w_k \to E[X]$ as $k \to \infty$).

### Generalization: Replacing $1/k$ with $\alpha_k$

Consider the more general algorithm:

$$\boxed{w_{k+1} = w_k - \alpha_k(w_k - x_k)}$$

where $\alpha_k > 0$ replaces $1/k$. This is equation (6.4) in the text.

- When $\alpha_k = 1/k$, we recover the explicit formula $w_{k+1} = \frac{1}{k}\sum_{i=1}^{k} x_i$.
- For general $\alpha_k$, no closed-form exists, but convergence $w_k \to E[X]$ is guaranteed under mild conditions on $\{\alpha_k\}$ (proven via the Robbins-Monro theorem in Section 6.2).

**Key insight (from slides)**: The convergence property does **not** depend on any assumption about the distribution of $X$.

**Forward reference**: TD algorithms in Chapter 7 have similar (but more complex) expressions of the same form.

---

## 6.2 Robbins-Monro Algorithm

### Definition of Stochastic Approximation

**Stochastic approximation** refers to a broad class of stochastic iterative algorithms for solving **root-finding** or **optimization** problems. Compared to many other root-finding algorithms (e.g., gradient-based methods), stochastic approximation is powerful because it does **not** require the expression of the objective function or its derivative.

### Problem Statement

Find the root of the equation:
$$g(w) = 0$$
where $w \in \mathbb{R}$ is the unknown variable and $g : \mathbb{R} \to \mathbb{R}$ is a function.

**Connection to optimization**: If $J(w)$ is an objective function to be minimized, this optimization problem can be converted to:
$$g(w) \doteq \nabla_w J(w) = 0$$

**Note**: An equation $g(w) = c$ (with $c$ a constant) can also be converted by rewriting $g(w) - c$ as a new function.

### The Black-Box Setup

The expression of $g$ (or its derivative) is **unknown**. We can only obtain a **noisy observation**:

$$\tilde{g}(w, \eta) = g(w) + \eta$$

where $\eta \in \mathbb{R}$ is the observation error (not necessarily Gaussian).

**Intuition**: It is a black-box system where only the input $w$ and the noisy output $\tilde{g}(w, \eta)$ are known. The goal is to solve $g(w) = 0$ using only $w$ and $\tilde{g}$.

**Philosophy (from slides)**: Without a model (the expression of $g$), we need data (input-output pairs).

### The RM Algorithm

$$\boxed{w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k), \quad k = 1, 2, 3, \ldots}$$

where:
- $w_k$ is the $k$th estimate of the root
- $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$ is the $k$th noisy observation
- $a_k > 0$ is a positive coefficient (step size / learning rate)

The algorithm requires **no information** about the function $g$. It only requires the input and output.

### Illustrative Example 1: $g(w) = w^3 - 5$

- True root: $5^{1/3} \approx 1.71$
- Observation: $\tilde{g}(w) = g(w) + \eta$, where $\eta \sim \mathcal{N}(0, 1)$ (i.i.d.)
- Parameters: $w_1 = 0$, $a_k = 1/k$
- **Result**: Despite noise corruption, $w_k$ converges to the true root.
- **Note**: The initial guess $w_1$ must be properly selected for $g(w) = w^3 - 5$ since it does not satisfy condition (a) of the RM theorem (its gradient is unbounded).

### Illustrative Example 2 (from slides): $g(w) = w - 10$

Manual computation with $w_1 = 20$, $a_k \equiv 0.5$, $\eta_k = 0$:

| Step | Value | $g(w_k)$ | Computation |
|---|---|---|---|
| $w_1$ | $20$ | $10$ | -- |
| $w_2$ | $15$ | $5$ | $20 - 0.5 \times 10$ |
| $w_3$ | $12.5$ | $2.5$ | $15 - 0.5 \times 5$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $w_k \to$ | $10$ | $0$ | converges to root |

### Convergence Intuition (Noise-Free Case)

Consider $g(w) = \tanh(w - 1)$ with true root $w^* = 1$. Setting $w_1 = 3$, $a_k = 1/k$, $\eta_k \equiv 0$:

The RM algorithm becomes $w_{k+1} = w_k - a_k g(w_k)$.

**Why it converges** (two cases):
1. **When $w_k > w^*$**: $g(w_k) > 0$ (since $g$ is increasing). Then $w_{k+1} = w_k - a_k g(w_k) < w_k$. If $a_k g(w_k)$ is sufficiently small: $w^* < w_{k+1} < w_k$. So $w_{k+1}$ is closer to $w^*$.
2. **When $w_k < w^*$**: $g(w_k) < 0$. Then $w_{k+1} = w_k - a_k g(w_k) > w_k$. If $|a_k g(w_k)|$ is sufficiently small: $w^* > w_{k+1} > w_k$. So $w_{k+1}$ is closer to $w^*$.

In either case, $w_{k+1}$ moves toward $w^*$. Hence $w_k$ converges to $w^*$.

---

### 6.2.1 Convergence Properties: The Robbins-Monro Theorem

**Theorem 6.1 (Robbins-Monro Theorem).** In the Robbins-Monro algorithm $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$, if:

**(a)** $0 < c_1 \le \nabla_w g(w) \le c_2$ for all $w$;

**(b)** $\sum_{k=1}^{\infty} a_k = \infty$ and $\sum_{k=1}^{\infty} a_k^2 < \infty$;

**(c)** $E[\eta_k | \mathcal{H}_k] = 0$ and $E[\eta_k^2 | \mathcal{H}_k] < \infty$;

where $\mathcal{H}_k = \{w_k, w_{k-1}, \ldots\}$, then $w_k$ **almost surely converges** to the root $w^*$ satisfying $g(w^*) = 0$.

### Detailed Explanation of the Three Conditions

#### Condition (a): Monotonicity and bounded gradient

- $0 < c_1 \le \nabla_w g(w)$ means $g(w)$ is **monotonically increasing**. This ensures the root of $g(w) = 0$ **exists and is unique**.
- If $g(w)$ is monotonically decreasing, replace it with $-g(w)$.
- **Connection to optimization**: When $g(w) \doteq \nabla_w J(w) = 0$, the condition that $g(w)$ is monotonically increasing means $J(w)$ is **convex** -- a standard assumption in optimization.
- $\nabla_w g(w) \le c_2$ means the gradient is **bounded from above**. For example, $g(w) = \tanh(w-1)$ satisfies this, but $g(w) = w^3 - 5$ does not.

#### Condition (b): Step size conditions

This condition on $\{a_k\}$ is frequently encountered in RL algorithms. It has two parts:

**Part 1**: $\sum_{k=1}^{\infty} a_k^2 < \infty$ implies $a_k \to 0$ as $k \to \infty$.

**Why is $a_k \to 0$ important?** Since $w_{k+1} - w_k = -a_k \tilde{g}(w_k, \eta_k)$:
- If $\tilde{g}(w_k, \eta_k)$ is bounded and $a_k \to 0$, then $w_{k+1} - w_k \to 0$, meaning the iterates stabilize.
- When $w_k \to w^*$, we have $g(w_k) \to 0$, and $\tilde{g}(w_k, \eta_k)$ is dominated by noise $\eta_k$. The shrinking $a_k$ suppresses this noise.
- If $a_k$ does not converge to zero, $w_k$ may keep fluctuating.

**Part 2**: $\sum_{k=1}^{\infty} a_k = \infty$ means $a_k$ should **not converge to zero too fast**.

**Why?** Summing $w_2 - w_1 = -a_1 \tilde{g}(w_1, \eta_1)$, $w_3 - w_2 = -a_2 \tilde{g}(w_2, \eta_2)$, $\ldots$ gives:
$$w_1 - w_\infty = \sum_{k=1}^{\infty} a_k \tilde{g}(w_k, \eta_k)$$

If $\sum_{k=1}^{\infty} a_k < \infty$, then $\left|\sum_{k=1}^{\infty} a_k \tilde{g}(w_k, \eta_k)\right|$ is bounded by some finite $b$. Then:
$$|w_1 - w_\infty| \le b$$

If the initial guess $w_1$ is selected far from $w^*$ such that $|w_1 - w^*| > b$, then $w_\infty = w^*$ is **impossible**. Thus $\sum_{k=1}^{\infty} a_k = \infty$ is necessary for convergence from **arbitrary** initial guesses.

#### Condition (c): Noise conditions

- $E[\eta_k | \mathcal{H}_k] = 0$: zero-mean noise conditioned on history (martingale difference condition).
- $E[\eta_k^2 | \mathcal{H}_k] < \infty$: bounded conditional variance.
- This condition is **mild**. It does not require $\eta_k$ to be Gaussian.
- **Important special case**: If $\{\eta_k\}$ is i.i.d. with $E[\eta_k] = 0$ and $E[\eta_k^2] < \infty$, then condition (c) holds because $\eta_k$ is independent of $\mathcal{H}_k$ and hence $E[\eta_k | \mathcal{H}_k] = E[\eta_k] = 0$.

### What Sequences Satisfy Condition (b)?

A typical sequence is $a_k = 1/k$.

**Verification of $\sum_{k=1}^{\infty} 1/k = \infty$**:

$$\lim_{n \to \infty} \left(\sum_{k=1}^{n} \frac{1}{k} - \ln n\right) = \kappa$$

where $\kappa \approx 0.577$ is the **Euler-Mascheroni constant**. Since $\ln n \to \infty$, we have $\sum_{k=1}^{\infty} 1/k = \infty$. (The partial sum $H_n = \sum_{k=1}^{n} 1/k$ is called the **harmonic number**.)

**Verification of $\sum_{k=1}^{\infty} 1/k^2 < \infty$**:

$$\sum_{k=1}^{\infty} \frac{1}{k^2} = \frac{\pi^2}{6} < \infty$$

This is known as the **Basel problem**.

**Slight modifications** such as $a_k = 1/(k+1)$ or $a_k = c_k / k$ (where $c_k$ is bounded) also satisfy condition (b).

**Practical note**: In many applications, $a_k$ is selected as a **sufficiently small constant**. Although condition (b) is not satisfied ($\sum a_k^2 = \infty$), the algorithm can still converge in a certain sense (see [24, Section 1.5]).

---

### 6.2.2 Application to Mean Estimation

The mean estimation algorithm $w_{k+1} = w_k + \alpha_k(x_k - w_k) = w_k - \alpha_k(w_k - x_k)$ is a **special RM algorithm**.

**Formulation as root-finding**: Define:
$$g(w) \doteq w - E[X]$$

The original problem (finding $E[X]$) becomes the root-finding problem $g(w) = 0$.

**Noisy observation**: Given a value of $w$, the noisy observation we can obtain is:
$$\tilde{g} \doteq w - x$$
where $x$ is a sample of $X$. This can be decomposed as:
$$\tilde{g}(w, \eta) = w - x = (w - E[X]) + (E[X] - x) \doteq g(w) + \eta$$
where $\eta \doteq E[X] - x$.

**The RM algorithm** for solving this problem is:
$$w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) = w_k - \alpha_k(w_k - x_k)$$

which is **exactly** the mean estimation algorithm.

**Convergence guarantee**: By Theorem 6.1, $w_k \to E[X]$ almost surely if:
- $\sum_{k=1}^{\infty} \alpha_k = \infty$ and $\sum_{k=1}^{\infty} \alpha_k^2 < \infty$
- $\{x_k\}$ is i.i.d.

The convergence does **not** rely on any assumption about the distribution of $X$.

---

## 6.3 Dvoretzky's Convergence Theorem

> **Note**: This section is mathematically intensive. It is recommended for readers interested in convergence analyses of stochastic algorithms. Otherwise, it can be skipped.

Dvoretzky's theorem is a classic result in stochastic approximation that can be used to analyze the convergence of both the RM algorithm and many RL algorithms.

### Theorem Statement

**Theorem 6.2 (Dvoretzky's Theorem).** Consider a stochastic process:

$$\boxed{\Delta_{k+1} = (1 - \alpha_k)\Delta_k + \beta_k \eta_k}$$

where $\{\alpha_k\}_{k=1}^{\infty}$, $\{\beta_k\}_{k=1}^{\infty}$, $\{\eta_k\}_{k=1}^{\infty}$ are stochastic sequences with $\alpha_k \ge 0$, $\beta_k \ge 0$ for all $k$. Then $\Delta_k$ converges to zero **almost surely** if:

**(a)** $\sum_{k=1}^{\infty} \alpha_k = \infty$, $\sum_{k=1}^{\infty} \alpha_k^2 < \infty$, and $\sum_{k=1}^{\infty} \beta_k^2 < \infty$ uniformly almost surely;

**(b)** $E[\eta_k | \mathcal{H}_k] = 0$ and $E[\eta_k^2 | \mathcal{H}_k] \le C$ almost surely;

where $\mathcal{H}_k = \{\Delta_k, \Delta_{k-1}, \ldots, \eta_{k-1}, \ldots, \alpha_{k-1}, \ldots, \beta_{k-1}, \ldots\}$.

### Important Clarifications

1. **Random coefficients**: In the RM algorithm, $\{a_k\}$ is deterministic. Dvoretzky's theorem allows $\{\alpha_k\}$, $\{\beta_k\}$ to be **random variables** depending on $\mathcal{H}_k$. This is useful when $\alpha_k$ or $\beta_k$ is a function of $\Delta_k$.

2. **"Uniformly almost surely"**: Since $\alpha_k$ and $\beta_k$ may be random variables, limits must be defined in the stochastic sense. Similarly, $E[\eta_k | \mathcal{H}_k]$ and $E[\eta_k^2 | \mathcal{H}_k]$ are random variables (since $\mathcal{H}_k$ contains random variables), so the equalities/inequalities hold "almost surely."

3. **Compared to [32]**: Theorem 6.2 does **not** require $\sum_{k=1}^{\infty} \beta_k = \infty$. When $\sum_{k=1}^{\infty} \beta_k < \infty$ (especially $\beta_k = 0$ for all $k$), the sequence can still converge.

### 6.3.1 Proof of Dvoretzky's Theorem (via Quasimartingales)

The proof uses the **quasimartingale convergence theorem** (Appendix C).

**Proof.** Let $h_k \doteq \Delta_k^2$. Then:

$$h_{k+1} - h_k = \Delta_{k+1}^2 - \Delta_k^2 = (\Delta_{k+1} - \Delta_k)(\Delta_{k+1} + \Delta_k)$$

Substituting $\Delta_{k+1} - \Delta_k = -\alpha_k \Delta_k + \beta_k \eta_k$:

$$h_{k+1} - h_k = (-\alpha_k \Delta_k + \beta_k \eta_k)[(2 - \alpha_k)\Delta_k + \beta_k \eta_k]$$
$$= -\alpha_k(2 - \alpha_k)\Delta_k^2 + \beta_k^2 \eta_k^2 + 2(1 - \alpha_k)\beta_k \eta_k \Delta_k$$

Taking conditional expectations:

$$E[h_{k+1} - h_k | \mathcal{H}_k] = -\alpha_k(2 - \alpha_k)\Delta_k^2 + \beta_k^2 E[\eta_k^2 | \mathcal{H}_k] + 2(1 - \alpha_k)\beta_k \Delta_k E[\eta_k | \mathcal{H}_k]$$

(Here $\Delta_k$, $\alpha_k$, $\beta_k$ can be taken outside the expectation since they are determined by $\mathcal{H}_k$.)

Since $E[\eta_k | \mathcal{H}_k] = 0$, the third term vanishes. Since $\sum \alpha_k^2 < \infty$ implies $\alpha_k \to 0$, for sufficiently large $k$ we have $\alpha_k \le 1$, so $-\alpha_k(2 - \alpha_k)\Delta_k^2 \le 0$. Thus:

$$E[h_{k+1} - h_k | \mathcal{H}_k] \le \beta_k^2 C$$

and hence:
$$\sum_{k=1}^{\infty} E[h_{k+1} - h_k | \mathcal{H}_k] \le \sum_{k=1}^{\infty} \beta_k^2 C < \infty$$

By the **quasimartingale convergence theorem**, $h_k$ converges almost surely.

**Determining the limit**: From the full equation:

$$\sum_{k=1}^{\infty} \alpha_k(2 - \alpha_k)\Delta_k^2 = \sum_{k=1}^{\infty} \beta_k^2 E[\eta_k^2 | \mathcal{H}_k] - \sum_{k=1}^{\infty} E[h_{k+1} - h_k | \mathcal{H}_k]$$

Both terms on the right are bounded. Since $\alpha_k \le 1$:

$$\infty > \sum_{k=1}^{\infty} \alpha_k(2 - \alpha_k)\Delta_k^2 \ge \sum_{k=1}^{\infty} \alpha_k \Delta_k^2 \ge 0$$

Since $\sum_{k=1}^{\infty} \alpha_k = \infty$ but $\sum_{k=1}^{\infty} \alpha_k \Delta_k^2 < \infty$, we must have $\Delta_k \to 0$ almost surely. $\blacksquare$

---

### 6.3.2 Application to Mean Estimation (via Dvoretzky)

The mean estimation algorithm $w_{k+1} = w_k + \alpha_k(x_k - w_k)$ can be analyzed directly using Dvoretzky's theorem.

**Proof.** Let $w^* = E[X]$ and $\Delta_k \doteq w_k - w^*$. Then:

$$w_{k+1} - w^* = w_k - w^* + \alpha_k(x_k - w^* + w^* - w_k)$$

$$\Delta_{k+1} = \Delta_k + \alpha_k(x_k - w^* - \Delta_k) = (1 - \alpha_k)\Delta_k + \alpha_k \underbrace{(x_k - w^*)}_{\eta_k}$$

This matches the Dvoretzky form with $\beta_k = \alpha_k$.

**Checking conditions**:
- Since $\{x_k\}$ is i.i.d.: $E[x_k | \mathcal{H}_k] = E[x_k] = w^*$
- $E[\eta_k | \mathcal{H}_k] = E[x_k - w^* | \mathcal{H}_k] = 0$
- $E[\eta_k^2 | \mathcal{H}_k] = E[x_k^2 | \mathcal{H}_k] - (w^*)^2 = E[x_k^2] - (w^*)^2$ is bounded if the variance of $x_k$ is finite.

All conditions of Dvoretzky's theorem are satisfied. Therefore $\Delta_k \to 0$, i.e., $w_k \to w^* = E[X]$ almost surely. $\blacksquare$

---

### 6.3.3 Application to the Robbins-Monro Theorem (via Dvoretzky)

Dvoretzky's theorem provides a clean proof of the RM theorem.

**Proof.** Let $w^*$ satisfy $g(w^*) = 0$ and let $\Delta_k \doteq w_k - w^*$. The RM algorithm gives:

$$w_{k+1} - w^* = w_k - w^* - a_k[g(w_k) - g(w^*) + \eta_k]$$

By the **mean value theorem**: $g(w_k) - g(w^*) = \nabla_w g(w'_k)(w_k - w^*)$ where $w'_k \in [w_k, w^*]$.

$$\Delta_{k+1} = \Delta_k - a_k[\nabla_w g(w'_k) \Delta_k + \eta_k] = \underbrace{[1 - a_k \nabla_w g(w'_k)]}_{\text{plays role of } (1 - \alpha_k)} \Delta_k + a_k(-\eta_k)$$

**Checking conditions**: Since $0 < c_1 \le \nabla_w g(w) \le c_2$ and $\sum a_k = \infty$, $\sum a_k^2 < \infty$:
- Let $\alpha_k = a_k \nabla_w g(w'_k)$. Then $\sum \alpha_k = \infty$ and $\sum \alpha_k^2 < \infty$.
- All conditions of Dvoretzky's theorem are satisfied.

Therefore $\Delta_k \to 0$ almost surely. $\blacksquare$

**Key insight**: The power of Dvoretzky's theorem is demonstrated here. In particular, $\alpha_k = a_k \nabla_w g(w'_k)$ is a **stochastic sequence** (depending on $w_k$), not deterministic. Dvoretzky's theorem handles this naturally.

---

### 6.3.4 An Extension of Dvoretzky's Theorem (Multi-Variable)

This extension handles **multiple variables** and is used to analyze the convergence of algorithms like **Q-learning**.

**Theorem 6.3.** Consider a finite set $\mathcal{S}$ of real numbers. For the stochastic process:

$$\Delta_{k+1}(s) = (1 - \alpha_k(s))\Delta_k(s) + \beta_k(s)\eta_k(s)$$

$\Delta_k(s)$ converges to zero almost surely for every $s \in \mathcal{S}$ if the following conditions hold for all $s \in \mathcal{S}$:

**(a)** $\sum_k \alpha_k(s) = \infty$, $\sum_k \alpha_k^2(s) < \infty$, $\sum_k \beta_k^2(s) < \infty$, and $E[\beta_k(s) | \mathcal{H}_k] \le E[\alpha_k(s) | \mathcal{H}_k]$ uniformly almost surely;

**(b)** $\|E[\eta_k(s) | \mathcal{H}_k]\|_\infty \le \gamma \|\Delta_k\|_\infty$, where $\gamma \in (0, 1)$;

**(c)** $\text{var}[\eta_k(s) | \mathcal{H}_k] \le C(1 + \|\Delta_k(s)\|_\infty)^2$, where $C$ is a constant.

Here $\mathcal{H}_k = \{\Delta_k, \Delta_{k-1}, \ldots, \eta_{k-1}, \ldots, \alpha_{k-1}, \ldots, \beta_{k-1}, \ldots\}$ and $\|\cdot\|_\infty$ is the **maximum norm** over the set $\mathcal{S}$:
$$\|E[\eta_k(s) | \mathcal{H}_k]\|_\infty \doteq \max_{s \in \mathcal{S}} |E[\eta_k(s) | \mathcal{H}_k]|, \quad \|\Delta_k(s)\|_\infty \doteq \max_{s \in \mathcal{S}} |\Delta_k(s)|$$

### Key Remarks on Theorem 6.3

1. **Notation**: The variable $s$ is an **index**. In RL, it represents a **state** or **state-action pair**.

2. **Generality over Dvoretzky's theorem**:
   - Handles **multiple variables** via the maximum norm operations (important for RL with multiple states).
   - Relaxes the noise conditions: Dvoretzky requires $E[\eta_k | \mathcal{H}_k] = 0$ and $\text{var}[\eta_k | \mathcal{H}_k] \le C$. This theorem only requires the expectation and variance to be **bounded by the error** $\Delta_k$.

3. **For RL convergence proofs**: When applying this theorem, we must show conditions hold for **every** state (or state-action pair) $s \in \mathcal{S}$.

**Forward reference**: This theorem is used to prove convergence of **Q-learning** and **TD learning** algorithms.

---

## 6.4 Stochastic Gradient Descent

SGD is widely used in machine learning and RL. The key relationships are:
- **SGD is a special RM algorithm**
- **The mean estimation algorithm is a special SGD algorithm**

### Problem Setup

$$\min_w J(w) = E[f(w, X)]$$

where:
- $w$ is the parameter to be optimized
- $X$ is a random variable (the expectation is with respect to $X$)
- $w$ and $X$ can be either scalars or vectors
- $f(\cdot)$ is a scalar-valued function

### Three Methods for Solving this Problem

#### Method 1: Gradient Descent (GD)

$$w_{k+1} = w_k - \alpha_k \nabla_w J(w_k) = w_k - \alpha_k E[\nabla_w f(w_k, X)]$$

- Uses the **true gradient** $E[\nabla_w f(w_k, X)]$.
- **Drawback**: Requires the probability distribution of $X$ (or the true expectation), which is often unknown.
- Can find optimal $w^*$ under mild conditions (e.g., convexity of $f$).

#### Method 2: Batch Gradient Descent (BGD)

Approximate the expected value using **all** $n$ i.i.d. samples $\{x_i\}_{i=1}^{n}$:

$$E[\nabla_w f(w_k, X)] \approx \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i)$$

$$\boxed{w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i)} \quad \text{(BGD)}$$

- **Drawback**: Requires **all** samples in **every** iteration. If samples are collected one by one, this is impractical.

#### Method 3: Stochastic Gradient Descent (SGD)

$$\boxed{w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)} \quad \text{(SGD)}$$

where $x_k$ is the sample collected at time step $k$.

- Called "stochastic" because it relies on stochastic samples $\{x_k\}$.
- Replaces the **true gradient** $E[\nabla_w f(w, X)]$ with the **stochastic gradient** $\nabla_w f(w_k, x_k)$.
- Uses a **single sample** per iteration (cf. BGD uses all $n$).

### Why SGD Works: Intuitive Explanation

Since $\nabla_w f(w_k, x_k) \ne E[\nabla_w f(w, X)]$, can such a replacement still ensure $w_k \to w^*$?

Decompose the stochastic gradient:
$$\nabla_w f(w_k, x_k) = E[\nabla_w f(w_k, X)] + \underbrace{\left(\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]\right)}_{\eta_k}$$

The SGD algorithm can be rewritten as:
$$w_{k+1} = w_k - \alpha_k E[\nabla_w f(w_k, X)] - \alpha_k \eta_k$$

This is the **regular gradient descent** plus a **perturbation** $\alpha_k \eta_k$.

Since $\{x_k\}$ is i.i.d., $E_{x_k}[\nabla_w f(w_k, x_k)] = E_X[\nabla_w f(w_k, X)]$. Therefore:
$$E[\eta_k] = E[\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]] = 0$$

The perturbation has **zero mean**, which intuitively suggests it does not jeopardize convergence.

---

### 6.4.1 Application to Mean Estimation

The mean estimation algorithm is a **special SGD algorithm**.

**Formulation as optimization**: Define:
$$\min_w J(w) = E\left[\frac{1}{2}\|w - X\|^2\right] \doteq E[f(w, X)]$$

where $f(w, X) = \|w - X\|^2 / 2$ and $\nabla_w f(w, X) = w - X$.

**Optimal solution**: Solving $\nabla_w J(w) = E[w - X] = w - E[X] = 0$ gives $w^* = E[X]$.

**Three algorithms for this problem** (worked example from text):

| Algorithm | Update Rule | Simplified Form |
|---|---|---|
| GD | $w_{k+1} = w_k - \alpha_k E[w_k - X]$ | $w_{k+1} = w_k - \alpha_k(w_k - E[X])$ |
| SGD | $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ | $w_{k+1} = w_k - \alpha_k(w_k - x_k)$ |

- **GD is inapplicable** since $E[X]$ is unknown (it is what we want to find).
- **SGD gives exactly the mean estimation algorithm** from equation (6.4).

**Conclusion**: The mean estimation algorithm is a special SGD algorithm designed for solving $\min_w E[\frac{1}{2}\|w - X\|^2]$.

---

### 6.4.2 Convergence Pattern of SGD

**Question**: Since the stochastic gradient is random, is the convergence of SGD slow or random?

**Answer**: SGD has an interesting convergence pattern:
- When $w_k$ is **far** from $w^*$: SGD behaves like regular GD, and convergence is **fast**.
- When $w_k$ is **close** to $w^*$: the randomness of the stochastic gradient becomes influential, and convergence exhibits more **randomness**.

#### Analysis: Relative Error

The relative error between stochastic and true gradients (scalar case):

$$\delta_k \doteq \frac{|\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]|}{|E[\nabla_w f(w_k, X)]|}$$

Since $E[\nabla_w f(w^*, X)] = 0$ at the optimum, using the **mean value theorem**:

$$\delta_k = \frac{|\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]|}{|E[\nabla_w^2 f(\tilde{w}_k, X)(w_k - w^*)]|}$$

where $\tilde{w}_k \in [w_k, w^*]$.

For **strictly convex** $f$ with $\nabla_w^2 f \ge c > 0$:

$$\delta_k \le \frac{\left|\overbrace{\nabla_w f(w_k, x_k)}^{\text{stochastic gradient}} - \overbrace{E[\nabla_w f(w_k, X)]}^{\text{true gradient}}\right|}{c \underbrace{|w_k - w^*|}_{\text{distance to optimum}}}$$

**Key insight**: $\delta_k$ is **inversely proportional** to $|w_k - w^*|$.
- Large $|w_k - w^*|$ $\Rightarrow$ small $\delta_k$ $\Rightarrow$ SGD $\approx$ GD $\Rightarrow$ fast convergence.
- Small $|w_k - w^*|$ $\Rightarrow$ large $\delta_k$ $\Rightarrow$ more randomness.

#### Worked Example: Mean Estimation Convergence Pattern

For $f(w, X) = |w - X|^2/2$ (scalar case):
$$\nabla_w f(w, x_k) = w - x_k, \quad E[\nabla_w f(w, X)] = w - E[X] = w - w^*$$

The relative error becomes:
$$\delta_k = \frac{|(w_k - x_k) - (w_k - E[X])|}{|w_k - w^*|} = \frac{|E[X] - x_k|}{|w_k - w^*|}$$

This clearly shows $\delta_k$ is inversely proportional to $|w_k - w^*|$. Additionally, $\delta_k$ is proportional to the variance of $X$ (through $|E[X] - x_k|$).

#### Simulation Example

**Setup**: $X \in \mathbb{R}^2$ uniform in a square centered at the origin with side length 20. True mean $E[X] = 0$. Mean estimation based on 100 i.i.d. samples. $\alpha_k = 1/k$.

**Results** (Figure 6.5):
- All algorithms (SGD with $m=1$, MBGD with $m=5$, MBGD with $m=50$) converge to the mean.
- SGD ($m=1$): Fast initial approach, then exhibits randomness near the origin.
- MBGD ($m=50$): Smoothest and fastest convergence.
- MBGD ($m=5$): Intermediate between SGD and $m=50$.

---

### 6.4.3 A Deterministic Formulation of SGD

One may encounter a **deterministic formulation** of SGD without random variables.

**Setup**: Given a set of real numbers $\{x_i\}_{i=1}^{n}$ (not necessarily samples of any random variable), minimize:

$$\min_w J(w) = \frac{1}{n} \sum_{i=1}^{n} f(w, x_i)$$

The GD algorithm:
$$w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i)$$

If the set is large and we can only fetch one number at a time, the incremental update is:
$$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$$

where $x_k$ is the number fetched at step $k$ (not necessarily the $k$th element of $\{x_i\}$).

**Converting to stochastic formulation**: Let $X$ be a random variable on $\{x_i\}_{i=1}^{n}$ with uniform distribution $p(X = x_i) = 1/n$. Then:
$$\min_w J(w) = \frac{1}{n} \sum_{i=1}^{n} f(w, x_i) = E[f(w, X)]$$

The last equality is **exact** (not approximate). Therefore:
- The algorithm is SGD.
- Convergence is guaranteed if $x_k$ is **uniformly and independently sampled** from $\{x_i\}_{i=1}^{n}$.
- Note: $x_k$ may repeatedly take the same number since it is randomly sampled.

---

### 6.4.4 BGD, SGD, and Mini-Batch GD (MBGD)

Given samples $\{x_i\}_{i=1}^{n}$ of $X$, to minimize $J(w) = E[f(w, X)]$:

| Algorithm | Update Rule | Samples per Iteration |
|---|---|---|
| **BGD** | $w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i)$ | All $n$ samples |
| **MBGD** | $w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in I_k} \nabla_w f(w_k, x_j)$ | $m$ samples ($I_k \subset \{1,\ldots,n\}$, $|I_k| = m$) |
| **SGD** | $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ | 1 sample |

**Comparison**:

| Property | BGD | MBGD | SGD |
|---|---|---|---|
| Samples per step | $n$ (all) | $m$ (subset) | $1$ |
| Gradient quality | Close to true gradient | Between BGD and SGD | Single-sample approximation |
| Randomness | Low | Medium | High |
| Flexibility | Low (needs all data) | High | High |

**Key relationships**:
- If $m = 1$: MBGD becomes SGD.
- If $m = n$: MBGD does **not** strictly become BGD. MBGD uses $n$ **randomly fetched** samples (may repeat), whereas BGD uses **all** $n$ distinct numbers exactly once.
- MBGD convergence is **faster** than SGD because averaging over $m$ samples reduces randomness.

#### Worked Example: Mean Estimation with BGD, MBGD, SGD

Given $\{x_i\}_{i=1}^{n}$, goal: calculate $\bar{x} = \sum_{i=1}^{n} x_i / n$.

Equivalent optimization: $\min_w J(w) = \frac{1}{2n}\sum_{i=1}^{n}\|w - x_i\|^2$.

The three algorithms:

$$w_{k+1} = w_k - \alpha_k(w_k - \bar{x}) \quad \text{(BGD)}$$

$$w_{k+1} = w_k - \alpha_k\left(w_k - \bar{x}_k^{(m)}\right) \quad \text{(MBGD)}$$

$$w_{k+1} = w_k - \alpha_k(w_k - x_k) \quad \text{(SGD)}$$

where $\bar{x}_k^{(m)} = \sum_{j \in I_k} x_j / m$.

**With $\alpha_k = 1/k$**, these can be solved explicitly:

| Algorithm | Explicit Solution | Behavior |
|---|---|---|
| BGD | $w_{k+1} = \bar{x}$ for all $k$ | Exact solution at every step |
| MBGD | $w_{k+1} = \frac{1}{k}\sum_{j=1}^{k} \bar{x}_j^{(m)}$ | Averages of mini-batch means |
| SGD | $w_{k+1} = \frac{1}{k}\sum_{j=1}^{k} x_j$ | Running average of individual samples |

BGD gives the exact solution at each step. MBGD converges faster than SGD because $\bar{x}_k^{(m)}$ is already an average.

---

### 6.4.5 Convergence of SGD (Rigorous Proof)

**Theorem 6.4 (Convergence of SGD).** For the SGD algorithm $w_{k+1} = w_k - a_k \nabla_w f(w_k, x_k)$, if:

**(a)** $0 < c_1 \le \nabla_w^2 f(w, X) \le c_2$;

**(b)** $\sum_{k=1}^{\infty} a_k = \infty$ and $\sum_{k=1}^{\infty} a_k^2 < \infty$;

**(c)** $\{x_k\}_{k=1}^{\infty}$ are i.i.d.;

then $w_k$ converges to the root of $\nabla_w E[f(w, X)] = 0$ **almost surely**.

#### Explanation of Conditions

- **Condition (a)**: Requires $f$ to be **strictly convex** with curvature bounded above and below. When $w$ is a scalar, $\nabla_w^2 f$ is a scalar. When $w$ is a vector, $\nabla_w^2 f$ is the **Hessian matrix**.
- **Condition (b)**: Same as the RM algorithm. In practice, $a_k$ is often a small constant (condition (b) is violated, but the algorithm still converges in a certain sense).
- **Condition (c)**: Standard i.i.d. requirement.

#### Proof (SGD is a Special RM Algorithm)

The optimization problem $\min_w J(w) = E[f(w, X)]$ is converted to root-finding:
$$g(w) \doteq \nabla_w J(w) = E[\nabla_w f(w, X)] = 0$$

The measurable quantity is:
$$\tilde{g}(w, \eta) = \nabla_w f(w, x) = \underbrace{E[\nabla_w f(w, X)]}_{g(w)} + \underbrace{\nabla_w f(w, x) - E[\nabla_w f(w, X)]}_{\eta}$$

The RM algorithm for $g(w) = 0$:
$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k) = w_k - a_k \nabla_w f(w_k, x_k)$$

which is **exactly** the SGD algorithm. Verifying the three RM conditions:

1. $\nabla_w g(w) = \nabla_w E[\nabla_w f(w, X)] = E[\nabla_w^2 f(w, X)]$. From $c_1 \le \nabla_w^2 f \le c_2$, we get $c_1 \le \nabla_w g(w) \le c_2$. Condition (a) of RM is satisfied.

2. Condition (b) of RM is the same as condition (b) here.

3. For condition (c) of RM ($E[\eta_k | \mathcal{H}_k] = 0$): Since $\{x_k\}$ is i.i.d. and $x_k$ is independent of $\mathcal{H}_k = \{w_k, w_{k-1}, \ldots\}$:
   $$E[\eta_k | \mathcal{H}_k] = E_{x_k}[\nabla_w f(w_k, x_k)] - E[\nabla_w f(w_k, X)] = 0$$
   Similarly, $E[\eta_k^2 | \mathcal{H}_k] < \infty$ if $|\nabla_w f(w, x)| < \infty$ for all $w$ given any $x$.

Since all three RM conditions are satisfied, convergence follows from Theorem 6.1. $\blacksquare$

---

## 6.5 Summary

### Algorithm Hierarchy

| Level | Algorithm | Purpose | Update Rule |
|---|---|---|---|
| General | **RM algorithm** | Solve $g(w) = 0$ using noisy observations $\{\tilde{g}(w_k, \eta_k)\}$ | $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$ |
| Special case of RM | **SGD algorithm** | Minimize $J(w) = E[f(w,X)]$ using samples $\{x_k\}$ | $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ |
| Special case of SGD | **Mean estimation** | Compute $E[X]$ using samples $\{x_k\}$ | $w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)$ |

### Convergence Theorem Hierarchy

| Level | Theorem | Key Feature |
|---|---|---|
| Most general | **Dvoretzky (extended, Thm 6.3)** | Multi-variable, handles Q-learning |
| General | **Dvoretzky (Thm 6.2)** | Single variable, stochastic coefficients |
| Derived from Dvoretzky | **Robbins-Monro (Thm 6.1)** | Root-finding with noisy observations |
| Derived from RM | **SGD convergence (Thm 6.4)** | Optimization with stochastic gradients |

---

## 6.6 Q&A -- Important Clarifications

### Q: What is stochastic approximation?
**A**: Stochastic approximation refers to a broad class of stochastic iterative algorithms for solving root-finding or optimization problems. The name was first used by Robbins and Monro in 1951.

### Q: Why do we need to study stochastic approximation?
**A**: Because the temporal-difference RL algorithms (Chapter 7) can be viewed as stochastic approximation algorithms. This chapter provides the necessary preparation so that encountering TD algorithms is not abrupt.

### Q: Why do we frequently discuss mean estimation?
**A**: State and action values in RL are defined as means (expectations) of random variables. TD learning algorithms (Chapter 7) are similar to stochastic approximation algorithms for mean estimation.

### Q: What is the advantage of the RM algorithm?
**A**: Compared to many root-finding algorithms, the RM algorithm does **not** require the expression of the objective function or its derivative. It is a **black-box** technique requiring only input-output pairs.

### Q: What is the basic idea of SGD?
**A**: SGD solves optimization problems involving random variables by using **samples** instead of probability distributions. Mathematically, it replaces the true gradient (an expectation) with a stochastic gradient (computed from a single sample).

### Q: Can SGD converge quickly?
**A**: SGD has an interesting convergence pattern: fast convergence when far from the optimal solution (SGD behaves like GD); slower and more random near the solution (the relative error of the stochastic gradient increases).

### Q: What is MBGD? Advantages?
**A**: MBGD is an intermediate version between SGD and BGD. Compared to SGD, it has **less randomness** (uses $m > 1$ samples). Compared to BGD, it does **not** require all samples per iteration, making it more **flexible**.

---

## Concept Index

| Concept | Notation / Formula | Section |
|---|---|---|
| Non-incremental algorithm | Collect all samples, then compute | 6.1 |
| Incremental algorithm | Update upon each new sample | 6.1 |
| Mean estimation (batch) | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | 6.1 |
| Mean estimation (incremental) | $w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)$ | 6.1 |
| Generalized mean estimation | $w_{k+1} = w_k - \alpha_k(w_k - x_k)$ | 6.1 |
| Stochastic approximation | Broad class of stochastic iterative algorithms | 6.2 |
| Root-finding problem | $g(w) = 0$ | 6.2 |
| Noisy observation | $\tilde{g}(w, \eta) = g(w) + \eta$ | 6.2 |
| Robbins-Monro algorithm | $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$ | 6.2 |
| Step size / learning rate | $a_k$ or $\alpha_k$ | 6.2 |
| Step size conditions | $\sum a_k = \infty$, $\sum a_k^2 < \infty$ | 6.2 |
| Harmonic number | $H_n = \sum_{k=1}^{n} 1/k$ | 6.2 |
| Euler-Mascheroni constant | $\kappa \approx 0.577$ | 6.2 |
| Basel problem | $\sum_{k=1}^{\infty} 1/k^2 = \pi^2/6$ | 6.2 |
| Almost sure convergence | $w_k \to w^*$ w.p.1 | 6.2 |
| Dvoretzky's theorem | $\Delta_{k+1} = (1-\alpha_k)\Delta_k + \beta_k \eta_k \to 0$ | 6.3 |
| Quasimartingale | Used in proof of Dvoretzky's theorem | 6.3.1 |
| Extended Dvoretzky (multi-variable) | $\Delta_{k+1}(s) = (1-\alpha_k(s))\Delta_k(s) + \beta_k(s)\eta_k(s)$ | 6.3.4 |
| Maximum norm over $\mathcal{S}$ | $\|\cdot\|_\infty = \max_{s \in \mathcal{S}} |\cdot|$ | 6.3.4 |
| Gradient descent (GD) | $w_{k+1} = w_k - \alpha_k E[\nabla_w f(w_k, X)]$ | 6.4 |
| Batch gradient descent (BGD) | $w_{k+1} = w_k - \frac{\alpha_k}{n}\sum_{i=1}^{n}\nabla_w f(w_k, x_i)$ | 6.4 |
| Stochastic gradient descent (SGD) | $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$ | 6.4 |
| True gradient | $E[\nabla_w f(w_k, X)]$ | 6.4 |
| Stochastic gradient | $\nabla_w f(w_k, x_k)$ | 6.4 |
| Mini-batch gradient descent (MBGD) | $w_{k+1} = w_k - \frac{\alpha_k}{m}\sum_{j \in I_k}\nabla_w f(w_k, x_j)$ | 6.4.4 |
| Mini-batch size | $m = |I_k|$ | 6.4.4 |
| Relative error (SGD vs GD) | $\delta_k = \frac{|\nabla_w f(w_k,x_k) - E[\nabla_w f(w_k,X)]|}{|E[\nabla_w f(w_k,X)]|}$ | 6.4.2 |
| Convergence pattern of SGD | $\delta_k \le \frac{|\text{stoch. grad.} - \text{true grad.}|}{c|w_k - w^*|}$ | 6.4.2 |
| Hessian matrix | $\nabla_w^2 f(w, X)$ (vector $w$ case) | 6.4.5 |
| Strict convexity condition | $0 < c_1 \le \nabla_w^2 f(w, X) \le c_2$ | 6.4.5 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| Incremental algorithm structure $w_{k+1} = w_k - \alpha_k(\cdot)$ | Ch 7 (TD learning has similar expressions) |
| Robbins-Monro algorithm and theorem | Ch 7 (TD algorithms are special RM algorithms) |
| Step size conditions $\sum \alpha_k = \infty$, $\sum \alpha_k^2 < \infty$ | Ch 7, 8, 9, 10 (convergence conditions for RL algorithms) |
| Dvoretzky's theorem (basic, Thm 6.2) | Ch 7 (convergence proofs for TD methods) |
| Extended Dvoretzky theorem (Thm 6.3) | Ch 7, 8 (convergence of Q-learning, multi-state algorithms) |
| SGD algorithm | Ch 8 (value function approximation), Ch 9 (policy gradient methods), Ch 10 (actor-critic) |
| Mean estimation as special SGD | Ch 5 (Monte Carlo is non-incremental mean estimation), Ch 7 (TD is incremental) |
| Convergence pattern of SGD | Ch 8, 9 (understanding behavior of function approximation training) |
| BGD / MBGD / SGD comparison | Ch 9, 10 (practical algorithm design choices) |
| Non-incremental vs. incremental distinction | Ch 5 vs. Ch 7 (MC vs. TD -- the fundamental algorithmic shift) |
