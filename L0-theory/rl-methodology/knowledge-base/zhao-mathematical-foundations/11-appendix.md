---
chapter: appendix
title: Mathematical Appendix
key_topics: [probability theory, random variables, expectation, conditional expectation, variance, covariance, gradient of expectation, measure-theoretic probability, sigma-algebra, probability triples, convergence of sequences, martingales, supermartingales, submartingales, quasimartingales, convexity, gradient descent, Lipschitz continuity, step size selection]
depends_on: []
required_by: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
---

# Mathematical Appendix

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Appendices A-D, pp. 237-270
> Errata applied:
> - Appendix A, "Gradient of expectation": Corrected `f(x, a)` to `f(x, beta)` in expectation and gradient formulas
> - Appendix A, "Variance, covariance, covariance matrix": Corrected capital X-bar/Y-bar to lowercase x-bar/y-bar
> - Appendix D, "Convexity": Corrected `f(cx + (1-x)y)` to `f(cx + (1-c)y)` in convex function definition

## Purpose and Context

The appendix provides the mathematical foundations that underpin the entire book. It covers four areas:
- **Appendix A**: Preliminaries for probability theory (used throughout all chapters)
- **Appendix B**: Measure-theoretic (rigorous) probability theory (used for convergence analysis in Chapters 6-7)
- **Appendix C**: Convergence of deterministic and stochastic sequences (used for algorithm convergence proofs in Chapters 6-7)
- **Appendix D**: Preliminaries for gradient descent (foundation for stochastic gradient descent in Chapter 6 and policy gradient methods in Chapters 9-10)

---

## Appendix A: Preliminaries for Probability Theory

Reinforcement learning heavily relies on probability theory. This section summarizes concepts and results frequently used throughout the book.

### Random Variable

**Definition**: A random variable can take values from a set of numbers, where the specific value taken follows a probability distribution.

**Notation conventions**:
- Capital letter $X$ denotes a random variable
- Lowercase letter $x$ denotes a value that $X$ can take
- This book mainly considers the case where a random variable takes a **finite** number of values
- A random variable can be a **scalar** or a **vector**

**Operations**: Random variables support normal mathematical operations (summation, product, absolute value). For random variables $X, Y$: $X + Y$, $X + 1$, $XY$ are all well-defined.

### Stochastic Sequence

**Definition**: A stochastic sequence is a sequence of random variables.

A common scenario is collecting a stochastic sampling sequence $\{x_i\}_{i=1}^n$ of a random variable $X$. For example, tossing a die $n$ times yields the sequence $\{x_1, x_2, \ldots, x_n\}$, where each $x_i$ is a random variable representing the value of the $i$-th toss.

**Important note**: Although $x_i$ is a lowercase letter, it still represents a random variable since $x_i$ can take any value in $\{1, \ldots, 6\}$. A realized sequence like $\{1, 6, 3, 5, \ldots\}$ is **not** a stochastic sequence because all elements are already determined.

---

### Probability

The notation $p(X = x)$ or $p_X(x)$ describes the probability of $X$ taking the value $x$. When the context is clear, $p(X = x)$ is often written as $p(x)$.

### Joint Probability

The notation $p(X = x, Y = y)$ or $p(x, y)$ describes the probability of $X$ taking value $x$ **and** $Y$ taking value $y$.

**Marginalization identity**:
$$\sum_y p(x, y) = p(x)$$

### Conditional Probability

The notation $p(X = x | A = a)$ or $p(x|a)$ describes the probability of $X$ taking value $x$ given that $A$ has taken value $a$.

**Key identities**:
$$p(x, a) = p(x|a) \, p(a)$$
$$p(x|a) = \frac{p(x, a)}{p(a)}$$

### Law of Total Probability

Since $p(x) = \sum_a p(x, a)$:

$$p(x) = \sum_a p(x, a) = \sum_a p(x|a) \, p(a)$$

This law is frequently used in reinforcement learning.

**Conditional form**:
$$p(x|a) = \sum_y p(x, y|a)$$

### Independence

**Definition**: Two random variables $X$ and $Y$ are independent if the sampling value of one does not affect the other:

$$p(x, y) = p(x) \, p(y)$$

Since $p(x, y) = p(x|y) \, p(y)$, independence implies:
$$p(x|y) = p(x)$$

### Conditional Independence

**Definition**: $X$ is conditionally independent of $A$ given $B$ if:
$$p(X = x | A = a, B = b) = p(X = x | B = b)$$

**RL context**: For three consecutive states $s_t, s_{t+1}, s_{t+2}$: if $s_{t+1}$ is given, then $s_{t+2}$ is conditionally independent of $s_t$:
$$p(s_{t+2} | s_{t+1}, s_t) = p(s_{t+2} | s_{t+1})$$
This is the **memoryless property** of Markov processes.

### Chain Rule of Conditional and Joint Probability

From the definition of conditional probability:
$$p(a, b) = p(a|b) \, p(b)$$

Extended to three variables:
$$p(a, b, c) = p(a|b, c) \, p(b, c) = p(a|b, c) \, p(b|c) \, p(c)$$

This yields:
$$p(a, b|c) = \frac{p(a, b, c)}{p(c)} = p(a|b, c) \, p(b|c)$$

**Useful consequence**:
$$p(x|a) = \sum_b p(x, b|a) = \sum_b p(x|b, a) \, p(b|a)$$

---

### Expectation / Expected Value / Mean

**Definition**: For a random variable $X$ with probability $p(x)$:
$$E[X] = \sum_x p(x) \, x$$

### Linearity of Expectation

$$E[X + Y] = E[X] + E[Y]$$
$$E[aX] = a \, E[X]$$

**Proof of $E[X+Y] = E[X] + E[Y]$**:
$$E[X + Y] = \sum_x \sum_y (x + y) \, p(X = x, Y = y) = \sum_x x \sum_y p(x,y) + \sum_y y \sum_x p(x,y) = \sum_x x \, p(x) + \sum_y y \, p(y) = E[X] + E[Y]$$

**General linear combination**:
$$E\left[\sum_i a_i X_i\right] = \sum_i a_i \, E[X_i]$$

**Matrix form**: For a deterministic matrix $A \in \mathbb{R}^{n \times n}$ and random vector $X \in \mathbb{R}^n$:
$$E[AX] = A \, E[X]$$

---

### Conditional Expectation

**Definition**:
$$E[X | A = a] = \sum_x x \, p(x|a)$$

### Law of Total Expectation

$$E[X] = \sum_a E[X | A = a] \, p(a)$$

**Proof**:
$$\sum_a E[X|A=a] \, p(a) = \sum_a \left[\sum_x p(x|a) \, x\right] p(a) = \sum_x \sum_a p(x|a) \, p(a) \, x = \sum_x \left[\sum_a p(x, a)\right] x = \sum_x p(x) \, x = E[X]$$

The law of total expectation is **frequently used** in reinforcement learning.

### Extended Conditional Expectation

$$E[X | A = a] = \sum_b E[X | A = a, B = b] \, p(b|a)$$

This equation is useful in the derivation of the **Bellman equation**. The proof hint uses the chain rule: $p(x|a, b) \, p(b|a) = p(x, b|a)$.

### $E[X|A = a]$ vs. $E[X|A]$

- $E[X|A = a]$ is a **value** (a specific number)
- $E[X|A]$ is a **random variable** (a function of the random variable $A$)

Defining $E[X|A]$ rigorously requires measure-theoretic probability theory (Appendix B).

---

### Gradient of Expectation

**[ERRATA CORRECTED]** Let $f(X, \beta)$ be a scalar function of a random variable $X$ and a deterministic parameter vector $\beta$. Then:

$$\nabla_\beta E[f(X, \beta)] = E[\nabla_\beta f(X, \beta)]$$

**Proof**: Since $E[f(X, \beta)] = \sum_x f(x, \beta) \, p(x)$, we have:
$$\nabla_\beta E[f(X, \beta)] = \nabla_\beta \sum_x f(x, \beta) \, p(x) = \sum_x \nabla_\beta f(x, \beta) \, p(x) = E[\nabla_\beta f(X, \beta)]$$

> **Errata note**: The original text had $f(x, a)$ instead of $f(x, \beta)$ in the expectation and gradient derivation steps. The corrected version uses $f(x, \beta)$ consistently.

**Key condition**: This interchange of gradient and expectation is valid because $p(x)$ does not depend on $\beta$. (When $p$ depends on $\beta$, the log-derivative trick / REINFORCE is needed -- see Chapter 9.)

---

### Variance, Covariance, Covariance Matrix

**[ERRATA CORRECTED]** The original text used capital $\bar{X}$ and $\bar{Y}$; the corrected version uses lowercase $\bar{x}$ and $\bar{y}$.

**Variance** (single random variable $X$):
$$\text{var}(X) = E[(X - \bar{x})^2], \quad \text{where } \bar{x} = E[X]$$

**Covariance** (two random variables $X, Y$):
$$\text{cov}(X, Y) = E[(X - \bar{x})(Y - \bar{y})]$$

**Covariance matrix** (random vector $X = [X_1, \ldots, X_n]^T$):
$$\text{var}(X) \doteq \Sigma = E[(X - \bar{x})(X - \bar{x})^T] \in \mathbb{R}^{n \times n}$$

The $(i,j)$-th entry: $[\Sigma]_{ij} = E[(X_i - \bar{x}_i)(X_j - \bar{x}_j)] = \text{cov}(X_i, X_j)$

**Trivial property**: $\text{var}(a) = 0$ if $a$ is deterministic.

**Affine transformation**:
$$\text{var}(AX + a) = \text{var}(AX) = A \, \text{var}(X) \, A^T = A \Sigma A^T$$

### Useful Facts about Variance and Covariance

**Fact 1**: $E[(X - \bar{x})(Y - \bar{y})] = E[XY] - \bar{x}\bar{y} = E[XY] - E[X] \, E[Y]$

**Proof**:
$$E[(X-\bar{x})(Y-\bar{y})] = E[XY - X\bar{y} - \bar{x}Y + \bar{x}\bar{y}] = E[XY] - E[X]\bar{y} - \bar{x}E[Y] + \bar{x}\bar{y} = E[XY] - E[X]E[Y]$$

**Fact 2**: $E[XY] = E[X] \, E[Y]$ if $X, Y$ are independent.

**Proof**: $E[XY] = \sum_x \sum_y p(x,y) \, xy = \sum_x \sum_y p(x) \, p(y) \, xy = \left(\sum_x p(x) \, x\right)\left(\sum_y p(y) \, y\right) = E[X] \, E[Y]$

**Fact 3**: $\text{cov}(X, Y) = 0$ if $X, Y$ are independent.

**Proof**: $\text{cov}(X, Y) = E[XY] - E[X] \, E[Y] = E[X] \, E[Y] - E[X] \, E[Y] = 0$

---

## Appendix B: Measure-Theoretic Probability Theory

Measure-theoretic (rigorous) probability theory is needed for rigorously analyzing the **convergence of stochastic sequences**, which arises in Chapter 6 and Chapter 7 (e.g., almost sure convergence). This appendix covers basic notions; comprehensive references include [96-98].

### Probability Triples

A **probability triple** (also called a probability space or probability measure space) consists of three ingredients: $(\Omega, \mathcal{F}, P)$.

#### Sample Space $\Omega$

**Definition**: A set called the **sample space** (or outcome space). Any element $\omega \in \Omega$ is called an **outcome**. Contains all possible outcomes of a random sampling process.

**Example**: Dice game: $\Omega = \{1, 2, 3, 4, 5, 6\}$.

#### Event Space $\mathcal{F}$

**Definition**: A set called the **event space**, which is a $\sigma$-algebra (or $\sigma$-field) of $\Omega$. An element $A \in \mathcal{F}$ is called an **event**.

- An **elementary event** refers to a single outcome in the sample space
- An event may be elementary or a combination of multiple elementary events

**Example**: Dice game. The event "number greater than 3" is $A = \{\omega \in \Omega : \omega > 3\} = \{4, 5, 6\}$.

#### Definition of a $\sigma$-algebra

A $\sigma$-algebra $\mathcal{F}$ is a set of subsets of $\Omega$ satisfying:
1. $\mathcal{F}$ contains $\emptyset$ and $\Omega$
2. $\mathcal{F}$ is closed under complements
3. $\mathcal{F}$ is closed under countable unions and intersections

The $\sigma$-algebras of a given $\Omega$ are **not unique**. $\mathcal{F}$ may contain all subsets of $\Omega$, or only some, as long as the three conditions are satisfied.

**Example**: For $\Omega = \{1,2,3,4,5,6\}$:
- $\mathcal{F} = \{\Omega, \emptyset, \{1,2,3\}, \{4,5,6\}\}$ is a $\sigma$-algebra
- $\{\Omega, \emptyset, \{1,2,3,4,5\}, \{6\}\}$ is also a $\sigma$-algebra
- The collection of **all** subsets of $\Omega$ is always a $\sigma$-algebra (for finite $\Omega$)

**Note**: The three conditions are not fully independent. For example, if $\mathcal{F}$ contains $\Omega$ and is closed under complements, then it naturally contains $\emptyset$.

#### Probability Measure $P$

**Definition**: A mapping from $\mathcal{F}$ to $[0, 1]$. For any $A \in \mathcal{F}$, $P(A)$ is the measure of the set $A$.

**Properties**:
- $P(\Omega) = 1$
- $P(\emptyset) = 0$

**Example**: Dice game, event "number > 3": $A = \{4, 5, 6\}$, so $P(A) = 1/2$.

---

### Random Variables (Measure-Theoretic Definition)

Random variables are called "variables" but are actually **functions** that map from $\Omega$ to $\mathbb{R}$:
$$X(\omega) : \Omega \to \mathbb{R}$$

**Formal definition**: A function $X : \Omega \to \mathbb{R}$ is a random variable if:
$$A = \{\omega \in \Omega \,|\, X(\omega) \leq x\} \in \mathcal{F}$$
for all $x \in \mathbb{R}$. This ensures that $X(\omega) \leq x$ is always an event in $\mathcal{F}$.

### Expectation of Simple Random Variables

A random variable is **simple** if $X(\omega)$ takes a finite number of values. Let $\mathcal{X}$ be the set of all possible values. A simple random variable is:
$$X(\omega) \doteq \sum_{x \in \mathcal{X}} x \, \mathbf{1}_{A_x}(\omega)$$
where:
$$A_x = \{\omega \in \Omega \,|\, X(\omega) = x\} \doteq X^{-1}(x)$$
$$\mathbf{1}_{A_x}(\omega) \doteq \begin{cases} 1, & \omega \in A_x \\ 0, & \text{otherwise} \end{cases}$$

Here $\mathbf{1}_{A_x}(\omega) : \Omega \to \{0, 1\}$ is an **indicator function**.

**Expectation definition**:
$$E[X] \doteq \sum_{x \in \mathcal{X}} x \, P(A_x)$$
where $A_x = \{\omega \in \Omega \,|\, X(\omega) = x\}$.

This is similar to but more formal than the non-measure-theoretic definition $E[X] = \sum_{x \in \mathcal{X}} x \, p(x)$.

### Expectation of the Indicator Function

The indicator function $\mathbf{1}_A$ is itself a random variable mapping $\Omega \to \{0, 1\}$. Its expectation is:
$$E[\mathbf{1}_A] = P(A)$$

**Proof**:
$$E[\mathbf{1}_A] = \sum_{z \in \{0,1\}} z \, P(\mathbf{1}_A = z) = 0 \cdot P(\mathbf{1}_A = 0) + 1 \cdot P(\mathbf{1}_A = 1) = P(A)$$

---

### Conditional Expectation as a Random Variable

Three cases to distinguish:
1. $E[X | Y = 2]$ or $E[X | Y = 5]$: a **specific number**
2. $E[X | Y = y]$ where $y$ is a variable: a **function of $y$**
3. $E[X | Y]$ where $Y$ is a random variable: a **random variable** (function of $Y$)

The third case frequently emerges in convergence analyses of stochastic sequences.

### Lemma B.1 (Basic Properties of Conditional Expectation)

Let $X, Y, Z$ be random variables. The following hold:

| Property | Statement |
|---|---|
| (a) | $E[a \| Y] = a$ where $a$ is a given number |
| (b) | $E[aX + bZ \| Y] = a \, E[X\|Y] + b \, E[Z\|Y]$ (linearity) |
| (c) | $E[X \| Y] = E[X]$ if $X, Y$ are independent |
| (d) | $E[X \, f(Y) \| Y] = f(Y) \, E[X\|Y]$ ("pull out what is known") |
| (e) | $E[f(Y) \| Y] = f(Y)$ |
| (f) | $E[X \| Y, f(Y)] = E[X\|Y]$ |
| (g) | If $X \geq 0$, then $E[X\|Y] \geq 0$ |
| (h) | If $X \geq Z$, then $E[X\|Y] \geq E[Z\|Y]$ |

**Proof of (a)**: Show $E[a|Y=y] = a$ for any $y$ that $Y$ can take. This is clearly true.

**Proof of (d)**: $E[Xf(Y)|Y=y] = \sum_x x \, f(y) \, p(x|y) = f(y) \sum_x x \, p(x|y) = f(y) \, E[X|Y=y]$.

### Lemma B.2 (Iterated Expectation Properties)

Since $E[X|Y]$ is a random variable, we can compute its expectation:

| Property | Statement |
|---|---|
| (a) | $E\big[E[X\|Y]\big] = E[X]$ (tower property / law of iterated expectations) |
| (b) | $E\big[E[X\|Y, Z]\big] = E[X]$ |
| (c) | $E\big[E[X\|Y] \,\big|\, Y\big] = E[X\|Y]$ |

**Proof of (a)**: Denote $f(Y) = E[X|Y]$. Then:
$$E[E[X|Y]] = E[f(Y)] = \sum_y f(Y=y) \, p(y) = \sum_y E[X|Y=y] \, p(y) = \sum_y \left(\sum_x x \, p(x|y)\right) p(y)$$
$$= \sum_x x \sum_y p(x|y) \, p(y) = \sum_x x \sum_y p(x,y) = \sum_x x \, p(x) = E[X]$$

**Proof of (b)**: Similar approach:
$$E[E[X|Y,Z]] = \sum_{y,z} E[X|y,z] \, p(y,z) = \sum_{y,z} \sum_x x \, p(x|y,z) \, p(y,z) = \sum_x x \, p(x) = E[X]$$

**Proof of (c)**: Since $E[X|Y]$ is a function of $Y$, denote it $f(Y)$. By Lemma B.1(e): $E[f(Y)|Y] = f(Y) = E[X|Y]$.

---

### Definitions of Stochastic Convergence

Consider the stochastic sequence $\{X_k\} \doteq \{X_1, X_2, \ldots\}$ where each element is a random variable defined on a triple $(\Omega, \mathcal{F}, P)$. There are several types of convergence.

#### Sure Convergence

**Definition**: $\{X_k\}$ converges **surely** (or everywhere, or pointwise) to $X$ if:
$$\lim_{k \to \infty} X_k(\omega) = X(\omega) \quad \text{for all } \omega \in \Omega$$

Equivalently: $A = \Omega$ where $A = \{\omega \in \Omega : \lim_{k \to \infty} X_k(\omega) = X(\omega)\}$.

#### Almost Sure Convergence

**Definition**: $\{X_k\}$ converges **almost surely** (or almost everywhere, or with probability 1, or w.p.1) to $X$ if:
$$P(A) = 1 \quad \text{where } A = \left\{\omega \in \Omega : \lim_{k \to \infty} X_k(\omega) = X(\omega)\right\}$$

The points for which the limit is invalid form a set of **zero measure**. Often written as:
$$P\left(\lim_{k \to \infty} X_k = X\right) = 1$$

Notation: $X_k \xrightarrow{a.s.} X$.

#### Convergence in Probability

**Definition**: $\{X_k\}$ converges **in probability** to $X$ if for any $\epsilon > 0$:
$$\lim_{k \to \infty} P(A_k) = 0 \quad \text{where } A_k = \{\omega \in \Omega : |X_k(\omega) - X(\omega)| > \epsilon\}$$

Simplified: $\lim_{k \to \infty} P(|X_k - X| > \epsilon) = 0$.

**Key difference from (almost) sure convergence**: Sure and almost sure convergence first evaluate convergence of every point in $\Omega$ and then check the measure. Convergence in probability first checks the points satisfying $|X_k - X| > \epsilon$ and then evaluates if their measure converges to zero.

#### Convergence in Mean

**Definition**: $\{X_k\}$ converges **in the $r$-th mean** (or in the $L^r$ norm) to $X$ if:
$$\lim_{k \to \infty} E[|X_k - X|^r] = 0$$

The most frequently used cases are $r = 1$ and $r = 2$.

**Important note**: Convergence in mean is **not** equivalent to $\lim_{k \to \infty} E[X_k - X] = 0$ or $\lim_{k \to \infty} E[X_k] = E[X]$, which only indicates that $E[X_k]$ converges but the variance may not.

#### Convergence in Distribution

**Definition**: The CDF of $X_k$ is $P(X_k \leq a)$ for $a \in \mathbb{R}$. Then $\{X_k\}$ converges to $X$ **in distribution** if:
$$\lim_{k \to \infty} P(X_k \leq a) = P(X \leq a) \quad \text{for all } a \in \mathbb{R}$$

Compact form: $\lim_{k \to \infty} P(A_k) = P(A)$ where $A_k = \{\omega \in \Omega : X_k(\omega) \leq a\}$ and $A = \{\omega \in \Omega : X(\omega) \leq a\}$.

### Relationships Between Convergence Types

$$\text{almost sure convergence} \Rightarrow \text{convergence in probability} \Rightarrow \text{convergence in distribution}$$
$$\text{convergence in mean} \Rightarrow \text{convergence in probability} \Rightarrow \text{convergence in distribution}$$

**Important**: Almost sure convergence and convergence in mean **do not imply** each other.

---

## Appendix C: Convergence of Sequences

These results are useful for analyzing the convergence of RL algorithms such as those in Chapters 6 and 7.

### C.1 Convergence of Deterministic Sequences

#### Convergence of Monotonic Sequences

Consider a deterministic sequence $\{x_k\} \doteq \{x_1, x_2, \ldots\}$ where $x_k \in \mathbb{R}$.

**Theorem C.1 (Convergence of monotonic sequences)**: If the sequence $\{x_k\}$ is:
- **Nonincreasing**: $x_{k+1} \leq x_k$ for all $k$
- **Bounded from below**: $x_k \geq \alpha$ for all $k$

then $x_k$ converges to a limit (the infimum of $\{x_k\}$) as $k \to \infty$.

Similarly, if $\{x_k\}$ is nondecreasing and bounded from above, then the sequence converges.

#### Convergence of Nonmonotonic Sequences

**Useful operator**: For any $z \in \mathbb{R}$, define:
$$z^+ \doteq \begin{cases} z, & \text{if } z \geq 0 \\ 0, & \text{if } z < 0 \end{cases} \qquad z^- \doteq \begin{cases} z, & \text{if } z \leq 0 \\ 0, & \text{if } z > 0 \end{cases}$$

Properties: $z^+ \geq 0$, $z^- \leq 0$, and $z = z^+ + z^-$ for all $z \in \mathbb{R}$.

**Decomposition of $x_k$**: Rewrite $x_k$ as:
$$x_k = \sum_{i=1}^{k-1} (x_{i+1} - x_i) + x_1 \doteq S_k + x_1$$

where $S_k \doteq \sum_{i=1}^{k-1}(x_{i+1} - x_i)$. Decompose $S_k$ as:
$$S_k = S_k^+ + S_k^- \quad \text{where } S_k^+ = \sum_{i=1}^{k-1}(x_{i+1} - x_i)^+ \geq 0, \quad S_k^- = \sum_{i=1}^{k-1}(x_{i+1} - x_i)^- \leq 0$$

**Properties of $S_k^+$ and $S_k^-$**:
- $\{S_k^+ \geq 0\}$ is a **nondecreasing** sequence since $S_{k+1}^+ \geq S_k^+$
- $\{S_k^- \leq 0\}$ is a **nonincreasing** sequence since $S_{k+1}^- \leq S_k^-$
- If $S_k^+$ is bounded from above, then $S_k^-$ is bounded from below (because $S_k^- \geq -S_k^+ - x_1$ since $S_k^- + S_k^+ + x_1 = x_k \geq 0$)

**Theorem C.2 (Convergence of nonmonotonic sequences)**: For any nonnegative sequence $\{x_k \geq 0\}$, if:
$$\sum_{k=1}^{\infty} (x_{k+1} - x_k)^+ < \infty$$
then $\{x_k\}$ converges as $k \to \infty$.

**Proof**:
1. The condition implies $S_k^+$ is bounded from above. Since $\{S_k^+\}$ is nondecreasing, it converges by Theorem C.1. Let $S_k^+ \to S_*^+$.
2. Boundedness of $S_k^+$ implies $S_k^-$ is bounded from below ($S_k^- \geq -S_k^+ - x_1$). Since $\{S_k^-\}$ is nonincreasing, it converges by Theorem C.1. Let $S_k^- \to S_*^-$.
3. Since $x_k = S_k^+ + S_k^- + x_1$, the sequence $\{x_k\}$ converges to $S_*^+ + S_*^- + x_1$.

**Relationship to Theorem C.1**: Theorem C.2 is more general because it allows $x_k$ to increase as long as the increase is damped. In the monotonic case ($x_{k+1} \leq x_k$), we have $\sum_{k=1}^{\infty}(x_{k+1} - x_k)^+ = 0$, so Theorem C.2 still applies.

#### Corollary C.1 (Perturbation Bound)

For any nonnegative sequence $\{x_k \geq 0\}$, if:
$$x_{k+1} \leq x_k + \eta_k$$
and $\{\eta_k \geq 0\}$ satisfies $\sum_{k=1}^{\infty} \eta_k < \infty$, then $\{x_k\}$ converges.

**Proof**: Since $x_{k+1} \leq x_k + \eta_k$, we have $(x_{k+1} - x_k)^+ \leq \eta_k$. Therefore:
$$\sum_{k=1}^{\infty} (x_{k+1} - x_k)^+ \leq \sum_{k=1}^{\infty} \eta_k < \infty$$
and convergence follows from Theorem C.2.

**Interpretation**: When $\eta_k = 0$, the sequence is monotonic ($x_{k+1} \leq x_k$). When $\eta_k \geq 0$, the sequence is not monotonic (may increase), but convergence is ensured if the perturbation $\eta_k$ is summable.

---

### C.2 Convergence of Stochastic Sequences

#### Martingales

**Definition**: A stochastic sequence $\{X_k\}_{k=1}^{\infty}$ is called a **martingale** if $E[|X_k|] < \infty$ and:
$$E[X_{k+1} | X_1, \ldots, X_k] = X_k$$
almost surely for all $k$.

Here $E[X_{k+1}|X_1, \ldots, X_k]$ is a random variable. It is often written as $E[X_{k+1}|H_k]$ where $H_k = \{X_1, \ldots, X_k\}$ is called a **filtration**.

**Example**: Random walk -- a stochastic process describing the position of a point moving randomly. If the mean of the one-step displacement is zero, then $E[X_{k+1}|X_1, \ldots, X_k] = X_k$ and $\{X_k\}$ is a martingale.

**Basic property of martingales**:
$$E[X_{k+1}] = E[X_k] \quad \text{for all } k$$
$$\Rightarrow E[X_k] = E[X_{k-1}] = \cdots = E[X_1]$$

This follows from taking expectations on both sides using Lemma B.2(b).

#### Submartingales

**Definition**: A stochastic sequence $\{X_k\}$ is a **submartingale** if $E[|X_k|] < \infty$ and:
$$E[X_{k+1} | X_1, \ldots, X_k] \geq X_k \quad \text{for all } k$$

**Property**: The expectation is **nondecreasing**:
$$E[X_k] \geq E[X_{k-1}] \geq \cdots \geq E[X_1]$$

**Memory aid**: "sub" contains the letter "b" which points **up**, so its expectation increases.

#### Supermartingales

**Definition**: A stochastic sequence $\{X_k\}$ is a **supermartingale** if $E[|X_k|] < \infty$ and:
$$E[X_{k+1} | X_1, \ldots, X_k] \leq X_k \quad \text{for all } k$$

**Property**: The expectation is **nonincreasing**:
$$E[X_k] \leq E[X_{k-1}] \leq \cdots \leq E[X_1]$$

**Memory aid**: "super" contains the letter "p" which points **down**, so its expectation decreases.

**Note on ordering**: For two random variables $X$ and $Y$, $X \leq Y$ means $X(\omega) \leq Y(\omega)$ for all $\omega \in \Omega$. It does **not** mean the maximum of $X$ is less than the minimum of $Y$.

#### Theorem C.3 (Martingale Convergence Theorem)

If $\{X_k\}$ is a submartingale (or supermartingale), then there is a **finite** random variable $X$ such that $X_k \to X$ **almost surely**.

A supermartingale/submartingale is comparable to a deterministic monotonic sequence: Theorem C.3 is the stochastic analog of Theorem C.1.

---

#### Quasimartingales

Quasimartingales generalize martingales to the case where expectations are **not monotonic** -- comparable to nonmonotonic deterministic sequences.

**Setup**: Define the event:
$$A_k \doteq \{\omega \in \Omega : E[X_{k+1} - X_k | H_k] \geq 0\}$$
where $H_k = \{X_1, \ldots, X_k\}$. Intuitively, $A_k$ indicates that $X_{k+1}$ is greater than $X_k$ in expectation.

**Indicator function**:
$$\mathbf{1}_{A_k} = \begin{cases} 1, & E[X_{k+1} - X_k | H_k] \geq 0 \\ 0, & E[X_{k+1} - X_k | H_k] < 0 \end{cases}$$

**Property of indicator functions**: $1 = \mathbf{1}_A + \mathbf{1}_{A^c}$ for any event $A$ with complement $A^c$. Hence for any random variable: $X = \mathbf{1}_A X + \mathbf{1}_{A^c} X$.

#### Theorem C.4 (Quasimartingale Convergence Theorem)

For a **nonnegative** stochastic sequence $\{X_k \geq 0\}$, if:
$$\sum_{k=1}^{\infty} E[(X_{k+1} - X_k) \, \mathbf{1}_{A_k}] < \infty$$
then $\sum_{k=1}^{\infty} E[(X_{k+1} - X_k) \, \mathbf{1}_{A_k^c}] > -\infty$ and there is a finite random variable $X$ such that $X_k \to X$ **almost surely** as $k \to \infty$.

Theorem C.4 is the stochastic analog of Theorem C.2 (for nonmonotonic deterministic sequences). The nonnegativity of $X_k$ implies that boundedness of the "increasing part" guarantees boundedness of the "decreasing part".

---

### Summary and Comparison: Deterministic vs. Stochastic Sequences

| Type | Monotonic | Convergence Result |
|---|---|---|
| **Deterministic, monotonic** | Yes (nonincreasing + bounded below) | Converges (Theorem C.1) |
| **Deterministic, nonmonotonic** | No, but $\sum(x_{k+1}-x_k)^+ < \infty$ | Converges (Theorem C.2) |
| **Stochastic, sub/supermartingale** | Yes (expectation monotonic) | Converges a.s. (Theorem C.3) |
| **Stochastic, quasimartingale** | No (nonmonotonic expectation) | Converges a.s. if variation damped (Theorem C.4) |

| Martingale Variant | Monotonicity of $E[X_k]$ |
|---|---|
| Martingale | Constant: $E[X_{k+1}] = E[X_k]$ |
| Submartingale | Increasing: $E[X_{k+1}] \geq E[X_k]$ |
| Supermartingale | Decreasing: $E[X_{k+1}] \leq E[X_k]$ |
| Quasimartingale | Non-monotonic |

---

## Appendix D: Preliminaries for Gradient Descent

The gradient descent method is one of the most frequently used optimization methods and is the foundation for the **stochastic gradient descent** method introduced in Chapter 6.

### Convexity

#### Convex Set

**Definition**: A subset $D \subseteq \mathbb{R}^n$ is **convex** if:
$$z \doteq cx + (1-c)y \in D \quad \text{for any } x, y \in D \text{ and any } c \in [0, 1]$$

#### Convex Function

**[ERRATA CORRECTED]** The original text had $f(cx + (1-x)y)$; the corrected version is $f(cx + (1-c)y)$.

**Definition**: Suppose $f : D \to \mathbb{R}$ where $D$ is convex. The function $f(x)$ is **convex** if:
$$f(cx + (1-c)y) \leq c \, f(x) + (1-c) \, f(y)$$
for any $x, y \in D$ and $c \in [0, 1]$.

### Convexity Conditions

#### First-Order Condition

For $f : D \to \mathbb{R}$ where $D$ is convex, $f$ is convex if:
$$f(y) - f(x) \geq \nabla f(x)^T (y - x) \quad \text{for all } x, y \in D$$

**Geometric interpretation**: When $x$ is a scalar, $\nabla f(x)$ is the slope of the tangent line. The condition means the point $(y, f(y))$ is always located **above** the tangent line at $x$.

#### Second-Order Condition

For $f : D \to \mathbb{R}$ where $D$ is convex, $f$ is convex if:
$$\nabla^2 f(x) \succeq 0 \quad \text{for all } x \in D$$
where $\nabla^2 f(x)$ is the **Hessian matrix**.

### Degree of Convexity

The Hessian matrix describes the degree of convexity:
- If $\nabla^2 f(x)$ is close to rank deficiency: function is **flat** (weakly convex)
- If the minimum singular value of $\nabla^2 f(x)$ is positive and large: function is **curvy** (strongly convex)

The degree of convexity influences **step size selection** in gradient descent.

#### Lower Bound (Strongly Convex)

**Definition**: A function is **strongly convex** (or strictly convex) if:
$$\nabla^2 f(x) \succeq \ell I_n \quad \text{where } \ell > 0 \text{ for all } x$$

#### Upper Bound (Lipschitz Gradient)

If $\nabla^2 f(x) \preceq L I_n$, then the change in $\nabla f(x)$ cannot be arbitrarily fast; equivalently, the function cannot be arbitrarily convex at a point.

**Lemma D.1**: Suppose $f$ is a convex function. If $\nabla f(x)$ is **Lipschitz continuous** with constant $L$:
$$\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\| \quad \text{for all } x, y$$
then $\nabla^2 f(x) \preceq L I_n$ for all $x$. Here $\|\cdot\|$ denotes the Euclidean norm.

---

### Gradient Descent Algorithm

**Problem**: $\min_x f(x)$ where $x \in D \subseteq \mathbb{R}^n$ and $f : D \to \mathbb{R}$.

**Algorithm**:
$$x_{k+1} = x_k - \alpha_k \nabla f(x_k), \quad k = 0, 1, 2, \ldots$$
where $\alpha_k > 0$ is the **step size** (or **learning rate**), which may be fixed or time-varying.

#### Direction of Change

$\nabla f(x_k)$ points in the direction of **fastest increase** of $f$. Hence $-\alpha_k \nabla f(x_k)$ changes $x_k$ in the direction of **fastest decrease**.

#### Magnitude of Change

The magnitude of $-\alpha_k \nabla f(x_k)$ depends on both $\alpha_k$ and $\|\nabla f(x_k)\|$:

**Effect of $\|\nabla f(x_k)\|$**:
- Near the optimum $x^*$ (where $\nabla f(x^*) = 0$): $\|\nabla f(x_k)\|$ is small, so updates are slow (avoids overshooting)
- Far from the optimum: $\|\nabla f(x_k)\|$ may be large, so updates are fast (approaches optimum quickly)

**Effect of $\alpha_k$**:
- Small $\alpha_k$: slow convergence
- Too large $\alpha_k$: aggressive updates leading to fast convergence or divergence

**Step size selection guideline**: $\alpha_k$ should depend on the degree of convexity:
- **Curvy** (strongly convex): $\alpha_k$ should be small
- **Flat** (weakly convex): $\alpha_k$ can be large

---

### Convergence Analysis of Gradient Descent

**Goal**: Show $x_k \to x^*$ where $\nabla f(x^*) = 0$.

**Assumptions**:
1. $f(x)$ is strongly convex: $\nabla^2 f(x) \succeq \ell I$ where $\ell > 0$
2. $\nabla f(x)$ is Lipschitz continuous with constant $L$, implying $\nabla^2 f(x) \preceq L I_n$ (by Lemma D.1)

**Proof**:

**Step 1**: By Taylor expansion with exact remainder:
$$f(x_{k+1}) = f(x_k) + \nabla f(x_k)^T(x_{k+1} - x_k) + \frac{1}{2}(x_{k+1} - x_k)^T \nabla^2 f(z_k)(x_{k+1} - x_k)$$
where $z_k$ is a convex combination of $x_k$ and $x_{k+1}$.

**Step 2**: Using $\|\nabla^2 f(z_k)\| \leq L$:
$$f(x_{k+1}) \leq f(x_k) + \nabla f(x_k)^T(x_{k+1} - x_k) + \frac{L}{2}\|x_{k+1} - x_k\|^2$$

**Step 3**: Substituting $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$:
$$f(x_{k+1}) \leq f(x_k) - \alpha_k \|\nabla f(x_k)\|^2 + \frac{\alpha_k^2 L}{2}\|\nabla f(x_k)\|^2$$
$$= f(x_k) - \underbrace{\alpha_k\left(1 - \frac{\alpha_k L}{2}\right)}_{\eta_k} \|\nabla f(x_k)\|^2$$

**Step 4**: If the step size satisfies:
$$0 < \alpha_k < \frac{2}{L}$$
then $\eta_k > 0$, so $f(x_{k+1}) \leq f(x_k)$. The sequence $\{f(x_k)\}$ is **nonincreasing**.

**Step 5**: Since $f(x_k) \geq f(x^*)$ (bounded below), $\{f(x_k)\}$ converges by Theorem C.1. Let $f^*$ be the limit. Taking the limit of both sides:
$$f^* \leq f^* - \lim_{k \to \infty} \eta_k \|\nabla f(x_k)\|^2$$
$$\Rightarrow 0 \leq -\lim_{k \to \infty} \eta_k \|\nabla f(x_k)\|^2$$

Since $\eta_k \|\nabla f(x_k)\|^2 \geq 0$, this implies $\lim_{k \to \infty} \eta_k \|\nabla f(x_k)\|^2 = 0$, so $x_k \to x^*$ where $\nabla f(x^*) = 0$.

### Key Insight: Step Size Selection

The condition $0 < \alpha_k < 2/L$ confirms the intuition:
- **Flat function** ($L$ small): step size can be **large** ($2/L$ is large)
- **Strongly convex** ($L$ large): step size must be **small** ($2/L$ is small)

---

## Notation Reference (from Symbols table)

| Symbol | Meaning |
|---|---|
| $=$ | Equality |
| $\approx$ | Approximation |
| $\doteq$ | Equality by definition |
| $\geq, >, \leq, <$ | Elementwise comparison |
| $\in$ | Is an element of |
| $\|\cdot\|_2$ | Euclidean norm of a vector or induced matrix norm |
| $\|\cdot\|_\infty$ | Maximum norm of a vector or induced matrix norm |
| $\ln$ | Natural logarithm |
| $\mathbb{R}$ | Set of real numbers |
| $\mathbb{R}^n$ | Set of $n$-dimensional real vectors |
| $\mathbb{R}^{n \times m}$ | Set of all $n \times m$ real matrices |
| $A \succeq 0$ ($A \succ 0$) | Matrix $A$ is positive semidefinite (definite) |
| $A \preceq 0$ ($A \prec 0$) | Matrix $A$ is negative semidefinite (definite) |
| $|x|$ | Absolute value of real scalar $x$ |
| $|S|$ | Number of elements in set $S$ |
| $\nabla_x f(x)$ | Gradient of scalar function $f(x)$ w.r.t. vector $x$ (written $\nabla f(x)$ for short) |
| $[A]_{ij}$ | Element in $i$-th row and $j$-th column of matrix $A$ |
| $[x]_i$ | $i$-th element of vector $x$ |
| $X \sim p$ | $p$ is the probability distribution of random variable $X$ |
| $p(X = x)$, $\Pr(X = x)$ | Probability of $X = x$ (often written $p(x)$ or $\Pr(x)$) |
| $p(x|y)$ | Conditional probability |
| $E_{X \sim p}[X]$ | Expectation of $X$ (often written $E[X]$) |
| $\text{var}(X)$ | Variance of random variable $X$ |
| $\arg\max_x f(x)$ | Maximizer of function $f(x)$ |
| $\mathbf{1}_n$ | Vector of all ones (written $\mathbf{1}$ when dimension is clear) |
| $I_n$ | $n \times n$ identity matrix (written $I$ when dimension is clear) |

---

## Concept Index

| Concept | Notation / Formula | Section |
|---|---|---|
| Random variable | $X$ (capital), value $x$ (lowercase) | A |
| Stochastic sequence | $\{x_i\}_{i=1}^n$ | A |
| Probability | $p(X = x)$ or $p(x)$ | A |
| Joint probability | $p(x, y)$; $\sum_y p(x,y) = p(x)$ | A |
| Conditional probability | $p(x|a) = p(x,a)/p(a)$ | A |
| Law of total probability | $p(x) = \sum_a p(x|a)p(a)$ | A |
| Independence | $p(x,y) = p(x)p(y)$ | A |
| Conditional independence | $p(x|a,b) = p(x|b)$ | A |
| Chain rule | $p(a,b,c) = p(a|b,c)p(b|c)p(c)$ | A |
| Expectation / mean | $E[X] = \sum_x p(x) \, x$ | A |
| Linearity of expectation | $E[X+Y] = E[X] + E[Y]$ | A |
| Conditional expectation | $E[X|A=a] = \sum_x x \, p(x|a)$ | A |
| Law of total expectation | $E[X] = \sum_a E[X|A=a]p(a)$ | A |
| $E[X|A=a]$ vs $E[X|A]$ | Value vs. random variable | A |
| Gradient of expectation | $\nabla_\beta E[f(X,\beta)] = E[\nabla_\beta f(X,\beta)]$ | A |
| Variance | $\text{var}(X) = E[(X - \bar{x})^2]$ | A |
| Covariance | $\text{cov}(X,Y) = E[(X-\bar{x})(Y-\bar{y})]$ | A |
| Covariance matrix | $\Sigma = E[(X-\bar{x})(X-\bar{x})^T]$ | A |
| Probability triple | $(\Omega, \mathcal{F}, P)$ | B |
| Sample space | $\Omega$ | B |
| Event space / $\sigma$-algebra | $\mathcal{F}$ | B |
| Probability measure | $P : \mathcal{F} \to [0,1]$ | B |
| Random variable (measure-theoretic) | $X(\omega) : \Omega \to \mathbb{R}$ | B |
| Simple random variable | $X(\omega) = \sum_x x \, \mathbf{1}_{A_x}(\omega)$ | B |
| Indicator function | $\mathbf{1}_A(\omega)$; $E[\mathbf{1}_A] = P(A)$ | B |
| Conditional expectation (random variable) | $E[X|Y]$ is a function of $Y$ | B |
| Tower property | $E[E[X|Y]] = E[X]$ | B (Lemma B.2) |
| Sure convergence | $\lim X_k(\omega) = X(\omega)$ for all $\omega$ | B |
| Almost sure convergence | $P(\lim X_k = X) = 1$; $X_k \xrightarrow{a.s.} X$ | B |
| Convergence in probability | $\lim P(|X_k - X| > \epsilon) = 0$ | B |
| Convergence in mean ($L^r$) | $\lim E[|X_k - X|^r] = 0$ | B |
| Convergence in distribution | $\lim P(X_k \leq a) = P(X \leq a)$ | B |
| Monotone convergence (deterministic) | Nonincreasing + bounded $\Rightarrow$ converges | C.1 (Thm C.1) |
| Nonmonotonic convergence (deterministic) | $\sum(x_{k+1}-x_k)^+ < \infty \Rightarrow$ converges | C.1 (Thm C.2) |
| Perturbation bound | $x_{k+1} \leq x_k + \eta_k$, $\sum \eta_k < \infty$ | C.1 (Cor C.1) |
| Martingale | $E[X_{k+1}|H_k] = X_k$; $E[X_k]$ constant | C.2 |
| Submartingale | $E[X_{k+1}|H_k] \geq X_k$; $E[X_k]$ nondecreasing | C.2 |
| Supermartingale | $E[X_{k+1}|H_k] \leq X_k$; $E[X_k]$ nonincreasing | C.2 |
| Martingale convergence theorem | Sub/supermartingale $\Rightarrow$ a.s. convergence | C.2 (Thm C.3) |
| Quasimartingale | Nonmonotonic expectation | C.2 |
| Quasimartingale convergence | $\sum E[(X_{k+1}-X_k)\mathbf{1}_{A_k}] < \infty \Rightarrow$ a.s. convergence | C.2 (Thm C.4) |
| Filtration | $H_k = \{X_1, \ldots, X_k\}$ | C.2 |
| Convex set | $cx + (1-c)y \in D$ for $c \in [0,1]$ | D |
| Convex function | $f(cx+(1-c)y) \leq cf(x)+(1-c)f(y)$ | D |
| First-order convexity condition | $f(y) - f(x) \geq \nabla f(x)^T(y-x)$ | D |
| Second-order convexity condition | $\nabla^2 f(x) \succeq 0$ | D |
| Strongly convex | $\nabla^2 f(x) \succeq \ell I_n$, $\ell > 0$ | D |
| Lipschitz continuous gradient | $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$ | D (Lemma D.1) |
| Gradient descent | $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$ | D |
| Step size / learning rate | $\alpha_k$; convergence requires $0 < \alpha_k < 2/L$ | D |

---

## Dependencies and Forward References

| This appendix concept | Used in / Extended by |
|---|---|
| Probability, expectation, conditional expectation | Every chapter (especially Ch 2 Bellman equation derivation) |
| Law of total probability, law of total expectation | Ch 2 (Bellman equation), Ch 5 (Monte Carlo), Ch 9 (policy gradient) |
| Chain rule of conditional/joint probability | Ch 2 (Bellman equation derivation) |
| Conditional independence / Markov property | Ch 1, Ch 2 (Bellman equation) |
| Gradient of expectation (when $p$ independent of $\beta$) | Ch 9 (policy gradient -- contrast with log-derivative trick when $p$ depends on $\beta$) |
| Variance, covariance | Ch 6 (stochastic approximation), Ch 9 (variance reduction in policy gradient) |
| Measure-theoretic probability, almost sure convergence | Ch 6, Ch 7 (convergence analysis of SA and TD algorithms) |
| Conditional expectation as random variable (Lemmas B.1-B.2) | Ch 6 (Robbins-Monro), Ch 7 (TD convergence) |
| Convergence of monotonic sequences (Thm C.1) | Ch 4 (value/policy iteration convergence), Ch D (gradient descent proof) |
| Convergence of nonmonotonic sequences (Thm C.2, Cor C.1) | Ch 6 (SA convergence with perturbations) |
| Martingale/supermartingale convergence (Thm C.3) | Ch 6 (Robbins-Monro convergence), Ch 7 (TD convergence) |
| Quasimartingale convergence (Thm C.4) | Ch 6-7 (convergence of nonmonotonic stochastic algorithms) |
| Convexity, gradient descent, step size selection | Ch 6 (stochastic gradient descent) |
| Lipschitz condition and convergence bounds | Ch 6 (SGD convergence), Ch 8 (function approximation) |
