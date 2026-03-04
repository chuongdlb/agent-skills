---
chapter: 8
title: Value Function Methods
key_topics: [function approximation, linear function approximation, feature vector, parameter vector, objective function, stationary distribution, TD learning with function approximation, TD-Linear, tabular as special case, polynomial features, Fourier features, convergence analysis, projected Bellman error, Bellman error, Bellman operator, least-squares TD, Sarsa with function approximation, Q-learning with function approximation, deep Q-learning, DQN, experience replay, replay buffer, target network, main network]
depends_on: [1, 2, 3, 4, 5, 6, 7]
required_by: [9, 10]
---

# Chapter 8: Value Function Methods

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 8, pp. 151-189
> Supplemented by: Lecture slides L8 (69 slides)
> Errata applied: Algorithm 8.2 and Algorithm 8.3 -- changed inconsistent "epsilon" notation to "ϵ" throughout

## Purpose and Context

This chapter marks the transition from **tabular representation** to **function representation** for state/action values. While Chapters 1-7 used tables to store values, this chapter introduces the **value function approximation** method, where values are represented by parameterized functions. This is the critical step that allows reinforcement learning to scale to large or continuous state/action spaces and is where **artificial neural networks** are incorporated into RL as function approximators.

**Position in the book**: Chapter 8 is the first chapter in Part 2 (Algorithms/Methods) that uses function representation. It builds directly on Chapter 7 (TD methods) and Chapter 6 (stochastic approximation). The function approximation idea introduced here for values is extended to policies in Chapter 9 (policy gradient methods), and both are combined in Chapter 10 (actor-critic methods).

**Key transitions in this chapter**:
- *Tabular representation* -> *Function representation* for values
- *Finite state spaces* -> *Potentially large or continuous state spaces*
- *Direct value storage* -> *Parameterized value approximation*
- *Simple TD updates* -> *Gradient-based optimization of objective functions*

---

## 8.1 Value Representation: From Table to Function

### Motivating Example

Suppose there are $n$ states $\{s_i\}_{i=1}^n$ with true state values $\{v_\pi(s_i)\}_{i=1}^n$ under a given policy $\pi$. Let $\{\hat{v}(s_i)\}_{i=1}^n$ denote the estimates.

**Tabular method**: Store estimated values in a table/array:

| State | $s_1$ | $s_2$ | $\cdots$ | $s_n$ |
|---|---|---|---|---|
| Estimated value | $\hat{v}(s_1)$ | $\hat{v}(s_2)$ | $\cdots$ | $\hat{v}(s_n)$ |

**Function approximation method**: Fit the $n$ points $\{(s_i, \hat{v}(s_i))\}_{i=1}^n$ with a parameterized curve.

### Linear Function Approximation (First-Order)

The simplest approximation is a straight line:

$$\hat{v}(s, w) = as + b = \underbrace{[s, 1]}_{\phi^T(s)} \underbrace{\begin{bmatrix} a \\ b \end{bmatrix}}_{w} = \phi^T(s)w \tag{8.1}$$

where:
- $\hat{v}(s, w)$ is the approximated value, jointly determined by the state $s$ and the parameter vector $w \in \mathbb{R}^2$
- $\phi(s) \in \mathbb{R}^2$ is the **feature vector** of $s$
- $w$ is the **parameter vector**
- The function is **linear in $w$** (though it may be nonlinear in $s$)

### Higher-Order Polynomial Approximation (Second-Order)

$$\hat{v}(s, w) = as^2 + bs + c = \underbrace{[s^2, s, 1]}_{\phi^T(s)} \underbrace{\begin{bmatrix} a \\ b \\ c \end{bmatrix}}_{w} = \phi^T(s)w \tag{8.2}$$

**Key insight**: As the polynomial order increases, approximation accuracy improves but the dimension of $w$ also increases, requiring more storage and computation. This is the **accuracy-efficiency tradeoff**.

### Differences Between Tabular and Function Methods

#### Difference 1: How to Retrieve a Value

| Tabular | Function Approximation |
|---|---|
| Directly read the entry in the table | Input state $s$ into the function, compute: $s \to \phi(s) \to \phi^T(s)w = \hat{v}(s,w)$ |

**Benefit -- Storage efficiency**: The tabular method needs to store $n$ values ($n = |\mathcal{S}|$). The function method only stores a lower-dimensional parameter vector $w \in \mathbb{R}^m$ where $m \ll n$.

**Cost**: The state values may not be accurately represented by the function. This is why the method is called **approximation**. Some information is lost when using a low-dimensional vector to represent a high-dimensional dataset.

#### Difference 2: How to Update a Value

| Tabular | Function Approximation |
|---|---|
| Directly rewrite the entry in the table | Update $w$ to change values indirectly |

**Benefit -- Generalization ability**: When using the tabular method, updating one state value does not change others. With function approximation, updating $w$ affects values of many states simultaneously. The experience sample for one state can **generalize** to help estimate values of other unvisited states.

**Illustrative example**: With three states $\{s_1, s_2, s_3\}$ and an experience sample for $s_3$:
- **Tabular**: Only $\hat{v}(s_3)$ is updated; $\hat{v}(s_1)$ and $\hat{v}(s_2)$ remain unchanged.
- **Function**: Updating $w$ for $s_3$ also changes $\hat{v}(s_1)$ and $\hat{v}(s_2)$.

### Optimal Parameter via Least Squares

If the true values $\{v_\pi(s_i)\}_{i=1}^n$ are known, finding the optimal $w$ is a least-squares problem:

$$J_1 = \sum_{i=1}^n \left(\hat{v}(s_i, w) - v_\pi(s_i)\right)^2 = \|\Phi w - v_\pi\|^2$$

where $\Phi = [\phi^T(s_1); \ldots; \phi^T(s_n)] \in \mathbb{R}^{n \times m}$ and $v_\pi = [v_\pi(s_1), \ldots, v_\pi(s_n)]^T \in \mathbb{R}^n$.

The optimal solution is:

$$w^* = (\Phi^T \Phi)^{-1} \Phi^T v_\pi$$

However, in RL the true values are unknown, so this direct approach is not applicable. The next section introduces how to learn $w$ without knowing $v_\pi$.

---

## 8.2 TD Learning of State Values Based on Function Approximation

This section covers:
1. **Objective function** (Section 8.2.1)
2. **Optimization algorithms** (Section 8.2.2)
3. **Selection of function approximators** (Section 8.2.3)
4. **Illustrative examples** (Section 8.2.4)
5. **Theoretical analysis** (Section 8.2.5)

### 8.2.1 Objective Function

Let $v_\pi(s)$ and $\hat{v}(s, w)$ be the true and approximated state values for $s \in \mathcal{S}$. The goal is to find an optimal $w$ so that $\hat{v}(s, w)$ best approximates $v_\pi(s)$ for every $s$.

**General objective function**:

$$J(w) = \mathbb{E}\left[(v_\pi(S) - \hat{v}(S, w))^2\right] \tag{8.3}$$

where the expectation is over the random variable $S \in \mathcal{S}$. The key question is: **what probability distribution should $S$ follow?**

#### Option 1: Uniform Distribution

Set the probability of each state to $1/n$:

$$J(w) = \frac{1}{n} \sum_{s \in \mathcal{S}} (v_\pi(s) - \hat{v}(s, w))^2 \tag{8.4}$$

**Drawback**: Does not consider the real dynamics of the Markov process. Some states may be rarely visited; treating all states equally is unreasonable.

#### Option 2: Stationary Distribution (Focus of This Chapter)

Let $\{d_\pi(s)\}_{s \in \mathcal{S}}$ denote the **stationary distribution** of the Markov process under policy $\pi$. Then:

$$J(w) = \sum_{s \in \mathcal{S}} d_\pi(s) (v_\pi(s) - \hat{v}(s, w))^2 \tag{8.5}$$

This is a **weighted average** of approximation errors, where states visited more frequently receive greater weight.

**Important**: The value of $d_\pi(s)$ is nontrivial to compute (requires knowing $P_\pi$), but we do **not** need to compute it explicitly to minimize this objective function.

#### Box 8.1: Stationary Distribution of a Markov Decision Process

**Definition and Background**:

The key tool is $P_\pi \in \mathbb{R}^{n \times n}$, the probability transition matrix under policy $\pi$, where $[P_\pi]_{ij}$ is the probability of transitioning from $s_i$ to $s_j$.

**Interpretation of $P_\pi^k$**: $[P_\pi^k]_{ij} = p_{ij}^{(k)}$ is the probability of transitioning from $s_i$ to $s_j$ in exactly $k$ steps.

**State distribution after $k$ steps**: Let $d_0 \in \mathbb{R}^n$ be the initial state distribution. Then:

$$d_k(s_i) = \sum_{j=1}^n d_0(s_j) [P_\pi^k]_{ji} \tag{8.6}$$

In matrix-vector form:

$$d_k^T = d_0^T P_\pi^k \tag{8.7}$$

**Limiting distribution**: Under certain conditions:

$$\lim_{k \to \infty} P_\pi^k = \mathbf{1}_n d_\pi^T \tag{8.8}$$

where $\mathbf{1}_n = [1, \ldots, 1]^T \in \mathbb{R}^n$. This yields:

$$\lim_{k \to \infty} d_k^T = d_0^T \lim_{k \to \infty} P_\pi^k = d_0^T \mathbf{1}_n d_\pi^T = d_\pi^T \tag{8.9}$$

The last equality holds because $d_0^T \mathbf{1}_n = 1$.

**Key property**: The limiting distribution $d_\pi$ is **independent of the initial distribution** $d_0$.

**Computing $d_\pi$**: Taking the limit of $d_k^T = d_{k-1}^T P_\pi$ gives:

$$d_\pi^T = d_\pi^T P_\pi \tag{8.10}$$

Therefore, $d_\pi$ is the **left eigenvector** of $P_\pi$ associated with eigenvalue 1. It satisfies $\sum_{s \in \mathcal{S}} d_\pi(s) = 1$ and $d_\pi(s) > 0$ for all $s$.

**Conditions for unique stationary distribution**:

- **Accessible**: State $s_j$ is accessible from $s_i$ if there exists finite $k$ such that $[P_\pi^k]_{ij} > 0$.
- **Communicate**: Two states communicate if they are mutually accessible.
- **Irreducible**: All states communicate with each other.
- **Regular**: There exists $k \geq 1$ such that $P_\pi^k > 0$ (elementwise). A regular process is irreducible, but not vice versa. However, if a process is irreducible and $[P_\pi]_{ii} > 0$ for some $i$, it is also regular.

**Policies leading to unique stationary distributions**: Exploratory policies such as **$\epsilon$-greedy policies** generally lead to regular Markov processes because they have positive probability of taking any action at any state.

**Worked Example** (4-state grid world, $\epsilon$-greedy with $\epsilon = 0.5$):

The transition matrix is:

$$P_\pi^T = \begin{bmatrix} 0.3 & 0.1 & 0.1 & 0 \\ 0.1 & 0.3 & 0 & 0.1 \\ 0.6 & 0 & 0.3 & 0.1 \\ 0 & 0.6 & 0.6 & 0.8 \end{bmatrix}$$

Eigenvalues of $P_\pi^T$: $\{-0.0449, 0.3, 0.4449, 1\}$.

The unit-length eigenvector for eigenvalue 1, after normalization (sum = 1):

$$d_\pi = \begin{bmatrix} 0.0345 \\ 0.1084 \\ 0.1330 \\ 0.7241 \end{bmatrix}$$

**Verification**: Running 1,000 steps starting from $s_1$, the visit proportions converge to the theoretical $d_\pi$.

---

### 8.2.2 Optimization Algorithms

To minimize $J(w)$ in (8.3), apply gradient descent:

$$w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)$$

**Computing the gradient**:

$$\nabla_w J(w) = -2\mathbb{E}\left[(v_\pi(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)\right]$$

The gradient descent algorithm becomes:

$$w_{k+1} = w_k + 2\alpha_k \mathbb{E}\left[(v_\pi(S) - \hat{v}(S, w_k)) \nabla_w \hat{v}(S, w_k)\right] \tag{8.11}$$

where the coefficient 2 can be absorbed into $\alpha_k$.

**Stochastic gradient descent (SGD)**: Replace the true gradient with a stochastic gradient:

$$w_{t+1} = w_t + \alpha_t \left(v_\pi(s_t) - \hat{v}(s_t, w_t)\right) \nabla_w \hat{v}(s_t, w_t) \tag{8.12}$$

where $s_t$ is a sample of $S$ at time $t$.

**Problem**: Equation (8.12) is **not implementable** because it requires the true state value $v_\pi$, which is unknown. Two approaches to make it implementable:

#### Monte Carlo Method

Replace $v_\pi(s_t)$ with the discounted return $g_t$:

$$w_{t+1} = w_t + \alpha_t \left(g_t - \hat{v}(s_t, w_t)\right) \nabla_w \hat{v}(s_t, w_t)$$

This is **Monte Carlo learning with function approximation**.

#### Temporal-Difference Method

Replace $v_\pi(s_t)$ with the TD target $r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)$:

$$w_{t+1} = w_t + \alpha_t \left[r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t)\right] \nabla_w \hat{v}(s_t, w_t) \tag{8.13}$$

This is **TD learning with function approximation**, summarized in Algorithm 8.1.

**Critical note**: The algorithm in (8.13) can only learn the **state values of a given policy** (policy evaluation). It will be extended to action values in Section 8.3.

### Algorithm 8.1: TD Learning of State Values with Function Approximation

```
Initialization: A function v_hat(s, w) differentiable in w. Initial parameter w_0.
Goal: Learn the true state values of a given policy pi.

For each episode {(s_t, r_{t+1}, s_{t+1})}_t generated by pi, do
    For each sample (s_t, r_{t+1}, s_{t+1}), do
        General case:
            w_{t+1} = w_t + alpha_t [r_{t+1} + gamma * v_hat(s_{t+1}, w_t) - v_hat(s_t, w_t)] * grad_w v_hat(s_t, w_t)
        Linear case:
            w_{t+1} = w_t + alpha_t [r_{t+1} + phi^T(s_{t+1}) w_t - phi^T(s_t) w_t] * phi(s_t)
```

---

### 8.2.3 Selection of Function Approximators

Two approaches for selecting $\hat{v}(s, w)$:

#### Approach 1: Artificial Neural Network (Nonlinear)

Input: state $s$. Output: $\hat{v}(s, w)$. Parameter: $w$ (network weights).

Neural networks serve as **black-box universal nonlinear approximators** -- more friendly to use since they do not require manual feature engineering.

#### Approach 2: Linear Function

$$\hat{v}(s, w) = \phi^T(s) w$$

where $\phi(s) \in \mathbb{R}^m$ is the feature vector (with $m \ll n$ typically). The gradient is simply:

$$\nabla_w \hat{v}(s, w) = \phi(s)$$

Substituting into (8.13) yields the **TD-Linear** algorithm:

$$w_{t+1} = w_t + \alpha_t \left(r_{t+1} + \gamma \phi^T(s_{t+1}) w_t - \phi^T(s_t) w_t\right) \phi(s_t) \tag{8.14}$$

**Advantages of linear case**:
- Theoretical properties are much better understood
- Tabular method is a special case (see Box 8.2)

**Disadvantages of linear case**:
- Limited approximation ability
- Nontrivial to select appropriate feature vectors

#### Box 8.2: Tabular TD Learning Is a Special Case of TD-Linear

Consider the special feature vector $\phi(s) = e_s \in \mathbb{R}^n$, where $e_s$ is the standard basis vector with entry corresponding to $s$ equal to 1 and all others 0. Then:

$$\hat{v}(s, w) = e_s^T w = w(s)$$

where $w(s)$ is the entry in $w$ corresponding to $s$. Substituting into (8.14):

$$w_{t+1} = w_t + \alpha_t \left(r_{t+1} + \gamma w_t(s_{t+1}) - w_t(s_t)\right) e_{s_t}$$

This updates only the $s_t$-th entry. Multiplying both sides by $e_{s_t}^T$:

$$w_{t+1}(s_t) = w_t(s_t) + \alpha_t \left(r_{t+1} + \gamma w_t(s_{t+1}) - w_t(s_t)\right)$$

This is exactly the **tabular TD algorithm** from equation (7.1) in Chapter 7. Therefore, the tabular and function representations are **unified**: TD-Table is a special case of TD-Linear.

---

### 8.2.4 Illustrative Examples

#### Setup

A **5x5 grid world** with:
- Given policy: $\pi(a|s) = 0.2$ for all $s, a$ (uniform random policy)
- Goal: Estimate the 25 state values (policy evaluation)
- Parameters: $r_{\text{forbidden}} = r_{\text{boundary}} = -1$, $r_{\text{target}} = 1$, $\gamma = 0.9$
- 500 episodes, each with 500 steps, starting from randomly selected state-action pairs
- Parameter $w$ randomly initialized from $\mathcal{N}(0, 1)$

**True state values** (5x5 grid, rows 1-5, columns 1-5):

| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|---|
| Row 1 | -3.8 | -3.8 | -3.6 | -3.1 | -3.2 |
| Row 2 | -3.8 | -3.8 | -3.8 | -3.1 | -2.9 |
| Row 3 | -3.6 | -3.9 | -3.4 | -3.2 | -2.9 |
| Row 4 | -3.9 | -3.6 | -3.4 | -2.9 | -3.2 |
| Row 5 | -4.5 | -4.2 | -3.4 | -3.4 | -3.5 |

#### Polynomial Feature Vectors

Let $x$ and $y$ denote the (normalized to $[-1, +1]$) column and row indexes.

**First-order (3 parameters)**:

$$\phi(s) = \begin{bmatrix} 1 \\ x \\ y \end{bmatrix} \in \mathbb{R}^3, \quad \hat{v}(s,w) = w_1 + w_2 x + w_3 y \tag{8.15}$$

This represents a 2D plane. Result: error converges but cannot reach zero (a plane cannot fit a non-planar surface).

**Second-order (6 parameters)**:

$$\phi(s) = [1, x, y, x^2, y^2, xy]^T \in \mathbb{R}^6 \tag{8.16}$$

$$\hat{v}(s,w) = w_1 + w_2 x + w_3 y + w_4 x^2 + w_5 y^2 + w_6 xy$$

This represents a quadratic 3D surface. Better approximation.

**Third-order (10 parameters)**:

$$\phi(s) = [1, x, y, x^2, y^2, xy, x^3, y^3, x^2 y, xy^2]^T \in \mathbb{R}^{10} \tag{8.17}$$

**Observation**: The longer the feature vector, the more accurately state values can be approximated. However, in all three cases the estimation error cannot converge to zero due to limited approximation ability.

#### Fourier Feature Vectors

Normalize $x$ and $y$ to $[0, 1]$. The Fourier feature vector is:

$$\phi(s) = \begin{bmatrix} \vdots \\ \cos(\pi(c_1 x + c_2 y)) \\ \vdots \end{bmatrix} \in \mathbb{R}^{(q+1)^2} \tag{8.18}$$

where $\pi = 3.1415\ldots$ (circumference ratio, not a policy), $c_1, c_2 \in \{0, 1, \ldots, q\}$, and $q$ is a user-specified integer.

**Example** ($q = 1$, dimension 4):

$$\phi(s) = \begin{bmatrix} \cos(\pi(0 \cdot x + 0 \cdot y)) \\ \cos(\pi(0 \cdot x + 1 \cdot y)) \\ \cos(\pi(1 \cdot x + 0 \cdot y)) \\ \cos(\pi(1 \cdot x + 1 \cdot y)) \end{bmatrix} = \begin{bmatrix} 1 \\ \cos(\pi y) \\ \cos(\pi x) \\ \cos(\pi(x+y)) \end{bmatrix} \in \mathbb{R}^4$$

**Results**: For $q = 1, 2, 3$, feature dimensions are $4, 9, 16$ respectively. Higher dimensions yield more accurate approximation.

---

### 8.2.5 Theoretical Analysis

**Important caveat**: The TD algorithm in (8.13) does **not** actually minimize the objective function $J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S,w))^2]$ directly. This subsection reveals what it does minimize.

#### Convergence Analysis

Consider the **deterministic** version of the TD-Linear algorithm:

$$w_{t+1} = w_t + \alpha_t \mathbb{E}\left[\left(r_{t+1} + \gamma \phi^T(s_{t+1}) w_t - \phi^T(s_t) w_t\right) \phi(s_t)\right] \tag{8.19}$$

where $s_t$ follows the stationary distribution $d_\pi$.

**Why study this deterministic algorithm?** (1) Its convergence is easier to analyze. (2) The stochastic TD algorithm (8.13) can be viewed as an SGD implementation of (8.19), so convergence of (8.19) implies convergence of (8.13).

**Key matrices**:

$$\Phi = \begin{bmatrix} \vdots \\ \phi^T(s) \\ \vdots \end{bmatrix} \in \mathbb{R}^{n \times m}, \quad D = \text{diag}(\ldots, d_\pi(s), \ldots) \in \mathbb{R}^{n \times n} \tag{8.20}$$

#### Lemma 8.1

The expectation in (8.19) simplifies to:

$$\mathbb{E}\left[\left(r_{t+1} + \gamma \phi^T(s_{t+1}) w_t - \phi^T(s_t) w_t\right) \phi(s_t)\right] = b - Aw_t$$

where:

$$A \triangleq \Phi^T D(I - \gamma P_\pi) \Phi \in \mathbb{R}^{m \times m} \tag{8.21}$$
$$b \triangleq \Phi^T D r_\pi \in \mathbb{R}^m$$

Here, $P_\pi$ and $r_\pi$ are from the Bellman equation $v_\pi = r_\pi + \gamma P_\pi v_\pi$.

**Proof sketch** (Box 8.3): Using the law of total expectation, the term splits into two parts:
- First term: $\sum_{s} d_\pi(s) \phi(s) r_\pi(s) = \Phi^T D r_\pi$
- Second term: $-\Phi^T D(I - \gamma P_\pi) \Phi w_t$

Combining gives $b - Aw_t$.

#### Simplified Deterministic Algorithm

With Lemma 8.1, equation (8.19) becomes:

$$w_{t+1} = w_t + \alpha_t (b - Aw_t) \tag{8.22}$$

#### Converged Value

If $w_t$ converges to $w^*$, then $w^* = w^* + \alpha_\infty(b - Aw^*)$, which implies:

$$w^* = A^{-1}b$$

**Key properties of this solution**:

1. **$A$ is invertible and positive definite** (proven in Box 8.4): For any nonzero $x$, $x^T A x > 0$. This is shown by proving that $M = D(I - \gamma P_\pi)$ is positive definite via strict diagonal dominance of $M + M^T$.

2. **Interpretation**: $w^* = A^{-1}b$ minimizes the **projected Bellman error** (see below).

3. **Special case -- tabular**: When $\phi(s) = e_s$ (standard basis vector), $\Phi = I$, and:

$$w^* = A^{-1}b = (I - \gamma P_\pi)^{-1} r_\pi = v_\pi \tag{8.23}$$

So the parameter vector learned is exactly the true state value vector.

#### Convergence Proofs

**Proof 1** (Direct): Define $\delta_t = w_t - w^*$. Then $\delta_{t+1} = (I - \alpha_t A) \delta_t$. For constant step size $\alpha$:

$$\|\delta_{t+1}\|_2 \leq \|I - \alpha A\|_2^{t+1} \|\delta_0\|_2$$

When $\alpha > 0$ is sufficiently small, $\|I - \alpha A\|_2 < 1$ (because $A$ is positive definite), so $\delta_t \to 0$.

**Proof 2** (Robbins-Monro): The algorithm $w_{t+1} = w_t + \alpha_t(b - Aw_t)$ is a root-finding algorithm for $g(w) = b - Aw = 0$. By the RM convergence theorem, $w_t \to w^*$ when $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$.

#### Box 8.4: Proof That $A = \Phi^T D(I - \gamma P_\pi)\Phi$ Is Positive Definite

The idea is to show $M = D(I - \gamma P_\pi) \succ 0$.

Since $M \succ 0$ iff $M + M^T \succ 0$ (because $x^T(M - M^T)x = 0$ for skew-symmetric part), we show $M + M^T$ is **strictly diagonally dominant**:

$$(M + M^T)\mathbf{1}_n = 2(1-\gamma)d_\pi > 0$$

Since diagonal entries of $M$ are positive and off-diagonal entries are nonpositive, this implies:

$$[M + M^T]_{ii} > \sum_{j \neq i} |[M + M^T]_{ij}|$$

Therefore $M + M^T$ is strictly diagonally dominant and hence positive definite. Since $\Phi$ has full column rank, $A = \Phi^T M \Phi \succ 0$.

#### Three Objective Functions

**Objective 1 -- True Value Error**:

$$J_E(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S,w))^2] = \|\hat{v}(w) - v_\pi\|_D^2$$

where $\|x\|_D^2 = x^T D x$ is a weighted norm. This requires the unknown true values.

**Objective 2 -- Bellman Error**:

$$J_{BE}(w) = \|\hat{v}(w) - (r_\pi + \gamma P_\pi \hat{v}(w))\|_D^2 = \|\hat{v}(w) - T_\pi(\hat{v}(w))\|_D^2 \tag{8.30}$$

where $T_\pi(x) \triangleq r_\pi + \gamma P_\pi x$ is the **Bellman operator**. Minimizing $J_{BE}$ is a standard least-squares problem, but $J_{BE}$ may not be minimized to zero due to limited approximation ability.

**Objective 3 -- Projected Bellman Error**:

$$J_{PBE}(w) = \|\hat{v}(w) - M T_\pi(\hat{v}(w))\|_D^2$$

where $M = \Phi(\Phi^T D \Phi)^{-1} \Phi^T D \in \mathbb{R}^{n \times n}$ is the **orthogonal projection matrix** onto the range space of $\Phi$ (the space of all linear approximations).

**Key result**: The TD-Linear algorithm minimizes the **projected Bellman error** $J_{PBE}$, not $J_E$ or $J_{BE}$.

Since $\hat{v}(w)$ lies in the range space of $\Phi$, we can always find $w$ that minimizes $J_{PBE}(w)$ to zero.

**Proof** (Box 8.5): Setting $J_{PBE}(w) = 0$ means $\hat{v}(w) = M T_\pi(\hat{v}(w))$. In the linear case ($\hat{v}(w) = \Phi w$):

$$\Phi w = \Phi(\Phi^T D \Phi)^{-1} \Phi^T D(r_\pi + \gamma P_\pi \Phi w)$$

Since $\Phi$ has full column rank, this simplifies to:

$$w = (\Phi^T D(I - \gamma P_\pi)\Phi)^{-1} \Phi^T D r_\pi = A^{-1}b$$

#### Error Bound (Theorem)

The estimation error of the TD-Linear solution satisfies:

$$\|\Phi w^* - v_\pi\|_D \leq \frac{1}{1-\gamma} \min_w \|\hat{v}(w) - v_\pi\|_D = \frac{1}{1-\gamma} \min_w \sqrt{J_E(w)} \tag{8.32}$$

**Interpretation**: The discrepancy between $\Phi w^*$ and $v_\pi$ is bounded above by the minimum achievable value error scaled by $\frac{1}{1-\gamma}$. This bound is loose when $\gamma \approx 1$ and is mainly of theoretical value.

**Proof sketch** (Box 8.6): Uses the triangle inequality, properties $\|M\|_D = 1$ and $\|P_\pi x\|_D \leq \|x\|_D$ (proven via Jensen's inequality and $d_\pi^T P_\pi = d_\pi^T$).

#### Least-Squares TD (LSTD)

LSTD is an alternative algorithm that also minimizes $J_{PBE}$. Since $w^* = A^{-1}b$ and:

$$A = \mathbb{E}\left[\phi(s_t)(\phi(s_t) - \gamma \phi(s_{t+1}))^T\right], \quad b = \mathbb{E}\left[r_{t+1} \phi(s_t)\right]$$

LSTD directly estimates $A$ and $b$ from samples:

$$\hat{A}_t = \sum_{k=0}^{t-1} \phi(s_k)(\phi(s_k) - \gamma \phi(s_{k+1}))^T, \quad \hat{b}_t = \sum_{k=0}^{t-1} r_{k+1} \phi(s_k) \tag{8.34}$$

Then: $w_t = \hat{A}_t^{-1} \hat{b}_t$

**Note**: The coefficient $1/t$ is omitted since it cancels in $\hat{A}_t^{-1} \hat{b}_t$. A small bias $\sigma I$ is added to $\hat{A}_t$ to ensure invertibility when $t$ is small.

**Recursive inverse update** (to avoid $O(m^3)$ matrix inversion):

$$\hat{A}_{t+1}^{-1} = \hat{A}_t^{-1} - \frac{\hat{A}_t^{-1} \phi(s_t)(\phi(s_t) - \gamma \phi(s_{t+1}))^T \hat{A}_t^{-1}}{1 + (\phi(s_t) - \gamma \phi(s_{t+1}))^T \hat{A}_t^{-1} \phi(s_t)}$$

with initial value $\hat{A}_0^{-1} = \sigma I$.

**Advantages of LSTD**:
- More sample-efficient and converges faster than TD-Linear
- Specifically designed using knowledge of the optimal solution

**Disadvantages of LSTD**:
- Can only estimate state values (not action values)
- Only works with linear approximators
- Higher computational cost per step ($O(m^2)$ for matrix update vs. $O(m)$ for TD)

---

## 8.3 TD Learning of Action Values Based on Function Approximation

This section extends the state value estimation algorithm to **action value estimation**, enabling the search for optimal policies.

### 8.3.1 Sarsa with Function Approximation

Replace state values with action values in (8.13). Let $q_\pi(s,a)$ be approximated by $\hat{q}(s, a, w)$:

$$w_{t+1} = w_t + \alpha_t \left[r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t)\right] \nabla_w \hat{q}(s_t, a_t, w_t) \tag{8.35}$$

**Linear case**: $\hat{q}(s, a, w) = \phi^T(s, a) w$, where $\phi(s, a)$ is the feature vector for state-action pairs. Then $\nabla_w \hat{q}(s, a, w) = \phi(s, a)$.

The value estimation step (8.35) is combined with a policy improvement step to learn optimal policies.

### Algorithm 8.2: Sarsa with Function Approximation

> **Errata applied**: Changed inconsistent epsilon notation to "ϵ" throughout.

```
Initialization: Initial parameter w_0. Initial policy pi_0. alpha_t = alpha > 0 for all t.
              ϵ in (0, 1).
Goal: Learn an optimal policy to lead the agent to the target state from initial state s_0.

For each episode, do
    Generate a_0 at s_0 following pi_0(s_0)
    If s_t (t = 0, 1, 2, ...) is not the target state, do
        Collect experience sample (r_{t+1}, s_{t+1}, a_{t+1}) given (s_t, a_t):
            generate r_{t+1}, s_{t+1} by interacting with environment;
            generate a_{t+1} following pi_t(s_{t+1}).
        Update q-value:
            w_{t+1} = w_t + alpha_t [r_{t+1} + gamma * q_hat(s_{t+1}, a_{t+1}, w_t)
                       - q_hat(s_t, a_t, w_t)] * grad_w q_hat(s_t, a_t, w_t)
        Update policy (ϵ-greedy):
            pi_{t+1}(a|s_t) = 1 - ϵ/|A(s_t)| * (|A(s_t)| - 1)
                              if a = argmax_{a in A(s_t)} q_hat(s_t, a, w_{t+1})
            pi_{t+1}(a|s_t) = ϵ/|A(s_t)|    otherwise
        s_t <- s_{t+1}, a_t <- a_{t+1}
```

**Key notes**:
- Action values are updated only once before switching to policy improvement (similar to tabular Sarsa)
- This implementation finds a good path from a prespecified starting state; with sufficient data, it can find optimal policies for every state

**Illustrative example**: 5x5 grid world with $\gamma = 0.9$, $\epsilon = 0.1$, $r_{\text{boundary}} = r_{\text{forbidden}} = -10$, $r_{\text{target}} = 1$, $\alpha = 0.001$. Using linear Fourier basis of order 5. Both total reward and episode length converge to steady values.

---

### 8.3.2 Q-learning with Function Approximation

The update rule replaces $\hat{q}(s_{t+1}, a_{t+1}, w_t)$ in Sarsa with $\max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t)$:

$$w_{t+1} = w_t + \alpha_t \left[r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t)\right] \nabla_w \hat{q}(s_t, a_t, w_t) \tag{8.36}$$

Can be implemented on-policy or off-policy (same as tabular case).

### Algorithm 8.3: Q-learning with Function Approximation (On-Policy Version)

> **Errata applied**: Changed inconsistent epsilon notation to "ϵ" throughout.

```
Initialization: Initial parameter w_0. Initial policy pi_0. alpha_t = alpha > 0 for all t.
              ϵ in (0, 1).
Goal: Learn an optimal path to lead the agent to the target state from initial state s_0.

For each episode, do
    If s_t (t = 0, 1, 2, ...) is not the target state, do
        Collect experience sample (a_t, r_{t+1}, s_{t+1}) given s_t:
            generate a_t following pi_t(s_t);
            generate r_{t+1}, s_{t+1} by interacting with environment.
        Update q-value:
            w_{t+1} = w_t + alpha_t [r_{t+1} + gamma * max_{a in A(s_{t+1})} q_hat(s_{t+1}, a, w_t)
                       - q_hat(s_t, a_t, w_t)] * grad_w q_hat(s_t, a_t, w_t)
        Update policy (ϵ-greedy):
            pi_{t+1}(a|s_t) = 1 - ϵ/|A(s_t)| * (|A(s_t)| - 1)
                              if a = argmax_{a in A(s_t)} q_hat(s_t, a, w_{t+1})
            pi_{t+1}(a|s_t) = ϵ/|A(s_t)|    otherwise
```

**Illustrative example**: 5x5 grid world with same parameters. Q-learning with linear Fourier basis functions of order 5 successfully learns an optimal policy.

**Important observation**: In Algorithms 8.2 and 8.3, although values are represented as functions, the policy $\pi(a|s)$ is **still represented as a table**, thus still assuming finite state and action spaces. Chapter 9 extends policies to function representation.

---

## 8.4 Deep Q-Learning

**Deep Q-learning** (also called **Deep Q-Network**, DQN) integrates deep neural networks into Q-learning. It is one of the earliest and most successful **deep reinforcement learning** algorithms.

**Note**: The networks do not have to be deep -- for simple tasks like grid worlds, shallow networks with 1-2 hidden layers may suffice.

### 8.4.1 Algorithm Description

#### Objective Function

Deep Q-learning minimizes:

$$J = \mathbb{E}\left[\left(R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w) - \hat{q}(S, A, w)\right)^2\right] \tag{8.37}$$

where $(S, A, R, S')$ are random variables (state, action, reward, next state).

**Interpretation**: This is the **squared Bellman optimality error**. Since the Bellman optimality equation states:

$$q(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a \in \mathcal{A}(S_{t+1})} q(S_{t+1}, a) \mid S_t = s, A_t = a\right]$$

the quantity $R + \gamma \max_a \hat{q}(S', a, w) - \hat{q}(S, A, w)$ should equal zero in expectation when $\hat{q}$ accurately approximates optimal action values.

#### Gradient Computation Challenge

The parameter $w$ appears in both $\hat{q}(S, A, w)$ and $y = R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w)$. Since the optimal $a$ depends on $w$:

$$\nabla_w y \neq \gamma \max_{a \in \mathcal{A}(S')} \nabla_w \hat{q}(S', a, w)$$

**Solution**: Assume $w$ in $y$ is **fixed** (for a short period). This motivates using **two networks**.

#### Technique 1: Two Networks (Main + Target)

Introduce:
- **Main network**: $\hat{q}(s, a, w)$ with parameter $w$
- **Target network**: $\hat{q}(s, a, w_T)$ with parameter $w_T$

The objective becomes:

$$J = \mathbb{E}\left[\left(R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w_T) - \hat{q}(S, A, w)\right)^2\right]$$

When $w_T$ is fixed, the gradient simplifies to:

$$\nabla_w J = -\mathbb{E}\left[\left(R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w_T) - \hat{q}(S, A, w)\right) \nabla_w \hat{q}(S, A, w)\right] \tag{8.38}$$

**Implementation details**:
1. Both networks initialized with the same parameter
2. Each iteration: draw mini-batch $\{(s, a, r, s')\}$ from replay buffer
3. For each sample, compute target value: $y_T = r + \gamma \max_{a \in \mathcal{A}(s')} \hat{q}(s', a, w_T)$
4. Train main network to minimize $\sum (y_T - \hat{q}(s, a, w))^2$ over the mini-batch
5. Main network updated every iteration; target network set to $w_T = w$ every $C$ iterations

**Key difference from non-deep RL**: We use mini-batch training with standard neural network toolkits rather than directly applying (8.36) to update parameters sample-by-sample.

#### Technique 2: Experience Replay

Experience samples are **not** used in the order collected. Instead:
1. Store samples in a **replay buffer** $\mathcal{B} = \{(s, a, r, s')\}$
2. Each training iteration: uniformly draw a mini-batch from $\mathcal{B}$

**Why is experience replay necessary?**

The objective function (8.37) requires specifying the distribution of $(S, A)$. The simplest assumption is **uniform distribution**. However, samples generated sequentially by a behavior policy are correlated and not uniformly distributed. Experience replay breaks this correlation by randomly sampling from the buffer.

**Why uniform distribution specifically?** Because no specific policy is given (we seek the optimal policy), so stationary distribution cannot be used. Uniform is the natural no-prior-knowledge choice.

**Additional benefit**: Each experience sample may be used **multiple times**, increasing data efficiency.

### Algorithm 8.3 (DQN): Deep Q-Learning (Off-Policy Version)

```
Initialization: A main network and a target network with the same initial parameter.
Goal: Learn an optimal target network to approximate optimal action values from
      experience samples generated by a behavior policy pi_b.

Store experience samples generated by pi_b in replay buffer B = {(s, a, r, s')}
For each iteration, do
    Uniformly draw a mini-batch of samples from B
    For each sample (s, a, r, s'), calculate target value:
        y_T = r + gamma * max_{a in A(s')} q_hat(s', a, w_T)
        where w_T is the parameter of the target network
    Update the main network to minimize sum of (y_T - q_hat(s, a, w))^2
        using the mini-batch
    Set w_T = w every C iterations
```

### 8.4.2 Illustrative Examples

#### Example 1: Episode with 1,000 Steps

- **Behavior policy**: Uniform random (equal probability for all actions at all states) -- highly exploratory
- **Replay buffer**: 1,000 experience samples
- **Mini-batch size**: 100
- **Network architecture**: 1 hidden layer with 100 neurons; 3 inputs (normalized row, column, action); 1 output (estimated q-value)
- **Parameters**: $\gamma = 0.9$, $r_{\text{boundary}} = r_{\text{forbidden}} = -10$, $r_{\text{target}} = 1$

**Results**: Loss function converges to zero. State value estimation error converges to zero. The learned greedy policy is optimal.

**Efficiency comparison**: Deep Q-learning needs only 1,000 steps to find the optimal policy, whereas tabular Q-learning required 100,000 steps. Reasons: (1) function approximation's generalization ability, (2) experience samples are reused via replay.

#### Example 2: Episode with Only 100 Steps

- **Mini-batch size**: 50
- **Result**: Loss function converges to zero (network fits training data well), but state value error does **not** converge to zero.
- **Interpretation**: The network fits the given samples but has too few samples to accurately estimate optimal action values. The learned policy is suboptimal.

**Key insight**: Fitting the loss well does not guarantee accurate value estimation when experience data is insufficient.

---

## 8.5 Summary

The chapter's key contributions:

1. **Transition from tabular to function representation**: Values are approximated by parameterized functions $\hat{v}(s, w)$ or $\hat{q}(s, a, w)$, enabling handling of large/continuous state spaces.

2. **Function approximation as optimization**: The core formulation is an optimization problem with an objective function (weighted squared error using stationary distribution).

3. **Stationary distribution**: A new and important concept that describes the long-run behavior of a Markov process under a given policy. Used to weight the objective function and critical for Chapter 9 as well.

4. **Three objective functions**: True value error $J_E$, Bellman error $J_{BE}$, and projected Bellman error $J_{PBE}$. The TD-Linear algorithm minimizes $J_{PBE}$.

5. **Algorithms**: TD with function approximation (Algorithm 8.1), Sarsa with function approximation (Algorithm 8.2), Q-learning with function approximation (Algorithm 8.3), and Deep Q-learning (DQN).

6. **Neural network integration**: Deep Q-learning uses two key techniques -- dual networks (main + target) and experience replay -- to successfully integrate deep learning with RL.

7. **Unification**: The tabular method is a special case of linear function approximation (Box 8.2).

---

## 8.6 Q&A -- Important Clarifications

### Q: What is the difference between tabular and function approximation methods?
**A**: The key difference is how values are **retrieved** and **updated**:
- **Retrieve**: Table reads directly; function requires computing $\hat{v}(s,w)$ (or forward propagation for neural networks).
- **Update**: Table rewrites entries directly; function updates the parameter $w$ indirectly.

### Q: What are the advantages of function approximation?
**A**: Two main advantages from the way values are retrieved and updated:
1. **Storage efficiency**: Only store $w \in \mathbb{R}^m$ instead of $|\mathcal{S}|$ values (where $m \ll |\mathcal{S}|$).
2. **Generalization**: Updating $w$ for one state also affects values of other states, so experience from one state generalizes to others.

### Q: Can we unify tabular and function approximation methods?
**A**: Yes. The tabular method is a special case of linear function approximation with $\phi(s) = e_s$ (standard basis vector). See Box 8.2.

### Q: What is the stationary distribution and why is it important?
**A**: The stationary distribution $d_\pi$ describes the long-term probability of visiting each state under policy $\pi$. It is necessary for defining valid objective functions for both value approximation (this chapter) and policy gradient methods (Chapter 9).

### Q: What are the advantages and disadvantages of linear function approximation?
**A**:
- **Advantages**: Theoretical properties are well-understood; tabular method is a special case; sufficient for simple tasks.
- **Disadvantages**: Limited approximation ability; nontrivial to select appropriate feature vectors.
- **Alternative**: Neural networks serve as universal nonlinear approximators but are harder to analyze theoretically.

### Q: Why does deep Q-learning require experience replay?
**A**: The objective function (8.37) assumes $(S, A)$ is uniformly distributed. Since sequential samples from a behavior policy are correlated and non-uniform, experience replay breaks the correlation by uniformly drawing from the replay buffer. An additional benefit is that samples can be reused, increasing data efficiency.

### Q: Can tabular Q-learning use experience replay?
**A**: Yes. Although not required, tabular Q-learning can use experience replay without problems due to its off-policy nature. It increases sample efficiency through reuse.

### Q: Why does deep Q-learning require two networks?
**A**: The fundamental reason is to simplify gradient computation. Since $w$ appears in both $\hat{q}(S, A, w)$ and the target $R + \gamma \max_a \hat{q}(S', a, w)$, computing the gradient directly is intractable. By fixing $w$ in the target (target network), the gradient becomes tractable (equation 8.38). The target network is updated periodically to track the main network.

### Q: How should neural network parameters be updated?
**A**: Do **not** directly update parameters via equations like (8.36). Instead, follow standard neural network training procedures using mature software toolkits (mini-batch gradient descent via backpropagation).

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| Function approximation | $\hat{v}(s, w) \approx v_\pi(s)$ | 8.1 |
| Parameter vector | $w \in \mathbb{R}^m$ | 8.1 |
| Feature vector | $\phi(s) \in \mathbb{R}^m$ | 8.1 |
| Linear function approximation | $\hat{v}(s,w) = \phi^T(s)w$ | 8.1, 8.2.3 |
| Objective function (general) | $J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S,w))^2]$ | 8.2.1 |
| Stationary distribution | $d_\pi(s)$, left eigenvector of $P_\pi$ for eigenvalue 1 | 8.2.1, Box 8.1 |
| Limiting distribution | $\lim_{k \to \infty} d_k = d_\pi$ | Box 8.1 |
| Probability transition matrix | $P_\pi \in \mathbb{R}^{n \times n}$ | Box 8.1 |
| Irreducible Markov process | All states communicate | Box 8.1 |
| Regular Markov process | $\exists k: P_\pi^k > 0$ elementwise | Box 8.1 |
| Uniform distribution objective | $J(w) = \frac{1}{n}\sum_s (v_\pi(s) - \hat{v}(s,w))^2$ | 8.2.1 |
| Weighted (stationary) objective | $J(w) = \sum_s d_\pi(s)(v_\pi(s) - \hat{v}(s,w))^2$ | 8.2.1 |
| TD with function approximation | $w_{t+1} = w_t + \alpha_t[\delta_t]\nabla_w\hat{v}(s_t,w_t)$ | 8.2.2, Alg. 8.1 |
| MC with function approximation | $w_{t+1} = w_t + \alpha_t(g_t - \hat{v}(s_t,w_t))\nabla_w\hat{v}(s_t,w_t)$ | 8.2.2 |
| TD-Linear | $w_{t+1} = w_t + \alpha_t(r_{t+1} + \gamma\phi^T(s_{t+1})w_t - \phi^T(s_t)w_t)\phi(s_t)$ | 8.2.3 |
| Tabular as special case | $\phi(s) = e_s$, TD-Linear = tabular TD | Box 8.2 |
| Polynomial features | $\phi(s) = [1, x, y, x^2, \ldots]^T$ | 8.2.4 |
| Fourier features | $\phi(s) = [\ldots, \cos(\pi(c_1 x + c_2 y)), \ldots]^T$ | 8.2.4 |
| Feature matrix | $\Phi \in \mathbb{R}^{n \times m}$ | 8.2.5 |
| Stationary distribution matrix | $D = \text{diag}(d_\pi)$ | 8.2.5 |
| Matrix $A$ | $A = \Phi^T D(I - \gamma P_\pi)\Phi$ | 8.2.5, Lemma 8.1 |
| Vector $b$ | $b = \Phi^T D r_\pi$ | 8.2.5, Lemma 8.1 |
| Converged parameter | $w^* = A^{-1}b$ | 8.2.5 |
| Bellman operator | $T_\pi(x) = r_\pi + \gamma P_\pi x$ | 8.2.5 |
| True value error | $J_E(w) = \|\hat{v}(w) - v_\pi\|_D^2$ | 8.2.5 |
| Bellman error | $J_{BE}(w) = \|\hat{v}(w) - T_\pi(\hat{v}(w))\|_D^2$ | 8.2.5 |
| Projected Bellman error | $J_{PBE}(w) = \|\hat{v}(w) - MT_\pi(\hat{v}(w))\|_D^2$ | 8.2.5 |
| Projection matrix | $M = \Phi(\Phi^T D\Phi)^{-1}\Phi^T D$ | 8.2.5 |
| Weighted norm | $\|x\|_D^2 = x^T D x$ | 8.2.5, Box 8.6 |
| Error bound | $\|\Phi w^* - v_\pi\|_D \leq \frac{1}{1-\gamma}\min_w\|\hat{v}(w) - v_\pi\|_D$ | 8.2.5 |
| Least-squares TD (LSTD) | $w_t = \hat{A}_t^{-1}\hat{b}_t$ | 8.2.5 |
| Sarsa with function approx. | Replace $\hat{v}$ with $\hat{q}$ in TD update | 8.3.1, Alg. 8.2 |
| Q-learning with function approx. | Use $\max_a \hat{q}$ instead of $\hat{q}(s_{t+1}, a_{t+1})$ | 8.3.2, Alg. 8.3 |
| Deep Q-learning / DQN | Neural network + experience replay + dual networks | 8.4 |
| Main network | $\hat{q}(s, a, w)$, updated every iteration | 8.4.1 |
| Target network | $\hat{q}(s, a, w_T)$, updated every $C$ iterations | 8.4.1 |
| Experience replay | Uniformly draw mini-batches from replay buffer | 8.4.1 |
| Replay buffer | $\mathcal{B} = \{(s, a, r, s')\}$ | 8.4.1 |
| DQN objective | $J = \mathbb{E}[(R + \gamma\max_a\hat{q}(S',a,w) - \hat{q}(S,A,w))^2]$ | 8.4.1 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| Function approximation $\hat{v}(s,w)$ | Ch 9 (extended to policy function $\pi(a \mid s, \theta)$) |
| Stationary distribution $d_\pi$ | Ch 9 (policy gradient theorem uses $d_\pi$ in the gradient expression) |
| TD with function approximation | Ch 10 (critic component of actor-critic uses this) |
| Feature vector $\phi(s)$ | Ch 9 (feature vectors for policy parameterization) |
| Bellman operator $T_\pi$ | Ch 9, 10 (analysis of value estimation in actor-critic) |
| Projected Bellman error $J_{PBE}$ | Ch 9 (understanding value estimation in policy optimization) |
| Q-learning with function approximation | Ch 10 (Q-value estimation in actor-critic) |
| Deep Q-learning / DQN | Ch 10 (DQN concepts extended in actor-critic with neural networks) |
| Experience replay | Ch 10 (used in deep actor-critic methods) |
| Two-network technique | Ch 10 (target networks in deep actor-critic) |
| Tabular TD (Ch 7) | This chapter: shown to be special case of TD-Linear |
| Stochastic approximation (Ch 6) | This chapter: SGD and RM convergence used throughout |
| Bellman equation (Ch 2) | This chapter: $v_\pi = r_\pi + \gamma P_\pi v_\pi$ used in analysis |
| Bellman optimality equation (Ch 3) | This chapter: motivates DQN objective function |
| Policy iteration / value iteration (Ch 4) | This chapter: policy improvement step in Algs. 8.2, 8.3 |
| $\epsilon$-greedy policies (Ch 7) | This chapter: used for exploration in all algorithms |
