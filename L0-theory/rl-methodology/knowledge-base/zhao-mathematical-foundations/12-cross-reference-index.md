---
type: synthesis
title: Cross-Reference Index and Algorithm Comparison
---

# Cross-Reference Index and Algorithm Comparison

This file provides a comprehensive cross-reference across all chapters of *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer, 2025). Use it to locate concepts, compare algorithms, find theorems, and look up key equations without reading each chapter file individually.

---

## 1. Master Concept Index

| Concept | Defined In | Also Used In |
|---------|-----------|--------------|
| State ($s \in \mathcal{S}$) | Ch 1 | All chapters |
| Action ($a \in \mathcal{A}$) | Ch 1 | All chapters |
| State transition probability $p(s'\|s,a)$ | Ch 1 | Ch 2, 3, 4, 5, 7, 8 |
| Policy $\pi(a\|s)$ | Ch 1 | All chapters |
| Deterministic policy | Ch 1 | Ch 3, 4, 5, 7, 10 |
| Stochastic policy | Ch 1 | Ch 2, 3, 4, 5, 7, 9, 10 |
| Reward $r$ and reward probability $p(r\|s,a)$ | Ch 1 | Ch 2, 3, 4, 5, 7, 8 |
| Trajectory | Ch 1 | Ch 5, 7, 9 |
| Return $G_t$ (discounted cumulative reward) | Ch 1 | Ch 2, 5, 7, 9 |
| Discount rate $\gamma$ | Ch 1 | All chapters |
| Episode | Ch 1 | Ch 5, 7, 9 |
| Absorbing state | Ch 1 | Ch 5, 7 |
| Markov property | Ch 1 | Ch 2, 3 |
| Markov Decision Process (MDP) | Ch 1 | All chapters |
| State value $v_\pi(s)$ | Ch 2 | Ch 3, 4, 5, 6, 7, 8, 9, 10 |
| Action value $q_\pi(s,a)$ | Ch 2 | Ch 3, 4, 5, 7, 8, 9, 10 |
| Bellman equation (elementwise) | Ch 2 | Ch 3, 4, 7, 8 |
| Bellman equation (matrix-vector form) | Ch 2 | Ch 3, 4, 8 |
| Bootstrapping | Ch 2 | Ch 7, 8 |
| Policy evaluation | Ch 2 | Ch 4, 5, 7, 8, 10 |
| Expected reward $r_\pi(s)$ | Ch 2 | Ch 3, 4, 8 |
| State transition matrix $P_\pi$ | Ch 2 | Ch 3, 4, 8, 9, 10 |
| Optimal policy $\pi^*$ | Ch 3 | Ch 4, 5, 7, 8, 9, 10 |
| Optimal state value $v^*(s)$ | Ch 3 | Ch 4, 7, 8 |
| Optimal action value $q^*(s,a)$ | Ch 3 | Ch 4, 7, 8 |
| Bellman optimality equation (BOE) | Ch 3 | Ch 4, 7, 8 |
| Contraction mapping | Ch 3 | Ch 4, 8 |
| Fixed point | Ch 3 | Ch 4, 8 |
| Greedy policy | Ch 3 | Ch 4, 5, 7 |
| Affine reward invariance | Ch 3 | (standalone result) |
| Value iteration | Ch 4 | Ch 5, 7, 8 |
| Policy iteration | Ch 4 | Ch 5, 7, 8, 9, 10 |
| Truncated policy iteration | Ch 4 | Ch 5 |
| Generalized policy iteration (GPI) | Ch 4 | Ch 5, 7, 8, 9, 10 |
| Policy improvement | Ch 4 | Ch 5, 7, 9, 10 |
| Dynamic programming | Ch 4 | Ch 7, 8 |
| Model-based vs model-free | Ch 4, 5 | Ch 7, 8, 9, 10 |
| Mean estimation | Ch 5 | Ch 6, 7 |
| Monte Carlo estimation | Ch 5 | Ch 7, 9 |
| Initial-visit / first-visit / every-visit | Ch 5 | (standalone) |
| Exploring starts | Ch 5 | (standalone) |
| Epsilon-greedy policy | Ch 5 | Ch 7, 8 |
| Soft policy | Ch 5 | Ch 7 |
| Exploration vs exploitation | Ch 5 | Ch 7, 8, 9, 10 |
| Incremental vs non-incremental updates | Ch 6 | Ch 7, 8, 9, 10 |
| Robbins-Monro (RM) algorithm | Ch 6 | Ch 7, 8 |
| Step size conditions ($\sum \alpha_k = \infty$, $\sum \alpha_k^2 < \infty$) | Ch 6 | Ch 7, 8, 9, 10 |
| Stochastic approximation | Ch 6 | Ch 7, 8, 9, 10 |
| Stochastic gradient descent (SGD) | Ch 6 | Ch 8, 9, 10 |
| Batch / mini-batch gradient descent | Ch 6 | Ch 8 |
| TD error $\delta_t$ | Ch 7 | Ch 8, 10 |
| TD target | Ch 7 | Ch 8 |
| TD learning (for state values) | Ch 7 | Ch 8, 10 |
| Sarsa | Ch 7 | Ch 8 |
| Expected Sarsa | Ch 7 | (standalone) |
| n-step Sarsa | Ch 7 | (standalone) |
| Q-learning | Ch 7 | Ch 8 |
| On-policy learning | Ch 7 | Ch 8, 9, 10 |
| Off-policy learning | Ch 7 | Ch 8, 10 |
| Behavior policy vs target policy | Ch 7 | Ch 10 |
| Unified TD viewpoint | Ch 7 | Ch 8 |
| Function approximation | Ch 8 | Ch 9, 10 |
| Feature vector $\phi(s)$ | Ch 8 | Ch 9, 10 |
| Parameter vector $w$ (value) / $\theta$ (policy) | Ch 8, 9 | Ch 10 |
| Linear value approximation | Ch 8 | (standalone) |
| Polynomial features | Ch 8 | (standalone) |
| Fourier features | Ch 8 | (standalone) |
| Neural network approximation | Ch 8 | Ch 10 |
| Stationary distribution $d_\pi$ | Ch 8 | Ch 9, 10 |
| Objective functions ($J_E$, $J_{BE}$, $J_{PBE}$) | Ch 8 | (standalone) |
| Projected Bellman error | Ch 8 | (standalone) |
| Semi-gradient method | Ch 8 | (standalone) |
| Experience replay / replay buffer | Ch 8 | Ch 10 |
| Target network | Ch 8 | Ch 10 |
| DQN (Deep Q-Network) | Ch 8 | (standalone) |
| LSTD (Least-Squares TD) | Ch 8 | (standalone) |
| Parameterized policy $\pi(a\|s,\theta)$ | Ch 9 | Ch 10 |
| Softmax policy | Ch 9 | Ch 10 |
| Average state value $\bar{v}_\pi$ | Ch 9 | Ch 10 |
| Average reward $\bar{r}_\pi$ | Ch 9 | Ch 10 |
| Discounted total probability $\eta_\gamma(s)$ | Ch 9 | Ch 10 |
| Log-derivative trick | Ch 9 | Ch 10 |
| Policy gradient theorem | Ch 9 | Ch 10 |
| REINFORCE | Ch 9 | Ch 10 |
| Poisson equation | Ch 9 | (standalone) |
| Actor-critic structure | Ch 10 | (standalone) |
| Baseline invariance | Ch 10 | (standalone) |
| Advantage function $A_\pi(s,a) = q_\pi(s,a) - v_\pi(s)$ | Ch 10 | (standalone) |
| Importance sampling | Ch 10 | (standalone) |
| Importance weight $\rho_t$ | Ch 10 | (standalone) |
| Deterministic policy $a = \mu(s,\theta)$ | Ch 10 | (standalone) |
| Deterministic policy gradient (DPG) | Ch 10 | (standalone) |
| DDPG | Ch 10 | (standalone) |
| Probability theory (expectation, variance, conditional expectation) | App A | Ch 1, 2, 5, 6, 9 |
| Gradient of expectation | App A | Ch 9, 10 |
| Measure-theoretic probability ($\sigma$-algebra, probability triple) | App B | Ch 6 |
| Martingale / supermartingale / submartingale | App C | Ch 6 |
| Quasimartingale | App C | Ch 6 |
| Convexity and convex functions | App D | Ch 6 |
| Gradient descent convergence theory | App D | Ch 6, 8, 9, 10 |
| Lipschitz continuity | App D | Ch 6 |

---

## 2. Algorithm Comparison Table

| Algorithm | Type | Model Required? | Representation | Update Rule (Core Idea) | Convergence Guarantee | On/Off-Policy |
|-----------|------|----------------|----------------|------------------------|----------------------|---------------|
| **Value Iteration** (Alg 4.1) | Value-based | Yes | Tabular | $v_{k+1}(s) = \max_a [r(s,a) + \gamma \sum_{s'} p(s'\|s,a) v_k(s')]$ | Yes (contraction mapping, Thm 3.3) | N/A (model-based) |
| **Policy Iteration** (Alg 4.2) | Value-based | Yes | Tabular | Alternates: (1) solve $v_\pi = r_\pi + \gamma P_\pi v_\pi$; (2) $\pi' = \text{greedy}(v_\pi)$ | Yes (Thm 4.1; finite steps) | N/A (model-based) |
| **Truncated Policy Iteration** (Alg 4.3) | Value-based | Yes | Tabular | Like policy iteration but truncates policy evaluation to $j$ iterations | Yes | N/A (model-based) |
| **MC Basic** (Alg 5.1) | Value-based | No | Tabular | Estimate $q_\pi(s,a)$ via episode returns; greedy policy update | Yes (LLN, with sufficient episodes) | On-policy |
| **MC Exploring Starts** (Alg 5.2) | Value-based | No | Tabular | Like MC Basic but starts each episode from random $(s,a)$; incremental mean updates | Yes (with exploring starts) | On-policy |
| **MC $\epsilon$-Greedy** (Alg 5.3) | Value-based | No | Tabular | $\epsilon$-greedy policy improvement; no exploring starts needed | Yes (converges to $\epsilon$-optimal) | On-policy |
| **TD Learning** (Sec 7.1) | Value-based | No | Tabular | $v(s_t) \leftarrow v(s_t) + \alpha_t [r_{t+1} + \gamma v(s_{t+1}) - v(s_t)]$ | Yes (Thm 7.1) | On-policy |
| **Sarsa** (Alg 7.1) | Value-based | No | Tabular | $q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha_t [r_{t+1} + \gamma q(s_{t+1},a_{t+1}) - q(s_t,a_t)]$ | Yes (Thm 7.2) | On-policy |
| **Expected Sarsa** (Sec 7.2) | Value-based | No | Tabular | Like Sarsa but TD target uses $\sum_{a'} \pi(a'\|s_{t+1}) q(s_{t+1},a')$ | Yes | On-policy |
| **n-step Sarsa** (Sec 7.3) | Value-based | No | Tabular | Uses $n$-step return; bridges Sarsa ($n=1$) and MC ($n=\infty$) | Yes (under standard conditions) | On-policy |
| **Q-learning** (Algs 7.2, 7.3) | Value-based | No | Tabular | $q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha_t [r_{t+1} + \gamma \max_{a'} q(s_{t+1},a') - q(s_t,a_t)]$ | Yes (under standard conditions) | **Off-policy** |
| **TD-Linear** (Sec 8.2) | Value-based | No | Linear FA | $w \leftarrow w + \alpha [r + \gamma \phi(s')^T w - \phi(s)^T w] \phi(s)$ | Yes (converges to TD fixed point) | On-policy |
| **Sarsa with FA** (Alg 8.2) | Value-based | No | Function approx | Semi-gradient Sarsa with parameterized $\hat{q}(s,a,w)$ | Yes (linear case) | On-policy |
| **Q-learning with FA** (Alg 8.3) | Value-based | No | Function approx | Semi-gradient Q-learning with parameterized $\hat{q}(s,a,w)$ | No (can diverge with nonlinear FA) | **Off-policy** |
| **DQN** (Sec 8.5) | Value-based | No | Neural network | Q-learning + experience replay + target network | No (empirical success) | **Off-policy** |
| **LSTD** (Sec 8.2) | Value-based | No | Linear FA | $w = A^{-1} b$ where $A = \Phi^T D(I - \gamma P_\pi)\Phi$, $b = \Phi^T D r_\pi$ | Yes (direct solution) | On-policy |
| **REINFORCE** (Alg 9.1) | Policy-based | No | Parameterized policy | $\theta \leftarrow \theta + \alpha \gamma^t q_t \nabla_\theta \ln \pi(a_t\|s_t,\theta)$ | Yes (under standard conditions) | On-policy |
| **QAC** (Alg 10.1) | Actor-Critic | No | Parameterized policy + value FA | Actor: policy gradient; Critic: TD-based $q$ estimation | Yes (under standard conditions) | On-policy |
| **A2C** (Alg 10.2) | Actor-Critic | No | Parameterized policy + value FA | Actor: advantage-based gradient; Critic: TD-based $v$ estimation | Yes (lower variance than QAC) | On-policy |
| **Off-policy AC** (Alg 10.3) | Actor-Critic | No | Parameterized policy + value FA | Uses importance sampling ratio $\frac{\pi(a\|s,\theta)}{\beta(a\|s)}$ | Yes (under standard conditions) | **Off-policy** |
| **DPG** (Alg 10.4) | Actor-Critic | No | Deterministic policy + value FA | $\theta \leftarrow \theta + \alpha \nabla_\theta \mu(s,\theta) \nabla_a q(s,a,w)\big\|_{a=\mu}$ | Yes (under standard conditions) | **Off-policy** |
| **DDPG** | Actor-Critic | No | Neural networks (actor + critic) | DPG + experience replay + target networks | No (empirical success) | **Off-policy** |

---

## 3. Theorem and Lemma Index

### Major Theorems

| Theorem | Name | One-Line Statement | Chapter | Key Conditions |
|---------|------|-------------------|---------|----------------|
| Thm 3.1 | Contraction Mapping Theorem | A contraction mapping on a complete metric space has a unique fixed point, reachable by iteration. | Ch 3 | Contraction factor $\gamma < 1$; complete metric space |
| Thm 3.2 | Contraction Property of $f(\mathbf{v})$ | The BOE operator $f$ is a contraction mapping under the infinity norm with factor $\gamma$. | Ch 3 | Discount rate $\gamma \in [0, 1)$ |
| Thm 3.3 | Existence, Uniqueness, and Algorithm | The BOE has a unique solution $v^*$; the iterative algorithm $v_{k+1} = f(v_k)$ converges to $v^*$ for any $v_0$. | Ch 3 | $\gamma < 1$; follows from Thms 3.1 and 3.2 |
| Thm 3.4 | Optimality of $v^*$ and $\pi^*$ | The solution $v^*$ of the BOE is the optimal state value; the greedy policy w.r.t. $v^*$ is optimal. | Ch 3 | BOE solution exists |
| Thm 3.5 | Greedy Optimal Policy | There always exists a deterministic optimal policy; any greedy policy w.r.t. $v^*$ is optimal. | Ch 3 | Finite action space |
| Thm 3.6 | Affine Reward Invariance | Optimal policy is unchanged under affine transformations $r' = \alpha r + \beta$ ($\alpha > 0$) of rewards. | Ch 3 | $\alpha > 0$; optimal values scale accordingly |
| Thm 4.1 | Convergence of Policy Iteration | Policy iteration converges: the sequence of state values $v_{\pi_k}$ converges to $v^*$. | Ch 4 | Finite state and action spaces |
| Thm 6.1 | Robbins-Monro Theorem | The RM algorithm $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$ converges to the root $w^*$ of $g(w) = 0$ a.s. | Ch 6 | $\sum a_k = \infty$, $\sum a_k^2 < \infty$; bounded variance; $g$ has unique root |
| Thm 6.2 | Dvoretzky's Theorem | A stochastic iterative process converges to zero a.s. under contraction and noise conditions. | Ch 6 | Contraction factor; bounded noise variance; step size conditions |
| Thm 6.3 | Extended Dvoretzky's Theorem | Extension to finite sets: convergence of componentwise stochastic iterations. | Ch 6 | Per-component step sizes; contraction; bounded noise |
| Thm 6.4 | SGD Convergence | SGD converges to the minimizer of $J(w) = E[f(w,X)]$ almost surely. | Ch 6 | Convex $J$; unique minimizer; step size conditions; bounded gradient variance |
| Thm 7.1 | Convergence of TD Learning | TD learning converges: $v_t(s) \to v_\pi(s)$ almost surely for all $s$. | Ch 7 | Step size conditions ($\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$); all states visited infinitely often |
| Thm 7.2 | Convergence of Sarsa | Sarsa converges: $q_t(s,a) \to q_\pi(s,a)$ almost surely for all $(s,a)$. | Ch 7 | Step size conditions; all state-action pairs visited infinitely often |
| Thm 9.1 | Policy Gradient Theorem (Master) | $\nabla_\theta J(\theta) = \sum_s \eta(s) \sum_a \nabla_\theta \pi(a\|s,\theta) \, q_\pi(s,a)$ for appropriate $\eta$ and $J$. | Ch 9 | Differentiable $\pi(a\|s,\theta)$; ergodic MDP (for average reward) |
| Thm 9.2 | Gradient of $\bar{v}_\pi^0$ (Discounted) | Gradient of initial-state value uses discounted total probability $\eta_\gamma(s)$. | Ch 9 | Discounted setting; differentiable policy |
| Thm 9.3 | Gradient of $\bar{r}_\pi$ and $\bar{v}_\pi$ (Discounted) | Gradient of average reward/value in discounted case uses stationary distribution $d_\pi$. | Ch 9 | Ergodic MDP; discounted setting |
| Thm 9.4 | Solution of Poisson Equation | The Poisson equation $(I - P_\pi) h = r_\pi - \bar{r}_\pi \mathbf{1}$ has a solution. | Ch 9 | Ergodic Markov chain |
| Thm 9.5 | Gradient of $\bar{r}_\pi$ (Undiscounted) | Gradient in undiscounted case: same form as Thm 9.1 with $\eta = d_\pi$ and $q_\pi$ replaced by differential value. | Ch 9 | Ergodic MDP; $\gamma = 1$ (average reward setting) |
| Thm 10.1 | Off-Policy Policy Gradient | Off-policy gradient uses behavior policy's stationary distribution $d_\beta$ and importance weights. | Ch 10 | Differentiable policy; behavior policy covers target policy |
| Thm 10.2 | Deterministic Policy Gradient (General) | $\nabla_\theta J(\theta) = \sum_s \eta(s) \nabla_\theta \mu(s,\theta) \nabla_a q_\mu(s,a)\big\|_{a=\mu(s)}$ | Ch 10 | Differentiable $\mu$ and $q$; continuous action space |
| Thm 10.3 | DPG (Discounted Case) | Discounted case: $\eta$ is the discounted total probability under the deterministic policy. | Ch 10 | Discounted setting |
| Thm 10.4 | DPG (Undiscounted Case) | Undiscounted case: $\eta$ is the stationary distribution $d_\mu$. | Ch 10 | Ergodic MDP; $\gamma = 1$ |
| Thm C.1 | Convergence of Monotonic Sequences | A bounded, monotonic sequence converges. | App C | Nonincreasing + bounded below (or nondecreasing + bounded above) |
| Thm C.2 | Convergence of Nonmonotonic Sequences | A nonneg. sequence converges if $\sum(x_{k+1} - x_k)^+ < \infty$. | App C | Nonnegative sequence; summable positive increments |
| Thm C.3 | Martingale Convergence Theorem | A submartingale (or supermartingale) converges almost surely. | App C | Sub- or supermartingale property |
| Thm C.4 | Quasimartingale Convergence | A nonneg. quasimartingale converges a.s. if the expected positive variation is summable. | App C | Nonnegative; summable expected positive increments |

### Lemmas and Propositions

| Result | Name | One-Line Statement | Chapter |
|--------|------|-------------------|---------|
| Lemma 4.1 | Policy Improvement | If $\pi'$ is greedy w.r.t. $v_\pi$, then $v_{\pi'} \geq v_\pi$ elementwise. | Ch 4 |
| Prop 4.1 | Value Improvement | In truncated PE, $v^{(j+1)}_{\pi_k} \geq v^{(j)}_{\pi_k}$ elementwise. | Ch 4 |
| Lemma 8.1 | Stationary Distribution Identity | $\Phi^T D P_\pi \Phi = \Phi^T D_\pi \Phi$ where $D$ is the stationary distribution matrix. | Ch 8 |
| Lemma 9.1 | Equivalence of $\bar{v}_\pi$ and $\bar{r}_\pi$ | Maximizing $\bar{v}_\pi$ is equivalent to maximizing $\bar{r}_\pi$: $\bar{v}_\pi = \bar{r}_\pi / (1 - \gamma)$. | Ch 9 |
| Lemma 9.2 | Gradient of $v_\pi(s)$ | $\nabla_\theta v_\pi(s) = \sum_{s'} \sum_{k=0}^{\infty} \gamma^k \Pr(s \to s', k, \pi) \sum_a \nabla_\theta \pi(a\|s',\theta) q_\pi(s',a)$ | Ch 9 |
| Lemma 9.3 | Invertibility | $I_n - P_\pi + \mathbf{1}_n d_\pi^T$ is invertible for ergodic chains. | Ch 9 |
| Lemma 10.1 | Gradient of $v_\mu(s)$ | $\nabla_\theta v_\mu(s) = \sum_{s'} \sum_{k=0}^{\infty} \gamma^k p(s \to s', k, \mu) \nabla_\theta \mu(s',\theta) \nabla_a q_\mu(s',a)\big\|_{a=\mu(s')}$ | Ch 10 |
| Corollary C.1 | Perturbation Bound | If $x_{k+1} \leq x_k + \eta_k$ with $x_k \geq 0$ and $\sum \eta_k < \infty$, then $\{x_k\}$ converges. | App C |

---

## 4. Equation Quick-Reference

The ~20 most important equations in the book, organized by topic.

### Bellman Equations (Ch 2)

**Bellman equation (elementwise form)**:
$$v_\pi(s) = \sum_{a} \pi(a|s) \left[ \sum_{r} p(r|s,a) \, r + \gamma \sum_{s'} p(s'|s,a) \, v_\pi(s') \right], \quad \forall s \in \mathcal{S}$$

**Bellman equation (matrix-vector form)**:
$$v_\pi = r_\pi + \gamma P_\pi v_\pi \quad \Longrightarrow \quad v_\pi = (I - \gamma P_\pi)^{-1} r_\pi$$

**Action value definition**:
$$q_\pi(s, a) = \sum_{r} p(r|s,a) \, r + \gamma \sum_{s'} p(s'|s,a) \, v_\pi(s')$$

**Relationship between $v_\pi$ and $q_\pi$**:
$$v_\pi(s) = \sum_{a} \pi(a|s) \, q_\pi(s, a)$$

### Bellman Optimality Equation (Ch 3)

**BOE (elementwise form)**:
$$v^*(s) = \max_{a} \left[ \sum_{r} p(r|s,a) \, r + \gamma \sum_{s'} p(s'|s,a) \, v^*(s') \right], \quad \forall s \in \mathcal{S}$$

**BOE (compact form)**:
$$v = f(v) \quad \text{where} \quad [f(v)]_s = \max_{a} \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a) \, v(s') \right]$$

### Value Iteration and Policy Iteration (Ch 4)

**Value iteration update**:
$$v_{k+1}(s) = \max_{a} \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a) \, v_k(s') \right]$$

**Policy iteration -- policy evaluation step**:
$$v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$$

**Policy iteration -- policy improvement step**:
$$\pi_{k+1}(s) = \arg\max_{a} \, q_{\pi_k}(s, a)$$

### Stochastic Approximation (Ch 6)

**Robbins-Monro algorithm**:
$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$$

**SGD update**:
$$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$$

### TD Methods (Ch 7)

**TD learning update**:
$$v(s_t) \leftarrow v(s_t) + \alpha_t \underbrace{[r_{t+1} + \gamma v(s_{t+1}) - v(s_t)]}_{\text{TD error } \delta_t}$$

**Sarsa update**:
$$q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha_t [r_{t+1} + \gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$

**Q-learning update**:
$$q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha_t [r_{t+1} + \gamma \max_{a'} q(s_{t+1}, a') - q(s_t, a_t)]$$

### Value Function Approximation (Ch 8)

**Linear value approximation**:
$$\hat{v}(s, w) = \phi(s)^T w$$

**TD-Linear update**:
$$w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \phi(s_{t+1})^T w_t - \phi(s_t)^T w_t] \phi(s_t)$$

### Policy Gradient (Ch 9)

**Policy gradient theorem (unified form)**:
$$\nabla_\theta J(\theta) = \sum_{s} \eta(s) \sum_{a} \nabla_\theta \pi(a|s,\theta) \, q_\pi(s, a)$$

**REINFORCE update**:
$$\theta_{t+1} = \theta_t + \alpha \gamma^t q_t \nabla_\theta \ln \pi(a_t | s_t, \theta_t)$$

### Actor-Critic (Ch 10)

**A2C actor update (with advantage)**:
$$\theta_{t+1} = \theta_t + \alpha_\theta \, \delta_t \, \nabla_\theta \ln \pi(a_t | s_t, \theta_t)$$

**Deterministic policy gradient**:
$$\nabla_\theta J(\theta) = \sum_{s} \eta(s) \, \nabla_\theta \mu(s, \theta) \, \nabla_a q_\mu(s, a)\big|_{a = \mu(s, \theta)}$$

---

## 5. Key Transitions

### Transition 1: Model-Based to Model-Free

| Aspect | Model-Based (Ch 2--4) | Model-Free (Ch 5, 7--10) |
|--------|----------------------|--------------------------|
| **Knowledge required** | $p(r\|s,a)$ and $p(s'\|s,a)$ known | Only experience samples (data) needed |
| **Core algorithms** | Value iteration, policy iteration | MC, TD, Sarsa, Q-learning, REINFORCE, AC |
| **How values are computed** | Exact computation via model | Estimated from sampled trajectories |
| **Key chapter** | Ch 4 (last model-based chapter) | Ch 5 (first model-free chapter) |
| **Bridge concept** | Policy iteration structure (evaluate + improve) is reused in model-free setting | Monte Carlo estimation replaces exact expectation computation |

### Transition 2: Tabular to Function Approximation

| Aspect | Tabular (Ch 1--7) | Function Approximation (Ch 8--10) |
|--------|-------------------|----------------------------------|
| **Value storage** | Table with one entry per state (or state-action pair) | Parameterized function $\hat{v}(s, w) = \phi(s)^T w$ or neural network |
| **Scalability** | Only finite, small state spaces | Large or continuous state spaces |
| **Update target** | Individual table entries | Parameter vector $w$ (affects all states simultaneously) |
| **Convergence** | Generally guaranteed | May diverge with nonlinear FA + off-policy + bootstrapping ("deadly triad") |
| **Key chapter** | Ch 7 (last tabular chapter) | Ch 8 (first FA chapter) |
| **Bridge concept** | Tabular is a special case of linear FA with one-hot features | $\phi(s) = e_s$ recovers tabular |

### Transition 3: Value-Based to Policy-Based to Actor-Critic

| Aspect | Value-Based (Ch 2--8) | Policy-Based (Ch 9) | Actor-Critic (Ch 10) |
|--------|----------------------|---------------------|---------------------|
| **What is learned** | Value function ($v$ or $q$); policy derived via greedy | Policy parameters $\theta$ directly | Both: policy $\theta$ (actor) and value $w$ (critic) |
| **Policy representation** | Implicit (greedy w.r.t. values) | Explicit $\pi(a\|s,\theta)$ | Explicit $\pi(a\|s,\theta)$ |
| **Optimization target** | Bellman equation / BOE | Scalar metric $J(\theta)$ via gradient ascent | Scalar metric $J(\theta)$ via gradient ascent |
| **Strengths** | Well-understood convergence; natural for discrete actions | Handles continuous actions; directly optimizes policy | Lower variance than pure policy gradient; handles continuous actions |
| **Weaknesses** | Difficult with continuous actions; indirect policy | High variance (MC-based gradient estimates) | More complex; two sets of parameters to tune |
| **Key algorithms** | VI, PI, TD, Sarsa, Q-learning, DQN | REINFORCE | QAC, A2C, Off-policy AC, DPG, DDPG |

### Transition 4: On-Policy to Off-Policy

| Aspect | On-Policy | Off-Policy |
|--------|-----------|------------|
| **Definition** | Behavior policy = target policy (agent learns about the policy it is currently following) | Behavior policy $\neq$ target policy (agent learns about one policy using data from another) |
| **Key algorithms** | Sarsa, Expected Sarsa, REINFORCE, QAC, A2C | Q-learning, DQN, Off-policy AC, DPG, DDPG |
| **Data reuse** | Limited (data generated under old policy becomes stale) | Can reuse data from any behavior policy (experience replay) |
| **Exploration** | Must balance exploration within the learned policy ($\epsilon$-greedy) | Can use an exploratory behavior policy while learning a greedy target policy |
| **Convergence** | Generally more stable | Can diverge with function approximation ("deadly triad") |
| **Correction mechanism** | None needed | Importance sampling ratios $\frac{\pi(a\|s)}{\beta(a\|s)}$ (Ch 10) or implicit correction (Q-learning max) |
| **Key chapter** | Ch 7 (Sarsa), Ch 9 (REINFORCE) | Ch 7 (Q-learning), Ch 8 (DQN), Ch 10 (off-policy AC, DPG) |
