---
chapter: 7
title: Temporal-Difference Methods
key_topics: [TD learning, TD error, TD target, Bellman expectation equation, Sarsa, Expected Sarsa, n-step Sarsa, Q-learning, on-policy, off-policy, behavior policy, target policy, bootstrapping, incremental update, convergence analysis, generalized policy iteration, epsilon-greedy policy, unified TD framework]
depends_on: [1, 2, 3, 4, 5, 6]
required_by: [8, 9, 10]
---

# Chapter 7: Temporal-Difference Methods

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 7, pp. 125-150
> Supplemented by: Lecture slides L7 (55 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter introduces **temporal-difference (TD) methods**, which are among the most well-known algorithms in reinforcement learning. Like Monte Carlo (MC) learning (Chapter 5), TD learning is **model-free**. However, TD learning has key advantages due to its **incremental** (online) form. With the stochastic approximation foundation from Chapter 6, TD algorithms can be understood as **special stochastic approximation algorithms** for solving the Bellman equation or Bellman optimality equation.

**Position in the book**: Chapter 7 is the first chapter in Part 2 (Algorithms/Methods). It bridges the fundamental tools (Chapters 1-6) to the more advanced methods (Chapters 8-10). TD methods introduced here use tabular representation; Chapter 8 extends them to function approximation.

**Chapter overview** -- the algorithms introduced and their relationships:
- **Section 7.1**: Basic TD algorithm for estimating **state values** of a given policy. This is the foundation for all other TD algorithms in this chapter.
- **Section 7.2**: **Sarsa** algorithm for estimating **action values** of a given policy. Obtained from the basic TD algorithm by replacing state value estimation with action value estimation. Can be combined with policy improvement to find optimal policies.
- **Section 7.3**: **n-step Sarsa**, a generalization of Sarsa. Sarsa ($n=1$) and MC learning ($n=\infty$) are two special/extreme cases.
- **Section 7.4**: **Q-learning** for directly estimating **optimal action values**. Unlike the others, Q-learning solves the Bellman optimality equation and is **off-policy**.
- **Section 7.5**: A **unified viewpoint** showing all TD algorithms (and MC) share a common expression differing only in the TD target.

---

## Motivating Examples (from Lecture Slides)

The lecture slides motivate TD algorithms through three progressively complex stochastic problems, all solvable by the Robbins-Monro (RM) algorithm:

**Problem 1 (Mean estimation)**: Calculate $w = E[X]$ from i.i.d. samples $\{x\}$.
- Define $g(w) = w - E[X]$, noisy observation $\tilde{g}(w, \eta) = w - x$.
- RM algorithm: $w_{k+1} = w_k - \alpha_k(w_k - x_k)$.

**Problem 2 (Mean of a function)**: Estimate $w = E[v(X)]$ from samples $\{x\}$.
- Define $g(w) = w - E[v(X)]$, noisy observation $\tilde{g}(w, \eta) = w - v(x)$.
- RM algorithm: $w_{k+1} = w_k - \alpha_k[w_k - v(x_k)]$.

**Problem 3 (Key motivating form)**: Calculate $w = E[R + \gamma v(X)]$ where $R, X$ are random variables, $\gamma$ is a constant, and $v(\cdot)$ is a function.
- Define $g(w) = w - E[R + \gamma v(X)]$, noisy observation $\tilde{g}(w, \eta) = w - [r + \gamma v(x)]$.
- RM algorithm: $w_{k+1} = w_k - \alpha_k[w_k - (r_k + \gamma v(x_k))]$.

**Key insight**: Problem 3 has the same structure as TD algorithms. All three are root-finding problems solved by the RM algorithm, each progressively closer to the TD update form.

---

## 7.1 TD Learning of State Values

TD learning in this section refers to a specific classic algorithm for estimating state values. It is the most basic TD algorithm and is fundamental for understanding all other TD algorithms in this chapter.

### 7.1.1 Algorithm Description

**Problem statement**: Given a policy $\pi$, estimate $v_\pi(s)$ for all $s \in \mathcal{S}$.

**Data**: Experience samples $(s_0, r_1, s_1, \ldots, s_t, r_{t+1}, s_{t+1}, \ldots)$ generated following $\pi$.

**The TD algorithm**:

$$v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)\Big[v_t(s_t) - \big(r_{t+1} + \gamma v_t(s_{t+1})\big)\Big] \tag{7.1}$$

$$v_{t+1}(s) = v_t(s), \quad \text{for all } s \neq s_t \tag{7.2}$$

where $t = 0, 1, 2, \ldots$. Here:
- $v_t(s_t)$ is the estimate of $v_\pi(s_t)$ at time $t$
- $\alpha_t(s_t)$ is the learning rate for $s_t$ at time $t$

**Important note**: At time $t$, only the value of the visited state $s_t$ is updated. The values of all unvisited states $s \neq s_t$ remain unchanged as shown in (7.2). Equation (7.2) is often omitted for simplicity but is essential for mathematical completeness.

### Derivation from the Bellman Equation (Box 7.1)

**Step 1**: Recall the definition of state value:

$$v_\pi(s) = E\big[R_{t+1} + \gamma G_{t+1} \mid S_t = s\big], \quad s \in \mathcal{S} \tag{7.3}$$

**Step 2**: Rewrite using the Bellman expectation equation. Since $E[G_{t+1}|S_t = s] = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) v_\pi(s') = E[v_\pi(S_{t+1})|S_t = s]$, we obtain:

$$v_\pi(s) = E\big[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s\big], \quad s \in \mathcal{S} \tag{7.4}$$

This is the **Bellman expectation equation**, another expression of the Bellman equation that is an important tool for designing and analyzing TD algorithms.

**Step 3**: Apply the Robbins-Monro algorithm to solve (7.4). For state $s_t$, define:

$$g(v_\pi(s_t)) \doteq v_\pi(s_t) - E\big[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s_t\big]$$

Then (7.4) is equivalent to $g(v_\pi(s_t)) = 0$. The noisy observation is:

$$\tilde{g}(v_\pi(s_t)) = v_\pi(s_t) - \big(r_{t+1} + \gamma v_\pi(s_{t+1})\big) = \underbrace{g(v_\pi(s_t))}_{\text{true value}} + \underbrace{E[\cdot] - (r_{t+1} + \gamma v_\pi(s_{t+1}))}_{\eta \text{ (noise)}}$$

The RM algorithm for solving $g(v_\pi(s_t)) = 0$ is:

$$v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)\big[v_t(s_t) - (r_{t+1} + \gamma v_\pi(s_{t+1}))\big] \tag{7.5}$$

**Two modifications** to go from (7.5) to the actual TD algorithm (7.1):
1. **Sequential samples**: The RM algorithm requires i.i.d. samples $\{(s, r_k, s'_k)\}$ for a fixed $s$. Modification: use sequential samples $\{(s_t, r_{t+1}, s_{t+1})\}$ from the episode so the algorithm can utilize sequential experience.
2. **Replacing true values**: The RM algorithm uses $v_\pi(s_{t+1})$, the true value. Modification: replace with the current estimate $v_t(s_{t+1})$ since $v_\pi$ is unknown. This replacement still ensures convergence (proven in Theorem 7.1).

### 7.1.2 Property Analysis

The TD algorithm (7.1) can be annotated as:

$$\underbrace{v_{t+1}(s_t)}_{\text{new estimate}} = \underbrace{v_t(s_t)}_{\text{current estimate}} - \alpha_t(s_t) \underbrace{\Big(\underbrace{v_t(s_t) - \underbrace{\big(r_{t+1} + \gamma v_t(s_{t+1})\big)}_{\text{TD target } \bar{v}_t}}_{\text{TD error } \delta_t}\Big)} \tag{7.6}$$

#### TD Target

$$\bar{v}_t \doteq r_{t+1} + \gamma v_t(s_{t+1})$$

**Why is $\bar{v}_t$ called the TD target?** Because $\bar{v}_t$ is the target value that the algorithm drives $v(s_t)$ toward.

**Proof**: Subtracting $\bar{v}_t$ from both sides of (7.6):

$$v_{t+1}(s_t) - \bar{v}_t = (1 - \alpha_t(s_t))(v_t(s_t) - \bar{v}_t)$$

Taking absolute values:

$$|v_{t+1}(s_t) - \bar{v}_t| = |1 - \alpha_t(s_t)| \cdot |v_t(s_t) - \bar{v}_t|$$

Since $\alpha_t(s_t)$ is a small positive number, $0 < 1 - \alpha_t(s_t) < 1$, so:

$$|v_{t+1}(s_t) - \bar{v}_t| < |v_t(s_t) - \bar{v}_t|$$

This inequality shows that the new value $v_{t+1}(s_t)$ is **closer** to $\bar{v}_t$ than the old value $v_t(s_t)$.

#### TD Error

$$\delta_t \doteq v_t(s_t) - \bar{v}_t = v_t(s_t) - \big(r_{t+1} + \gamma v_t(s_{t+1})\big)$$

**Interpretation 1 -- Temporal difference**: $\delta_t$ reflects the discrepancy between two time steps $t$ and $t+1$.

**Interpretation 2 -- Estimation accuracy**: The TD error is zero in the expectation sense when the state value estimate is accurate. When $v_t = v_\pi$:

$$E[\delta_t | S_t = s_t] = E\big[v_\pi(S_t) - (R_{t+1} + \gamma v_\pi(S_{t+1})) \mid S_t = s_t\big] = v_\pi(s_t) - E\big[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s_t\big] = 0$$

(by (7.3)). Therefore, if $\delta_t$ is nonzero, then $v_t \neq v_\pi$.

**Interpretation 3 -- Innovation**: The TD error represents **new information** obtained from the experience sample $(s_t, r_{t+1}, s_{t+1})$. The fundamental idea of TD learning is to correct the current estimate based on newly obtained information. This concept of innovation is fundamental in many estimation problems such as Kalman filtering.

#### Additional Properties

- The TD algorithm in (7.1) can **only estimate state values** of a given policy. It does not estimate action values and does not search for optimal policies.
- To find optimal policies, action values must be computed (Section 7.2), followed by policy improvement.
- This basic algorithm is fundamental for understanding the more complex TD algorithms in this chapter.

### TD Learning vs. MC Learning Comparison

| Property | TD Learning | MC Learning |
|---|---|---|
| **Update timing** | **Incremental/online**: Can update state/action values immediately after receiving an experience sample | **Non-incremental/offline**: Must wait until an entire episode has been completely collected (needs to compute discounted return) |
| **Task types** | **Episodic and continuing tasks**: Since TD is incremental, it handles both task types | **Episodic tasks only**: Since MC is non-incremental, it requires episodes that terminate after finitely many steps |
| **Bootstrapping** | **Bootstrapping**: Update of a state/action value relies on the previous estimate of this value; requires an initial guess | **Non-bootstrapping**: Can directly estimate state/action values without initial guesses |
| **Estimation variance** | **Low variance**: Fewer random variables involved. E.g., Sarsa requires samples of only three random variables: $R_{t+1}, S_{t+1}, A_{t+1}$ | **High variance**: Many random variables involved. To estimate $q_\pi(s_t, a_t)$, need samples of $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$ If episode length is $L$ and each state has $|\mathcal{A}|$ actions, there are $|\mathcal{A}|^L$ possible episodes under a soft policy |

### 7.1.3 Convergence Analysis

**Theorem 7.1 (Convergence of TD learning)**: Given a policy $\pi$, by the TD algorithm in (7.1), $v_t(s)$ converges almost surely to $v_\pi(s)$ as $t \to \infty$ for all $s \in \mathcal{S}$ if:

$$\sum_t \alpha_t(s) = \infty \quad \text{and} \quad \sum_t \alpha_t^2(s) < \infty \quad \text{for all } s \in \mathcal{S}$$

**Remarks on the learning rate conditions**:
1. The conditions must hold for **all** $s \in \mathcal{S}$. At time $t$, $\alpha_t(s) > 0$ if $s$ is being visited; $\alpha_t(s) = 0$ otherwise. The condition $\sum_t \alpha_t(s) = \infty$ requires every state to be visited an infinite (or sufficiently many) number of times. This requires either **exploring starts** or an **exploratory policy**.
2. In practice, $\alpha_t$ is often selected as a **small positive constant**. In this case, $\sum_t \alpha_t^2(s) < \infty$ is no longer valid. With constant $\alpha$, the algorithm still converges **in the sense of expectation**.

#### Proof Sketch (Box 7.2)

The proof is based on Theorem 6.3 (Chapter 6). Define the estimation error:

$$\Delta_t(s) \doteq v_t(s) - v_\pi(s)$$

**For $s = s_t$**: From the TD algorithm:

$$\Delta_{t+1}(s) = (1 - \alpha_t(s))\Delta_t(s) + \alpha_t(s)\underbrace{\big(r_{t+1} + \gamma v_t(s_{t+1}) - v_\pi(s)\big)}_{\eta_t(s)} \tag{7.9}$$

**For $s \neq s_t$**: $\Delta_{t+1}(s) = \Delta_t(s)$, which has the same form with $\alpha_t(s) = 0$ and $\eta_t(s) = 0$.

**Unified expression**: $\Delta_{t+1}(s) = (1 - \alpha_t(s))\Delta_t(s) + \alpha_t(s)\eta_t(s)$

This matches the stochastic process in Theorem 6.3. The three conditions are verified:

1. **Condition 1** (learning rate): Valid by assumption in Theorem 7.1.
2. **Condition 2** ($\|E[\eta_t(s) | H_t]\|_\infty \leq \gamma \|\Delta_t(s)\|_\infty$):
   - For $s \neq s_t$: $\eta_t(s) = 0$, so $|E[\eta_t(s)]| = 0 \leq \gamma \|\Delta_t(s)\|_\infty$.
   - For $s = s_t$: Using $v_\pi(s_t) = E[r_{t+1} + \gamma v_\pi(s_{t+1}) | s_t]$:
     $$E[\eta_t(s)] = \gamma E[v_t(s_{t+1}) - v_\pi(s_{t+1}) | s_t] = \gamma \sum_{s' \in \mathcal{S}} p(s'|s_t)[v_t(s') - v_\pi(s')]$$
     Therefore $|E[\eta_t(s)]| \leq \gamma \max_{s'} |v_t(s') - v_\pi(s')| = \gamma \|\Delta_t(s)\|_\infty$.
3. **Condition 3** (bounded variance): Since $r_{t+1}$ is bounded, $\text{var}[\eta_t(s)|H_t]$ is bounded.

Since all three conditions are satisfied, Theorem 6.3 guarantees convergence. (Proof inspired by [32].)

---

## 7.2 TD Learning of Action Values: Sarsa

The TD algorithm from Section 7.1 can only estimate state values. **Sarsa** directly estimates **action values**, which is important because action values can be combined with policy improvement to learn optimal policies.

### 7.2.1 Algorithm Description

**Problem statement**: Given a policy $\pi$, estimate the action values $q_\pi(s, a)$ for all $(s, a)$.

**Data**: Experience samples generated following $\pi$: $(s_0, a_0, r_1, s_1, a_1, \ldots, s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}, \ldots)$.

**The Sarsa algorithm**:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\Big[q_t(s_t, a_t) - \big(r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})\big)\Big] \tag{7.12}$$

$$q_{t+1}(s, a) = q_t(s, a), \quad \text{for all } (s, a) \neq (s_t, a_t)$$

where $t = 0, 1, 2, \ldots$ and $\alpha_t(s_t, a_t)$ is the learning rate. At time $t$, only the q-value of the visited pair $(s_t, a_t)$ is updated.

### Key Properties of Sarsa

**Why "Sarsa"?** Each iteration requires $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ -- **S**tate-**A**ction-**R**eward-**S**tate-**A**ction. The algorithm was first proposed in [35] and named by [3].

**Relationship to TD learning**: Sarsa is obtained from the basic TD algorithm (7.1) by replacing state value estimation $v(s)$ with action value estimation $q(s, a)$. It is the action-value version of the TD algorithm.

**What Sarsa solves mathematically**: Sarsa is a stochastic approximation algorithm for solving the Bellman equation expressed in terms of action values:

$$q_\pi(s, a) = E\big[R + \gamma q_\pi(S', A') \mid s, a\big], \quad \text{for all } (s, a) \tag{7.13}$$

#### Proof that (7.13) is the Bellman Equation (Box 7.3)

The Bellman equation in terms of action values (Section 2.8.2) is:

$$q_\pi(s, a) = \sum_r r \, p(r|s,a) + \gamma \sum_{s'} p(s'|s,a) \sum_{a'} q_\pi(s', a') \pi(a'|s') \tag{7.14}$$

Since $p(s', a'|s, a) = p(s'|s,a) \cdot p(a'|s', s, a) = p(s'|s,a) \cdot \pi(a'|s')$ (by conditional independence), (7.14) becomes:

$$q_\pi(s, a) = \sum_r r \, p(r|s,a) + \gamma \sum_{s'} \sum_{a'} q_\pi(s', a') p(s', a'|s, a)$$

By the definition of expected value, this is equivalent to (7.13). Hence (7.13) is the Bellman equation.

### Convergence of Sarsa

**Theorem 7.2 (Convergence of Sarsa)**: Given a policy $\pi$, by the Sarsa algorithm in (7.12), $q_t(s, a)$ converges almost surely to $q_\pi(s, a)$ as $t \to \infty$ for all $(s, a)$ if:

$$\sum_t \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_t \alpha_t^2(s, a) < \infty \quad \text{for all } (s, a)$$

The proof is similar to that of Theorem 7.1. The condition $\sum_t \alpha_t(s, a) = \infty$ requires that **every state-action pair** must be visited an infinite (or sufficiently many) number of times. At time $t$, $\alpha_t(s, a) > 0$ if $(s, a) = (s_t, a_t)$; otherwise $\alpha_t(s, a) = 0$.

### 7.2.2 Optimal Policy Learning via Sarsa

Sarsa alone can only estimate action values of a given policy. To find optimal policies, combine it with a **policy improvement** step. The combination is also commonly called Sarsa.

**Algorithm 7.1: Optimal policy learning by Sarsa**

```
Initialization:
  alpha_t(s,a) = alpha > 0 for all (s,a) and all t
  epsilon in (0, 1)
  Initial q_0(s,a) for all (s,a)
  Initial epsilon-greedy policy pi_0 derived from q_0

Goal: Learn an optimal policy from initial state s_0 to target state.

For each episode, do:
  Generate a_0 at s_0 following pi_0(s_0)
  If s_t (t = 0, 1, 2, ...) is not the target state, do:

    Collect experience sample (r_{t+1}, s_{t+1}, a_{t+1}) given (s_t, a_t):
      generate r_{t+1}, s_{t+1} by interacting with the environment
      generate a_{t+1} following pi_t(s_{t+1})

    Update q-value for (s_t, a_t):
      q_{t+1}(s_t, a_t) = q_t(s_t, a_t)
        - alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + gamma * q_t(s_{t+1}, a_{t+1}))]

    Update policy for s_t (epsilon-greedy):
      pi_{t+1}(a|s_t) = 1 - epsilon/(|A(s_t)|) * (|A(s_t)| - 1)
                         if a = argmax_a q_{t+1}(s_t, a)
      pi_{t+1}(a|s_t) = epsilon / |A(s_t)|   otherwise

    s_t <- s_{t+1},  a_t <- a_{t+1}
```

**Key observations about Algorithm 7.1**:
- Each iteration has **two steps**: (1) update the q-value of the visited state-action pair; (2) update the policy to an $\epsilon$-greedy one.
- The q-value update only affects the single state-action pair visited at time $t$. The policy of $s_t$ is immediately updated afterward.
- The policy is **not fully evaluated** before being updated -- this is the idea of **generalized policy iteration** (GPI).
- The policy is $\epsilon$-greedy to maintain **exploration**.

### Simulation Example (Sarsa)

**Setup**: 5x5 grid world. All episodes start from the top-left state and terminate at the target state (bottom-right). Goal: find an optimal path from start to target.
- Rewards: $r_{\text{target}} = 0$, $r_{\text{forbidden}} = r_{\text{boundary}} = -10$, $r_{\text{other}} = -1$
- Learning rate: $\alpha = 0.1$, exploration: $\epsilon = 0.1$
- Initial values: $q_0(s, a) = 0$ for all $(s, a)$
- Initial policy: uniform distribution $\pi_0(a|s) = 0.2$ for all $s, a$

**Results**:
- **Learned policy**: Successfully leads to the target state from the starting state. However, policies of some states that are not well explored may not be optimal.
- **Total reward per episode**: Increases gradually as the policy improves. Initial episodes have large negative rewards due to the poor initial policy.
- **Episode length**: Decreases gradually. Initial episodes involve many detours. Occasional abrupt increases occur because the $\epsilon$-greedy policy sometimes takes non-optimal actions. Using **decaying $\epsilon$** can mitigate this.

### Expected Sarsa (Box 7.4)

**Expected Sarsa** is a variant of Sarsa:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\Big[q_t(s_t, a_t) - \big(r_{t+1} + \gamma E[q_t(s_{t+1}, A)]\big)\Big]$$

$$q_{t+1}(s, a) = q_t(s, a), \quad \text{for all } (s, a) \neq (s_t, a_t)$$

where:

$$E[q_t(s_{t+1}, A)] = \sum_a \pi_t(a|s_{t+1}) q_t(s_{t+1}, a) \doteq v_t(s_{t+1})$$

**Comparison with Sarsa**: The only difference is the TD target:
- **Expected Sarsa** TD target: $r_{t+1} + \gamma E[q_t(s_{t+1}, A)]$
- **Sarsa** TD target: $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$

**Advantage**: Expected Sarsa reduces estimation variance by eliminating the random variable $A_{t+1}$. The random variables are reduced from $\{S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}\}$ (Sarsa) to $\{S_t, A_t, R_{t+1}, S_{t+1}\}$ (Expected Sarsa). The trade-off is slightly increased computational complexity for computing the expectation.

**What Expected Sarsa solves**: It is a stochastic approximation algorithm for:

$$q_\pi(s, a) = E\Big[R_{t+1} + \gamma E[q_\pi(S_{t+1}, A_{t+1}) | S_{t+1}] \;\Big|\; S_t = s, A_t = a\Big] \tag{7.15}$$

This is another expression of the Bellman equation. Substituting $E[q_\pi(S_{t+1}, A_{t+1}) | S_{t+1}] = \sum_{A'} q_\pi(S_{t+1}, A')\pi(A'|S_{t+1}) = v_\pi(S_{t+1})$ into (7.15) gives $q_\pi(s, a) = E[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a]$, which is clearly the Bellman equation.

---

## 7.3 TD Learning of Action Values: n-step Sarsa

n-step Sarsa is an extension of Sarsa that **unifies Sarsa and MC learning** as two extreme cases.

### Decompositions of the Discounted Return

Recall the action value definition:

$$q_\pi(s, a) = E[G_t | S_t = s, A_t = a] \tag{7.16}$$

where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$

The return $G_t$ can be decomposed in different ways:

| Decomposition | Expression | Corresponding Algorithm |
|---|---|---|
| $G_t^{(1)}$ | $R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1})$ | **Sarsa** ($n = 1$) |
| $G_t^{(2)}$ | $R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, A_{t+2})$ | 2-step Sarsa |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $G_t^{(n)}$ | $R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n})$ | **n-step Sarsa** |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $G_t^{(\infty)}$ | $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ | **MC learning** ($n = \infty$) |

**Critical note**: $G_t = G_t^{(1)} = G_t^{(2)} = G_t^{(n)} = G_t^{(\infty)}$. The superscripts merely indicate different decomposition structures; they all equal the same $G_t$.

### Algorithms for Different $n$

**When $n = 1$ (Sarsa)**:

$$q_\pi(s, a) = E[G_t^{(1)} | s, a] = E[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | s, a]$$

The stochastic approximation algorithm:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\Big[q_t(s_t, a_t) - \big(r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})\big)\Big]$$

This is the Sarsa algorithm in (7.12).

**When $n = \infty$ (MC learning)**:

$$q_\pi(s, a) = E[G_t^{(\infty)} | s, a] = E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots | s, a]$$

The algorithm: $q_{t+1}(s_t, a_t) = g_t \doteq r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$, where $g_t$ is a sample of $G_t$. This is the MC learning algorithm -- approximating the action value using the discounted return of an episode starting from $(s_t, a_t)$.

**For general $n$ (n-step Sarsa)**:

$$q_\pi(s, a) = E[G_t^{(n)} | s, a] = E[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) | s, a]$$

The algorithm:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\Big[q_t(s_t, a_t) - \big(r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_t(s_{t+n}, a_{t+n})\big)\Big] \tag{7.17}$$

### Implementation Considerations

To implement n-step Sarsa, we need $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}, \ldots, r_{t+n}, s_{t+n}, a_{t+n})$. Since $(r_{t+n}, s_{t+n}, a_{t+n})$ has not been collected at time $t$, we must **wait until time $t + n$** to update the q-value of $(s_t, a_t)$:

$$q_{t+n}(s_t, a_t) = q_{t+n-1}(s_t, a_t) - \alpha_{t+n-1}(s_t, a_t)\Big[q_{t+n-1}(s_t, a_t) - \big(r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_{t+n-1}(s_{t+n}, a_{t+n})\big)\Big]$$

### Bias-Variance Trade-off

Since n-step Sarsa includes Sarsa and MC learning as extreme cases, its performance is a blend:

| $n$ value | Behavior | Variance | Bias |
|---|---|---|---|
| **Small $n$** (close to Sarsa) | Close to Sarsa | Relatively **low** variance | Relatively **large** bias (due to initial guess) |
| **Large $n$** (close to MC) | Close to MC learning | Relatively **high** variance | **Small** bias |

n-step Sarsa is for **policy evaluation**. It must be combined with policy improvement to learn optimal policies (implementation is similar to that of Sarsa).

---

## 7.4 TD Learning of Optimal Action Values: Q-learning

Q-learning is one of the most classic reinforcement learning algorithms. Unlike Sarsa (which estimates action values of a **given** policy and needs a separate policy improvement step), Q-learning can **directly** estimate optimal action values and find optimal policies.

### 7.4.1 Algorithm Description

**The Q-learning algorithm**:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\bigg[q_t(s_t, a_t) - \Big(r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})} q_t(s_{t+1}, a)\Big)\bigg] \tag{7.18}$$

$$q_{t+1}(s, a) = q_t(s, a), \quad \text{for all } (s, a) \neq (s_t, a_t)$$

where $t = 0, 1, 2, \ldots$. Here $q_t(s_t, a_t)$ is the estimate of the **optimal** action value and $\alpha_t(s_t, a_t)$ is the learning rate.

**Comparison with Sarsa**: Q-learning and Sarsa differ only in their TD targets:

| Algorithm | TD Target | Required data per step |
|---|---|---|
| **Q-learning** | $r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$ | $(s_t, a_t, r_{t+1}, s_{t+1})$ |
| **Sarsa** | $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$ | $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ |

Note that Q-learning does **not** need $a_{t+1}$.

### What Q-learning Solves Mathematically

Q-learning is a stochastic approximation algorithm for solving:

$$q(s, a) = E\bigg[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \;\bigg|\; S_t = s, A_t = a\bigg] \tag{7.19}$$

This is the **Bellman optimality equation** expressed in terms of action values.

#### Proof that (7.19) is the Bellman Optimality Equation (Box 7.5)

By the definition of expectation, (7.19) can be rewritten as:

$$q(s, a) = \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) \max_{a \in \mathcal{A}(s')} q(s', a)$$

Taking the maximum over $a \in \mathcal{A}(s)$ on both sides and denoting $v(s) \doteq \max_{a \in \mathcal{A}(s)} q(s, a)$:

$$v(s) = \max_{a \in \mathcal{A}(s)} \bigg[\sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s')\bigg] = \max_\pi \sum_{a \in \mathcal{A}(s)} \pi(a|s) \bigg[\sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s')\bigg]$$

This is clearly the Bellman optimality equation in terms of state values (Chapter 3).

**Convergence**: Similar to Theorem 7.1, Q-learning converges under the same learning rate conditions.

### 7.4.2 Off-policy vs. On-policy

Two policies exist in any reinforcement learning task:
- **Behavior policy** ($\pi_b$): Used to **generate** experience samples.
- **Target policy** ($\pi_T$): Constantly **updated** toward an optimal policy.

| Concept | Definition |
|---|---|
| **On-policy** | Behavior policy **is the same as** the target policy |
| **Off-policy** | Behavior policy **is different from** the target policy |

**Advantage of off-policy learning**: Can learn optimal policies from experience generated by **any** other policy (e.g., a human operator's policy, or a highly exploratory policy). This is beneficial for generating episodes that visit every state-action pair sufficiently many times.

#### Why Sarsa is On-policy

Sarsa has two steps per iteration:
1. **Evaluate** policy $\pi$ by solving its Bellman equation using samples **generated by $\pi$**. So $\pi$ is the behavior policy.
2. **Improve** the policy based on estimated values. So $\pi$ is the target policy that is continuously updated.

Since the behavior policy and target policy are the same, Sarsa is on-policy.

**Sample generation in Sarsa**: Each iteration requires $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$:

$$s_t \xrightarrow{\pi_b} a_t \xrightarrow{\text{model}} r_{t+1}, s_{t+1} \xrightarrow{\pi_b} a_{t+1}$$

The behavior policy $\pi_b$ generates both $a_t$ and $a_{t+1}$. Sarsa evaluates this same policy. The target policy $\pi_T$ equals $\pi_b$.

#### Why Q-learning is Off-policy

The fundamental reason: Q-learning solves the **Bellman optimality equation** (not the Bellman equation of a given policy). The optimal values and policies are **independent** of which policy generates the samples.

**Sample generation in Q-learning**: Each iteration requires $(s_t, a_t, r_{t+1}, s_{t+1})$:

$$s_t \xrightarrow{\pi_b} a_t \xrightarrow{\text{model}} r_{t+1}, s_{t+1}$$

The behavior policy $\pi_b$ generates $a_t$ at $s_t$. The estimation of the optimal action value for $(s_t, a_t)$ relies on $(r_{t+1}, s_{t+1})$, which is governed by the environment model, **not** by $\pi_b$. Therefore, $\pi_b$ can be **any** policy. The target policy $\pi_T$ is the greedy policy obtained from the estimated optimal values.

#### MC learning is On-policy

The target policy to be evaluated and improved is the same as the behavior policy that generates samples -- similar reasoning to Sarsa.

### Online vs. Offline (Not to Be Confused with On/Off-policy)

| Concept | Definition |
|---|---|
| **Online learning** | Agent updates values and policies **while interacting** with the environment |
| **Offline learning** | Agent updates values and policies using **pre-collected** experience data, without interacting with the environment |

**Relationship**: On-policy algorithms can be implemented online but cannot use pre-collected data from other policies. Off-policy algorithms can be implemented in **either** online or offline fashion.

### 7.4.3 Implementation

#### Algorithm 7.2: Q-learning (On-policy Version)

```
Initialization:
  alpha_t(s,a) = alpha > 0 for all (s,a) and all t
  epsilon in (0, 1)
  Initial q_0(s,a) for all (s,a)
  Initial epsilon-greedy policy pi_0 derived from q_0

Goal: Learn an optimal path from initial state s_0 to target state.

For each episode, do:
  If s_t (t = 0, 1, 2, ...) is not the target state, do:

    Collect experience sample (a_t, r_{t+1}, s_{t+1}) given s_t:
      generate a_t following pi_t(s_t)
      generate r_{t+1}, s_{t+1} by interacting with the environment

    Update q-value for (s_t, a_t):
      q_{t+1}(s_t, a_t) = q_t(s_t, a_t)
        - alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + gamma * max_a q_t(s_{t+1}, a))]

    Update policy for s_t (epsilon-greedy):
      pi_{t+1}(a|s_t) = 1 - epsilon/(|A(s_t)|) * (|A(s_t)| - 1)
                         if a = argmax_a q_{t+1}(s_t, a)
      pi_{t+1}(a|s_t) = epsilon / |A(s_t)|   otherwise
```

This implementation is similar to Sarsa (Algorithm 7.1). The behavior policy is the same as the target policy ($\epsilon$-greedy).

#### Algorithm 7.3: Q-learning (Off-policy Version)

```
Initialization:
  Initial guess q_0(s,a) for all (s,a)
  Behavior policy pi_b(a|s) for all (s,a)
  alpha_t(s,a) = alpha > 0 for all (s,a) and all t

Goal: Learn an optimal target policy pi_T for all states from
      experience samples generated by pi_b.

For each episode {s_0, a_0, r_1, s_1, a_1, r_2, ...} generated by pi_b, do:
  For each step t = 0, 1, 2, ... of the episode, do:

    Update q-value for (s_t, a_t):
      q_{t+1}(s_t, a_t) = q_t(s_t, a_t)
        - alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + gamma * max_a q_t(s_{t+1}, a))]

    Update target policy for s_t (greedy):
      pi_{T,t+1}(a|s_t) = 1   if a = argmax_a q_{t+1}(s_t, a)
      pi_{T,t+1}(a|s_t) = 0   otherwise
```

**Key differences from on-policy version**:
- The behavior policy $\pi_b$ can be **any** policy (preferably exploratory).
- The target policy $\pi_T$ is **greedy** (not $\epsilon$-greedy) since it is not used to generate samples and hence does not need to be exploratory.
- The off-policy version is implemented **offline**: all experience samples are collected first and then processed.

### 7.4.4 Illustrative Examples

#### Example 1: On-policy Q-learning (path finding)

**Setup**: 5x5 grid world, episodes start from top-left, terminate at target. Rewards: $r_{\text{target}} = 0$, $r_{\text{forbidden}} = r_{\text{boundary}} = -10$, $r_{\text{other}} = -1$. Learning rate $\alpha = 0.1$, $\epsilon = 0.1$.

**Results**: Q-learning finds an optimal path. Episode length decreases and total reward increases during learning.

#### Example 2: Off-policy Q-learning (optimal policy for all states)

**Setup**: 5x5 grid world. Reward: $r_{\text{boundary}} = r_{\text{forbidden}} = -1$, $r_{\text{target}} = 1$. Discount rate $\gamma = 0.9$, learning rate $\alpha = 0.1$.

**Ground truth** (from model-based policy iteration):
- Optimal state values range from 5.6 to 10.0 across the grid

**Behavior policy**: Uniform distribution ($\pi_b(a|s) = 0.2$ for all $s, a$). A single episode with 100,000 steps is generated.

**Results**:
- The learned target policy is **optimal** (state value error converges to zero).
- The learned policy may differ from the ground truth in specific actions but has the **same optimal state values** (multiple optimal policies exist).

**Effect of initial values** (bootstrapping sensitivity):
- $q_0(s, a) = 0$: Converges within approximately 10,000 steps.
- $q_0(s, a) = 10$: Requires moderately more steps.
- $q_0(s, a) = 100$: Requires significantly more steps but still converges.

**Effect of behavior policy exploration**:
- $\epsilon = 1.0$ (uniform): Best exploration, fastest convergence.
- $\epsilon = 0.5$: Reduced exploration, slower learning.
- $\epsilon = 0.1$: Poor exploration, significantly degraded performance because insufficient experience samples are generated for many state-action pairs.

---

## 7.5 A Unified Viewpoint

All TD algorithms (and MC learning) can be expressed in a single **unified expression**:

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)\big[q_t(s_t, a_t) - \bar{q}_t\big] \tag{7.20}$$

where $\bar{q}_t$ is the **TD target**. Different algorithms have different TD targets.

### Table 7.2: Unified TD Target Expressions

| Algorithm | TD Target $\bar{q}_t$ |
|---|---|
| **Sarsa** | $\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$ |
| **n-step Sarsa** | $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_t(s_{t+n}, a_{t+n})$ |
| **Q-learning** | $\bar{q}_t = r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$ |
| **Monte Carlo** | $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$ |

**MC as a special case**: Setting $\alpha_t(s_t, a_t) = 1$ in (7.20) gives $q_{t+1}(s_t, a_t) = \bar{q}_t$, which is the MC learning algorithm.

### Table 7.2: Underlying Equations

| Algorithm | Equation Solved |
|---|---|
| **Sarsa** | **BE**: $q_\pi(s, a) = E[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]$ |
| **n-step Sarsa** | **BE**: $q_\pi(s, a) = E[R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) \mid S_t = s, A_t = a]$ |
| **Q-learning** | **BOE**: $q(s, a) = E\big[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \mid S_t = s, A_t = a\big]$ |
| **Monte Carlo** | **BE**: $q_\pi(s, a) = E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \mid S_t = s, A_t = a]$ |

(BE = Bellman Equation; BOE = Bellman Optimality Equation)

**Key insight**: Algorithm (7.20) can be viewed as a stochastic approximation algorithm for solving the unified equation $q(s, a) = E[\bar{q}_t | s, a]$. All algorithms solve the Bellman equation **except Q-learning**, which solves the Bellman optimality equation.

---

## 7.6 Summary

Key takeaways from this chapter:

1. **TD learning algorithms** are model-free methods that can be viewed as **stochastic approximation algorithms** for solving Bellman or Bellman optimality equations.

2. All TD algorithms except Q-learning are used to **evaluate a given policy** (estimate state/action values). Together with policy improvement, they can learn optimal policies. These algorithms are **on-policy**: the target policy is used as the behavior policy.

3. **Q-learning** is special: it is **off-policy**. The target policy can differ from the behavior policy. The fundamental reason is that Q-learning solves the **Bellman optimality equation** rather than the Bellman equation of a given policy.

4. **Importance sampling** (introduced in Chapter 10) can convert on-policy algorithms to off-policy.

5. Extensions exist: **TD($\lambda$)** provides a more general and unified framework for TD learning.

---

## 7.7 Q&A -- Important Clarifications

### Q: What does "TD" in TD learning mean?
**A**: Every TD algorithm has a TD error representing the discrepancy between the new sample and the current estimate. Since this discrepancy is calculated between **different time steps**, it is called **temporal-difference**.

### Q: What does "learning" in TD learning mean?
**A**: From a mathematical viewpoint, "learning" simply means **estimation** -- estimating state/action values from samples and then obtaining policies based on the estimated values.

### Q: How can Sarsa (which only estimates action values) be used to learn optimal policies?
**A**: Through **generalized policy iteration**: after a value is updated, the corresponding policy is immediately updated. The updated policy generates new samples for further value estimation. This alternating process converges to an optimal policy.

### Q: Why does Sarsa update policies to be $\epsilon$-greedy?
**A**: Because the policy is also used to **generate samples** for value estimation. It must be exploratory to ensure sufficient experience samples are generated for all state-action pairs.

### Q: Why is $\alpha_t$ often set as a small constant in practice (despite convergence theorems requiring decaying $\alpha_t$)?
**A**: The fundamental reason is that the policy being evaluated keeps **changing** (is nonstationary). If the policy were fixed, a decaying learning rate would be fine. However, in optimal policy learning, the policy changes every iteration. A constant learning rate is needed; otherwise, a decaying rate may become too small to effectively evaluate the changing policies. Although constant learning rates cause the value estimate to fluctuate, the fluctuation is negligible when the constant is sufficiently small.

### Q: Should we learn optimal policies for all states or a subset?
**A**: It depends on the task. Some tasks only require finding an optimal path from a specific start to a target (e.g., Figure 7.2). Such tasks need less data but the obtained path is **not guaranteed** to be globally optimal since unexplored state-action pairs may contain better paths. Given sufficient data, a good or locally optimal path can still be found.

### Q: Why is Q-learning off-policy while others are on-policy?
**A**: Q-learning solves the **Bellman optimality equation**, which yields optimal values independent of the data-generating policy. The other algorithms solve the **Bellman equation of a given policy**, requiring samples from that specific policy.

### Q: Why does off-policy Q-learning use greedy (not $\epsilon$-greedy) target policies?
**A**: The target policy is not used to generate experience samples, so it does **not need to be exploratory**. The exploration is handled by the behavior policy.

---

## Concept Index

| Concept | Notation / Formula | Section |
|---|---|---|
| TD learning (state values) | $v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)[v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))]$ | 7.1 |
| TD target (state value) | $\bar{v}_t = r_{t+1} + \gamma v_t(s_{t+1})$ | 7.1 |
| TD error (state value) | $\delta_t = v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))$ | 7.1 |
| Bellman expectation equation (state) | $v_\pi(s) = E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$ | 7.1 |
| Innovation (new information) | TD error as new information from $(s_t, r_{t+1}, s_{t+1})$ | 7.1 |
| Bootstrapping | Updating values based on previous estimates | 7.1 |
| Sarsa | $q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))]$ | 7.2 |
| Bellman equation (action values) | $q_\pi(s,a) = E[R + \gamma q_\pi(S', A') \mid s, a]$ | 7.2 |
| $\epsilon$-greedy policy | $\pi(a|s) = 1 - \frac{\epsilon}{|\mathcal{A}|}(|\mathcal{A}|-1)$ for greedy $a$; $\frac{\epsilon}{|\mathcal{A}|}$ otherwise | 7.2 |
| Generalized policy iteration (GPI) | Alternating value update and policy improvement | 7.2 |
| Expected Sarsa | Uses $r_{t+1} + \gamma E[q_t(s_{t+1}, A)]$ as TD target | 7.2 |
| n-step Sarsa | $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^n q_t(s_{t+n}, a_{t+n})$ | 7.3 |
| Return decomposition | $G_t^{(n)} = R_{t+1} + \cdots + \gamma^n q_\pi(S_{t+n}, A_{t+n})$ | 7.3 |
| Q-learning | $q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a))]$ | 7.4 |
| Bellman optimality equation (action) | $q(s,a) = E[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \mid s, a]$ | 7.4 |
| Behavior policy | $\pi_b$: generates experience samples | 7.4 |
| Target policy | $\pi_T$: updated toward optimal policy | 7.4 |
| On-policy | Behavior policy = target policy | 7.4 |
| Off-policy | Behavior policy $\neq$ target policy | 7.4 |
| Online learning | Update while interacting with environment | 7.4 |
| Offline learning | Update using pre-collected data | 7.4 |
| Unified TD expression | $q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - \bar{q}_t]$ | 7.5 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| TD learning algorithms (Sarsa, Q-learning) | Ch 8 (extended with function approximation: DQN, etc.) |
| On-policy / off-policy distinction | Ch 8 (on/off-policy value function methods); Ch 10 (importance sampling for off-policy) |
| $\epsilon$-greedy exploration | Ch 8, 9, 10 (exploration strategies in advanced methods) |
| Bellman equation solving via stochastic approximation | Ch 8 (semi-gradient methods); Ch 10 (actor-critic) |
| Bellman optimality equation (Q-learning) | Ch 8 (Deep Q-Network / DQN) |
| Generalized policy iteration | Ch 8, 9, 10 (all advanced methods use GPI) |
| TD error / innovation concept | Ch 10 (advantage function, TD error in actor-critic) |
| Importance sampling (mentioned) | Ch 10 (detailed treatment for converting on-policy to off-policy) |
| TD($\lambda$) (mentioned as extension) | Not covered in this book; see [3, 20, 46] |
| Stochastic approximation (Ch 6) | Foundation for all convergence proofs in this chapter |
| Bellman equation (Ch 2) | Equations (7.4), (7.13) are alternative forms |
| Bellman optimality equation (Ch 3) | Equation (7.19) is an alternative form |
| Monte Carlo methods (Ch 5) | Unified with TD via n-step Sarsa (Section 7.3) |
| Policy iteration / value iteration (Ch 4) | Ground truth for verification; GPI concept |
