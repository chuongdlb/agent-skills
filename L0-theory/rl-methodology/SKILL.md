---
name: rl-methodology
description: >
  Comprehensive RL methodology — mathematical analysis, convergence proofs, algorithm design patterns, and implementation templates grounded in Zhao's "Mathematical Foundations of Reinforcement Learning."
layer: L0
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: []
tags: [theory, convergence, algorithm-design, implementation, stochastic-approximation]
---

# RL Methodology

## Purpose & When to Use

Invoke this skill when you need to:
- Mathematically analyze an RL algorithm's update rule, classify it, or identify failure modes
- Prove or disprove convergence using contraction mapping, Dvoretzky, or Robbins-Monro theorems
- Design a new RL algorithm by composing known design patterns
- Implement an RL algorithm from pseudocode with correct mathematical grounding
- Debug an RL implementation by checking mathematical correctness
- Look up theoretical foundations (Bellman equations, SA conditions, policy gradient theorem)

## Knowledge Base Navigation

The `knowledge-base/zhao-mathematical-foundations/` directory contains the full mathematical reference. Use this lookup table to find the right file:

| Topic | KB File | Key Content |
|-------|---------|-------------|
| MDPs, states, actions, rewards | `01-basic-concepts.md` | Definitions, notation |
| Bellman equation, v_pi, q_pi | `02-bellman-equation.md` | Matrix-vector form, closed-form solution |
| Bellman optimality equation, v* | `03-bellman-optimality-equation.md` | BOE, contraction mapping |
| Value iteration, policy iteration | `04-value-iteration-policy-iteration.md` | VI, PI, truncated PI, GPI |
| Monte Carlo methods | `05-monte-carlo-methods.md` | MC prediction, MC control, exploring starts |
| Stochastic approximation theory | `06-stochastic-approximation.md` | Robbins-Monro, Dvoretzky theorems |
| TD(0), Sarsa, Q-learning | `07-temporal-difference-methods.md` | TD methods, convergence proofs |
| Function approximation, DQN | `08-value-function-methods.md` | Linear FA, semi-gradient TD, deadly triad |
| Policy gradient, REINFORCE | `09-policy-gradient-methods.md` | PG theorem, baseline subtraction |
| Actor-critic (A2C, DDPG, SAC) | `10-actor-critic-methods.md` | QAC, A2C, off-policy AC, DPG |
| Math prerequisites | `11-appendix.md` | Norms, probability, linear algebra |
| Algorithm comparison & index | `12-cross-reference-index.md` | Convergence table, theorem index |

## Analysis Procedure

### Step 1: Identify the Update Rule
Extract the core update equation. Express it in the standard SA form:
```
w_{k+1} = w_k - alpha_k * [w_k - target_k]
```
or equivalently:
```
w_{k+1} = (1 - alpha_k) * w_k + alpha_k * target_k
```

### Step 2: Identify the Fixed Point Equation
Determine what equation the algorithm solves at convergence:
- **Bellman equation** (BE): `v_pi = r_pi + gamma * P_pi * v_pi` — evaluates a given policy
- **Bellman optimality equation** (BOE): `v* = max_pi (r_pi + gamma * P_pi * v*)` — finds optimal policy
- **Projected Bellman equation** (PBE): `Phi * w = Pi_D * T_pi(Phi * w)` — FA setting

### Step 3: Classify the Algorithm

**Model-free vs Model-based:**
- Model-free: Does NOT require p(s'|s,a) or p(r|s,a) — uses samples instead
- Model-based: Requires the transition/reward model

**On-policy vs Off-policy:**
- On-policy: Behavior policy = target policy (Sarsa, REINFORCE)
- Off-policy: Behavior policy ≠ target policy (Q-learning, DPG)
- Key test: Does the update use actions from the behavior policy that differ from what the target policy would choose?

**Value-based vs Policy-based vs Actor-Critic:**
- Value-based: Learns v(s) or q(s,a), derives policy via argmax (Q-learning, DQN)
- Policy-based: Directly parameterizes and optimizes pi(a|s,theta) (REINFORCE)
- Actor-Critic: Maintains both policy (actor) and value (critic) networks (A2C, DDPG)

### Step 4: Formulate as Stochastic Approximation
Rewrite the update as a Robbins-Monro or Dvoretzky problem:

**Robbins-Monro form:** `w_{k+1} = w_k - alpha_k * g_tilde(w_k, eta_k)`
where g_tilde(w, eta) = g(w) + eta, E[eta|history] = 0

**Dvoretzky form:** `Delta_{k+1} = (1 - alpha_k) * Delta_k + beta_k * eta_k`
where Delta_k = w_k - w* is the error

For Q-learning specifically, use the **Extended Dvoretzky** (Theorem 6.3):
- The expectation condition allows bias: `||E[eta_k | H_k]||_inf <= gamma * ||Delta_k||_inf`
- This is critical because Q-learning's target contains the max operator

### Step 5: Check for Failure Modes

**The Deadly Triad** (divergence risk when ALL THREE present):
1. Function approximation (not tabular)
2. Bootstrapping (using current estimates in targets)
3. Off-policy learning (behavior ≠ target policy)

**Common failure modes:**
- Constant learning rate with stochastic updates → oscillation, no convergence
- Learning rate too large → divergence
- No exploration → stuck at suboptimal policy
- Linear FA + off-policy + bootstrapping → potential divergence (Baird's counterexample)
- Nonlinear FA → no convergence guarantees in general

### Step 6: Analyze Convergence Properties
Reference the appropriate convergence theorem:
- **Tabular value-based:** Contraction mapping theorem (gamma < 1 guarantees convergence)
- **Tabular TD/Q-learning:** Dvoretzky's theorem or Extended Dvoretzky
- **Linear FA + on-policy:** TD converges to w* = A^{-1}b where A is positive definite
- **Policy gradient:** Converges to local optimum under standard conditions
- **Nonlinear FA + off-policy:** No general guarantees

**Output format** — when analyzing an algorithm, produce:
1. **Algorithm Classification:** (model-free/based, on/off-policy, value/policy/AC)
2. **Update Rule:** (in standard form)
3. **Fixed Point:** (what equation it solves)
4. **SA Formulation:** (Robbins-Monro or Dvoretzky form)
5. **Convergence:** (theorem applicable, conditions required)
6. **Failure Modes:** (potential issues and mitigations)

## Convergence Proof Procedure

### Step 1: Identify the Algorithm Type
| Type | Primary Tool |
|------|-------------|
| Value iteration (exact) | Contraction mapping theorem |
| Policy iteration (exact) | Monotonic improvement + bounded values |
| TD methods (tabular) | Dvoretzky's theorem |
| Q-learning (tabular) | Extended Dvoretzky's theorem |
| TD with linear FA | Matrix A positive definiteness |
| Policy gradient | Standard gradient ascent convergence |
| SGD-based | Robbins-Monro theorem |

### Step 2: Formulate as SA Problem
Rewrite the update rule in one of these forms:
- **Error form:** Delta_{k+1} = (1-alpha_k)*Delta_k + beta_k*eta_k
- **RM form:** w_{k+1} = w_k - alpha_k*(g(w_k) + noise_k)
- **Operator form:** v_{k+1} = T(v_k) [for contraction mapping]

### Step 3: Verify Conditions
Check each condition of the applicable theorem:

**Learning rate conditions (almost always needed):**
- [ ] sum(alpha_k) = infinity (ensures we can reach any point)
- [ ] sum(alpha_k^2) < infinity (ensures noise averages out)
- Common valid schedules: alpha_k = 1/k, alpha_k = c/(c+k)

**Noise conditions:**
- [ ] E[eta_k | H_k] = 0 (for Dvoretzky) or ||E[eta_k | H_k]||_inf <= gamma*||Delta||_inf (for Extended Dvoretzky)
- [ ] Second moment bounded: E[eta_k^2 | H_k] <= C

**Contraction conditions:**
- [ ] gamma < 1 (discount factor)
- [ ] Operator satisfies ||T(v1) - T(v2)|| <= gamma * ||v1 - v2||

### Step 4: Handle Common Difficulties

**Problem: Biased noise (e.g., Q-learning)**
- Solution: Use Extended Dvoretzky instead of standard Dvoretzky
- Show bias is bounded by gamma * ||Delta_k||_inf

**Problem: Function approximation**
- Linear FA + on-policy: Show matrix A is positive definite
- Linear FA + off-policy: Check for potential divergence (deadly triad)
- Nonlinear FA: Generally no convergence guarantees; use empirical validation

### Step 5: State the Result
Clearly state:
1. The fixed point the algorithm converges to
2. The conditions required for convergence
3. The convergence rate (if available)
4. Any limitations or caveats

**Output format** — when proving convergence, produce:
1. **Algorithm:** (name and update rule)
2. **Theorem Applied:** (which convergence theorem)
3. **SA Formulation:** (error form or RM form)
4. **Condition Verification:** (check each condition with justification)
5. **Conclusion:** (converges to what, under what conditions, at what rate)
6. **Caveats:** (limitations, failure modes)

## Quick Reference: Convergence Results

| Algorithm | Converges? | Conditions | Fixed Point |
|-----------|-----------|------------|-------------|
| Value Iteration | Yes | gamma < 1 | v* (optimal value) |
| Policy Iteration | Yes | gamma < 1 | v* (optimal value) |
| TD(0) tabular | Yes | SA learning rates | v_pi |
| Sarsa tabular | Yes | SA learning rates + GLIE | q* |
| Q-learning tabular | Yes | SA learning rates + exploration | q* |
| TD linear FA, on-policy | Yes | SA learning rates | w* = A^{-1}b |
| Q-learning linear FA | May diverge | Deadly triad risk | — |
| Q-learning nonlinear FA | No guarantee | Empirical only | — |
| REINFORCE | Yes (local) | SA learning rates | Local optimum of J |
| A2C | Yes (local) | SA learning rates, two timescales | Local optimum |

### Core Convergence Theorems

**Theorem 1: Contraction Mapping** — Let T: X → X be a contraction with modulus gamma in [0,1). Then T has a unique fixed point x*, and x_{k+1} = T(x_k) converges with rate gamma^k.

**Theorem 2: Dvoretzky's (Thm 6.2)** — For Delta_{k+1} = (1-alpha_k)*Delta_k + beta_k*eta_k, if SA learning rates hold, sum(beta_k^2) < inf, E[eta_k|H_k] = 0, and E[eta_k^2|H_k] <= C, then Delta_k → 0 a.s.

**Theorem 3: Extended Dvoretzky (Thm 6.3)** — Same as above but allows biased noise: ||E[eta_k|H_k]||_inf <= gamma*||Delta_k||_inf. Critical for Q-learning convergence.

**Theorem 4: Robbins-Monro** — For w_{k+1} = w_k - alpha_k*g_tilde(w_k, eta_k), if g is monotone, SA learning rates hold, and noise is zero-mean, then w_k → w* a.s.

**Theorem 5: Linear TD Convergence** — For TD(0) with linear FA v_hat(s,w) = phi(s)^T*w: A = Phi^T*D*(I-gamma*P_pi)*Phi is positive definite, so w* = A^{-1}*b exists and is unique. Error bound: ||Phi*w* - v_pi||_D <= (1/sqrt(1-gamma^2))*min_w||Phi*w - v_pi||_D.

## Algorithm Design Procedure

### Step 1: Define the Problem Setting
- State space: discrete or continuous?
- Action space: discrete or continuous?
- Model availability: known or unknown?
- Episode structure: episodic or continuing?
- Data regime: online or batch?

### Step 2: Choose the Core Approach
| Setting | Recommended |
|---------|------------|
| Discrete S, A, known model | Dynamic Programming (VI/PI) |
| Discrete S, A, unknown model, episodic | MC or TD methods |
| Large/continuous S, discrete A | DQN-style (value + FA) |
| Continuous S and A | Actor-Critic (A2C, DDPG, SAC) |
| Need sample efficiency | Off-policy + replay buffer |

### Step 3: Select Design Patterns
Compose from the 10 core patterns:

**Pattern 1: Generalized Policy Iteration (GPI)** — Alternating evaluation and improvement. VI (j=1), PI (j=∞), truncated PI (j steps).

**Pattern 2: Bootstrapping** — Using current estimates in targets: r + gamma*v(s'). Trades bias for variance. Risk: part of deadly triad.

**Pattern 2.5: Monte Carlo Estimation** — Complete episode returns as unbiased estimates. High variance but no bootstrapping bias.

**Pattern 3: Stochastic Approximation** — Replace expectations with samples. Use diminishing step sizes: sum(alpha_k)=inf, sum(alpha_k^2)<inf.

**Pattern 4: Epsilon-Greedy Exploration** — Greedy with prob 1-eps, random with prob eps. Variants: decaying epsilon, Boltzmann, UCB.

**Pattern 5: Experience Replay** — Store transitions in buffer, sample for training. Breaks correlation, reuses data. Used in DQN, DDPG, SAC.

**Pattern 6: Target Networks** — Separate slowly-updated network for targets. Prevents moving target problem. Used in DQN, DDPG, TD3.

**Pattern 7: Importance Sampling** — Correct for distribution mismatch in off-policy learning. Weight by rho = pi(a|s)/beta(a|s).

**Pattern 8: Baseline Subtraction** — Subtract b(s) from returns in PG. Key: E[grad ln pi * b(S)] = 0. Best baseline: b(s) = v_pi(s).

**Pattern 9: Function Approximation** — Parameterize value/policy: v_hat(s,w) = phi(s)^T*w or neural net. Enables generalization.

**Pattern 10: Actor-Critic Decomposition** — Separate actor (policy) and critic (value). Lower variance than pure PG, handles continuous actions.

### Step 4: Derive the Update Rule
**Value-based:** `w_{t+1} = w_t + alpha * [target - v_hat(s_t, w_t)] * grad_w v_hat(s_t, w_t)`

**Policy gradient:** `theta_{t+1} = theta_t + alpha * grad_theta ln pi(a_t|s_t, theta) * [G_t - b(s_t)]`

**Actor-critic:**
```
Critic: w_{t+1} = w_t + alpha_w * delta_t * grad_w v_hat(s_t, w_t)
Actor:  theta_{t+1} = theta_t + alpha_theta * delta_t * grad_theta ln pi(a_t|s_t, theta)
```
where delta_t = r_{t+1} + gamma * v_hat(s_{t+1}, w) - v_hat(s_t, w)

### Step 5: Verify Mathematical Soundness
- Check convergence conditions (reference `06-stochastic-approximation.md`)
- Verify no deadly triad combination
- Ensure learning rate schedule satisfies SA conditions
- Check that the update targets the correct fixed point

### Step 6: Specify Hyperparameters
- Learning rate schedule: alpha_k = 1/k or alpha_k = alpha_0 / (1 + k/tau)
- Discount factor: gamma in [0.9, 0.999] typical
- Exploration: epsilon schedule, entropy coefficient
- Target network: update frequency C or soft update tau
- Replay buffer: size, mini-batch size, prioritization

**Output format** — when designing an algorithm, produce:
1. **Problem Setting:** (state/action spaces, model availability, etc.)
2. **Algorithm Name:** (descriptive name)
3. **Design Patterns Used:** (list of patterns composed)
4. **Update Rules:** (complete mathematical specification)
5. **Pseudocode:** (step-by-step algorithm)
6. **Hyperparameters:** (with recommended defaults)
7. **Convergence Notes:** (conditions and guarantees)

## Implementation Templates

### Environment Interfaces

Templates below are written against three interfaces, from most to least specific:

**Tabular MDP interface** (VI, PI, MC, Sarsa, Q-learning, REINFORCE, A2C):
```python
n_states: int                                    # |S|, states are integers 0..n_states-1
n_actions: int                                   # |A|, actions are integers 0..n_actions-1
step(state, action) -> (next_state, reward)      # transition (deterministic or stochastic)
is_terminal(state) -> bool                       # episode termination check
sample_start() -> state                          # sample initial state
```

**Known-model extension** (VI, PI only — adds the transition/reward model):
```python
P: ndarray[n_states, n_actions, n_states]        # p(s'|s,a) transition probabilities
R: ndarray[n_states, n_actions]                  # E[r|s,a] expected rewards
```

**Gymnasium interface** (DQN — stateful, continuous observations):
```python
reset() -> (observation, info)                   # start episode
step(action) -> (observation, reward, terminated, truncated, info)
```

Any environment that implements the appropriate interface works with the corresponding templates.

---

### Model-Based: Value Iteration (Algorithm 4.1)
```python
import numpy as np

def value_iteration(P, R, n_states, n_actions, gamma=0.9, theta=1e-6,
                    max_iter=1000):
    """v_{k+1}(s) = max_a [R(s,a) + gamma * sum_{s'} P(s,a,s') * v_k(s')]

    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states).
        R: Expected rewards, shape (n_states, n_actions).
    Returns:
        V: Optimal value function, shape (n_states,).
        policy: Greedy policy, shape (n_states,).
    """
    V = np.zeros(n_states)
    for _ in range(max_iter):
        # Q(s,a) = R(s,a) + gamma * P(s,a,:) @ V
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new
    Q = R + gamma * np.einsum('ijk,k->ij', P, V)
    policy = np.argmax(Q, axis=1)
    return V, policy
```

### Model-Based: Policy Iteration (Algorithm 4.2)
```python
def policy_iteration(P, R, n_states, n_actions, gamma=0.9, theta=1e-6,
                     max_iter=1000):
    """Evaluate pi exactly, then improve greedily.

    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states).
        R: Expected rewards, shape (n_states, n_actions).
    Returns:
        V: Optimal value function, shape (n_states,).
        policy: Optimal policy, shape (n_states,).
    """
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    for _ in range(max_iter):
        # Policy evaluation: solve v_pi = r_pi + gamma * P_pi @ v_pi
        P_pi = P[np.arange(n_states), policy]        # (n_states, n_states)
        r_pi = R[np.arange(n_states), policy]        # (n_states,)
        V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, r_pi)

        # Policy improvement
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        new_policy = np.argmax(Q, axis=1)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy
```

### Model-Free: MC Epsilon-Greedy (Algorithm 5.3)
```python
def mc_epsilon_greedy(env, n_episodes=5000, gamma=0.9, epsilon=0.1,
                      max_steps=200):
    """First-visit MC with epsilon-greedy. Uses complete episode returns.

    env must provide: n_states, n_actions, step(s, a), is_terminal(s),
    sample_start().
    """
    Q = np.zeros((env.n_states, env.n_actions))
    returns_count = np.zeros((env.n_states, env.n_actions))
    for _ in range(n_episodes):
        trajectory = []
        state = env.sample_start()
        for _ in range(max_steps):
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
            next_state, reward = env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            if env.is_terminal(state):
                break
        # Backward return computation (first-visit)
        G = 0.0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_count[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / returns_count[s_t, a_t]
    return Q, np.argmax(Q, axis=1)
```

### Model-Free: Sarsa (Algorithm 7.1)
```python
def sarsa(env, n_episodes=5000, gamma=0.9, alpha_0=0.5, epsilon_0=0.1,
          max_steps=200):
    """On-policy TD control: q(s,a) += alpha * [r + gamma*q(s',a') - q(s,a)]

    env must provide: n_states, n_actions, step(s, a), is_terminal(s),
    sample_start().
    """
    Q = np.zeros((env.n_states, env.n_actions))
    visit_count = np.zeros((env.n_states, env.n_actions))

    def eps_greedy(state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        return np.argmax(Q[state])

    for episode in range(n_episodes):
        state = env.sample_start()
        epsilon = epsilon_0 / (1 + episode / 1000)
        action = eps_greedy(state, epsilon)
        for _ in range(max_steps):
            next_state, reward = env.step(state, action)
            visit_count[state, action] += 1
            next_action = eps_greedy(next_state, epsilon)
            alpha = alpha_0 / visit_count[state, action]
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            if env.is_terminal(state):
                break
    return Q, np.argmax(Q, axis=1)
```

### Model-Free: Q-Learning (Algorithm 7.x)
```python
def q_learning(env, n_episodes=5000, gamma=0.9, alpha_0=0.5, epsilon_0=0.1,
               max_steps=200):
    """Off-policy TD control: q(s,a) += alpha * [r + gamma*max q(s',.) - q(s,a)]

    env must provide: n_states, n_actions, step(s, a), is_terminal(s),
    sample_start().
    """
    Q = np.zeros((env.n_states, env.n_actions))
    visit_count = np.zeros((env.n_states, env.n_actions))
    for episode in range(n_episodes):
        state = env.sample_start()
        for _ in range(max_steps):
            epsilon = epsilon_0 / (1 + episode / 1000)
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
            next_state, reward = env.step(state, action)
            visit_count[state, action] += 1
            alpha = alpha_0 / visit_count[state, action]
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if env.is_terminal(state):
                break
    return Q, np.argmax(Q, axis=1)
```

### Policy Gradient: REINFORCE (Algorithm 9.1)
```python
def reinforce(env, n_episodes=5000, gamma=0.99, alpha=0.01, max_steps=200):
    """theta += alpha * gamma^t * G_t * grad ln pi(a|s,theta)

    Tabular softmax policy. env must provide: n_states, n_actions,
    step(s, a), is_terminal(s), sample_start().
    """
    theta = np.zeros((env.n_states, env.n_actions))

    def softmax_policy(state):
        logits = theta[state] - np.max(theta[state])
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    for episode in range(n_episodes):
        trajectory = []
        state = env.sample_start()
        for _ in range(max_steps):
            probs = softmax_policy(state)
            action = np.random.choice(env.n_actions, p=probs)
            next_state, reward = env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            if env.is_terminal(state):
                break
        G = 0.0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            probs = softmax_policy(s_t)
            grad_ln_pi = -probs.copy()
            grad_ln_pi[a_t] += 1.0
            theta[s_t] += alpha * (gamma ** t) * G * grad_ln_pi
    return theta, softmax_policy
```

### Actor-Critic: A2C (Algorithm 10.2)
```python
def a2c(env, n_episodes=5000, gamma=0.99, alpha_theta=0.01, alpha_w=0.05,
        max_steps=200):
    """Actor (theta) + critic (w) with TD error as advantage estimate.

    Tabular softmax actor, tabular state-value critic. env must provide:
    n_states, n_actions, step(s, a), is_terminal(s), sample_start().
    """
    theta = np.zeros((env.n_states, env.n_actions))
    w = np.zeros(env.n_states)

    def softmax_policy(state):
        logits = theta[state] - np.max(theta[state])
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    for episode in range(n_episodes):
        state = env.sample_start()
        for _ in range(max_steps):
            probs = softmax_policy(state)
            action = np.random.choice(env.n_actions, p=probs)
            next_state, reward = env.step(state, action)
            # TD error = advantage estimate: delta = r + gamma*V(s') - V(s)
            delta = reward + gamma * w[next_state] - w[state]
            w[state] += alpha_w * delta
            grad_ln_pi = -probs.copy()
            grad_ln_pi[action] += 1.0
            theta[state] += alpha_theta * delta * grad_ln_pi
            state = next_state
            if env.is_terminal(state):
                break
    return theta, w, softmax_policy
```

### Function Approximation: TD(0) with Linear FA (Ch 8)
```python
def td_linear(env, feature_fn, n_episodes=5000, gamma=0.9, alpha_0=0.01,
              max_steps=200):
    """Semi-gradient TD(0): w += alpha * [r + gamma*phi(s')^T w - phi(s)^T w] * phi(s)

    env must provide: n_actions, step(s, a), is_terminal(s), sample_start().
    feature_fn(state) -> ndarray of shape (d,).
    """
    d = feature_fn(0).shape[0]
    w = np.zeros(d)
    for episode in range(n_episodes):
        state = env.sample_start()
        alpha = alpha_0 / (1 + episode / 1000)
        for _ in range(max_steps):
            action = np.random.randint(env.n_actions)  # uniform random behavior
            next_state, reward = env.step(state, action)
            phi_s = feature_fn(state)
            phi_s_next = feature_fn(next_state)
            td_error = reward + gamma * (phi_s_next @ w) - (phi_s @ w)
            w += alpha * td_error * phi_s
            state = next_state
            if env.is_terminal(state):
                break
    return w
```

### Deep RL: DQN (Algorithm 8.3 extended)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(env, obs_dim, n_actions, n_episodes=10000, gamma=0.99,
              lr=1e-3, buffer_size=10000, batch_size=64, target_update=100):
    """DQN with experience replay and target network.

    env must follow Gymnasium interface: reset() -> (obs, info),
    step(action) -> (obs, reward, terminated, truncated, info).
    """
    q_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=buffer_size)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        epsilon = max(0.01, 1.0 - episode / (n_episodes * 0.5))
        for _ in range(200):
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(obs)).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                s, a, r, s2, d = zip(*batch)
                s_t = torch.FloatTensor(np.array(s))
                a_t = torch.LongTensor(a)
                r_t = torch.FloatTensor(r)
                s2_t = torch.FloatTensor(np.array(s2))
                d_t = torch.FloatTensor(d)
                q_vals = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    targets = r_t + gamma * target_net(s2_t).max(1)[0] * (1 - d_t)
                loss = nn.MSELoss()(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if done:
                break
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
    return q_net
```

---

### Reference Environment: Zhao's 3x3 Grid World

Concrete implementation of the tabular MDP interface, matching the book's examples:

```python
class GridWorld:
    """3x3 grid world from Zhao's 'Mathematical Foundations of RL'.

    Implements the tabular MDP interface: n_states, n_actions,
    step(s, a), is_terminal(s), sample_start().
    """

    def __init__(self, size=3, gamma=0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 5  # up, down, left, right, stay
        self.gamma = gamma
        self.action_effects = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }
        self.forbidden_states = set()
        self.terminal_states = set()  # e.g., {8} for goal at bottom-right
        self.rewards = {}             # (s, a, s') -> r; default is -1

    def step(self, state, action):
        r, c = state // self.size, state % self.size
        dr, dc = self.action_effects[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = nr * self.size + nc
            if next_state not in self.forbidden_states:
                reward = self.rewards.get((state, action, next_state), -1)
                return next_state, reward
        return state, self.rewards.get((state, action, state), -1)

    def is_terminal(self, state):
        return state in self.terminal_states

    def sample_start(self):
        return np.random.randint(self.n_states)

    def get_model(self):
        """Returns (P, R) for model-based algorithms.

        P: shape (n_states, n_actions, n_states) — transition probabilities.
        R: shape (n_states, n_actions) — expected rewards.
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.full((self.n_states, self.n_actions), -1.0)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_next, reward = self.step(s, a)
                P[s, a, s_next] = 1.0
                R[s, a] = reward
        return P, R
```

## Learning Rate Schedules

### SA-Compliant Schedules (sum=inf, sum-sq<inf)
```python
# Schedule 1: Harmonic (1/k)
alpha_k = 1.0 / (visit_count + 1)

# Schedule 2: Polynomial decay
alpha_k = alpha_0 / (1 + k / tau)  # tau controls decay speed

# Schedule 3: Per-state-action (recommended for tabular)
alpha_k_sa = 1.0 / n_visits[s, a]
```

### Practical Schedules (constant or slow decay)
```python
# Constant (does NOT satisfy SA conditions, but works with target networks)
alpha = 1e-3  # for DQN, DDPG

# Linear decay
alpha_k = alpha_0 * (1 - k / n_total)

# Exponential decay
alpha_k = alpha_0 * decay_rate ** k
```

## Feature Engineering

### Tabular (one-hot) Features
```python
def tabular_features(state, n_states):
    """Exact representation — linear FA with one-hot recovers tabular methods."""
    phi = np.zeros(n_states)
    phi[state] = 1.0
    return phi
```

### Polynomial Features (2D grid)
```python
def polynomial_features(coords, degree=3):
    """Polynomial basis for 2D state spaces.

    Args:
        coords: (x, y) normalized to [0, 1].
        degree: Maximum polynomial degree.
    Returns:
        Feature vector of dimension (degree+1)(degree+2)/2.
    """
    x, y = coords
    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append(x**i * y**j)
    return np.array(features)
```

### Fourier Basis Features (2D grid)
```python
def fourier_features(coords, order=3):
    """Fourier cosine basis for 2D state spaces.

    Args:
        coords: (x, y) normalized to [0, 1].
        order: Maximum frequency order.
    Returns:
        Feature vector of dimension (order+1)^2.
    """
    x, y = coords
    features = []
    for i in range(order + 1):
        for j in range(order + 1):
            features.append(np.cos(np.pi * (i * x + j * y)))
    return np.array(features)
```

### Adapting to any state representation
```python
# For a grid world, convert state index to normalized coords:
def grid_to_coords(state, size):
    r, c = state // size, state % size
    return r / (size - 1), c / (size - 1)

# Then compose: feature_fn = lambda s: polynomial_features(grid_to_coords(s, 3))
```

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Values diverge | Learning rate too high | Reduce alpha or use SA schedule |
| No learning | Learning rate too low or no exploration | Increase alpha_0 or epsilon |
| Oscillating values | Constant alpha with noise | Use decreasing alpha_k |
| Wrong policy | Reward function incorrect | Verify r(s,a,s') |
| Slow convergence | Poor exploration | Increase epsilon, use exploring starts |
| FA divergence | Deadly triad | Use target networks, reduce bootstrap |

## Key Equations Reference

### Bellman Equation (matrix-vector form)
```
v_pi = r_pi + gamma * P_pi * v_pi
```
Closed-form: `v_pi = (I - gamma * P_pi)^{-1} * r_pi`

### Bellman Optimality Equation
```
v*(s) = max_a [sum_r p(r|s,a)*r + gamma * sum_{s'} p(s'|s,a) * v*(s')]
```

### Action Value
```
q_pi(s,a) = sum_r p(r|s,a) * r + gamma * sum_{s'} p(s'|s,a) * v_pi(s')
```
Relationship: `v_pi(s) = sum_a pi(a|s) * q_pi(s,a)`

### TD(0) Update
```
v_{t+1}(s_t) = v_t(s_t) + alpha_t * [r_{t+1} + gamma * v_t(s_{t+1}) - v_t(s_t)]
```

### Q-learning Update
```
q_{t+1}(s_t,a_t) = q_t(s_t,a_t) + alpha_t * [r_{t+1} + gamma * max_{a'} q_t(s_{t+1},a') - q_t(s_t,a_t)]
```

### Policy Gradient
```
grad_theta J(theta) = E_{S~d_pi, A~pi}[grad_theta ln pi(A|S,theta) * q_pi(S,A)]
```
