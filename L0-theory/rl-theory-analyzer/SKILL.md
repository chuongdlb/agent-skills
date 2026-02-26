---
name: rl-theory-analyzer
description: >
  Analyzes RL algorithms mathematically — derives update rules, proves convergence, identifies failure modes, and classifies algorithms along standard taxonomic dimensions.
layer: L0
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: []
tags: [theory, analysis, bellman, stochastic-approximation]
---

# RL Theory Analyzer

## Purpose
Analyze RL algorithms mathematically — derive update rules, prove convergence, identify failure modes, and classify algorithms along standard taxonomic dimensions.

## When to Use
Invoke this skill when you need to:
- Mathematically analyze an RL algorithm's update rule
- Determine what equation an algorithm solves (Bellman equation vs Bellman optimality equation)
- Classify an algorithm (on/off-policy, model-free/based, value/policy/AC)
- Identify potential failure modes (deadly triad, divergence conditions)
- Formulate an algorithm as a stochastic approximation problem

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
- **Bellman equation** (BE): `v_pi = r_pi + gamma * P_pi * v_pi` → evaluates a given policy
- **Bellman optimality equation** (BOE): `v* = max_pi (r_pi + gamma * P_pi * v*)` → finds optimal policy
- **Projected Bellman equation** (PBE): `Phi * w = Pi_D * T_pi(Phi * w)` → FA setting

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

## Key Equations Reference

### Bellman Equation (matrix-vector form)
```
v_pi = r_pi + gamma * P_pi * v_pi
```
Closed-form: `v_pi = (I - gamma * P_pi)^{-1} * r_pi`

### Bellman Optimality Equation
```
v*(s) = max_a sum_{s'} p(s'|s,a) * [sum_r p(r|s,a)*r + gamma * v*(s')]
```

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

## Output Format
When analyzing an algorithm, produce:
1. **Algorithm Classification:** (model-free/based, on/off-policy, value/policy/AC)
2. **Update Rule:** (in standard form)
3. **Fixed Point:** (what equation it solves)
4. **SA Formulation:** (Robbins-Monro or Dvoretzky form)
5. **Convergence:** (theorem applicable, conditions required)
6. **Failure Modes:** (potential issues and mitigations)
