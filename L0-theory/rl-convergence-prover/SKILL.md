---
name: rl-convergence-prover
description: >
  Proves or disproves convergence of RL algorithms using contraction mapping, Dvoretzky, Extended Dvoretzky, and Robbins-Monro theorems.
layer: L0
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [rl-theory-analyzer]
tags: [convergence, proof, stochastic-approximation, theory]
---

# RL Convergence Prover

## Purpose
Prove or disprove convergence of RL algorithms using the mathematical framework from Zhao's "Mathematical Foundations of Reinforcement Learning."

## When to Use
Invoke this skill when you need to:
- Prove that an RL algorithm converges
- Identify conditions under which convergence fails
- Verify that learning rate schedules satisfy required conditions
- Determine the fixed point an algorithm converges to
- Analyze convergence rate

## Core Convergence Theorems

### Theorem 1: Contraction Mapping Theorem
**Statement:** Let (X, d) be a complete metric space and T: X → X be a contraction mapping with modulus gamma in [0,1), i.e., d(T(x), T(y)) <= gamma * d(x,y) for all x,y. Then:
1. T has a unique fixed point x* such that T(x*) = x*
2. For any x_0, the sequence x_{k+1} = T(x_k) converges to x*
3. Convergence rate: d(x_k, x*) <= gamma^k * d(x_0, x*)

**Application to RL:**
- The Bellman operator T_pi(v) = r_pi + gamma * P_pi * v is a contraction in the infinity norm with modulus gamma
- The BOE operator f(v)(s) = max_a [r(s,a) + gamma * sum_{s'} p(s'|s,a) * v(s')] is also a contraction with modulus gamma
- Therefore value iteration converges: ||v_{k+1} - v*||_inf <= gamma * ||v_k - v*||_inf

**Proof recipe for contraction:**
1. Define the operator T
2. Show ||T(v1) - T(v2)||_inf <= gamma * ||v1 - v2||_inf for all v1, v2
3. The key step usually involves: max or sum operations preserve the contraction, and gamma < 1 provides the contraction factor
4. Conclude: unique fixed point exists and iteration converges

### Theorem 2: Dvoretzky's Theorem (Theorem 6.2)
**Statement:** Consider the iteration:
```
Delta_{k+1} = (1 - alpha_k) * Delta_k + beta_k * eta_k
```
where Delta_k is the error sequence. If:
- (a) 0 <= alpha_k <= 1, sum(alpha_k) = inf, sum(alpha_k^2) < inf
- (b) sum(beta_k^2) < inf
- (c) E[eta_k | H_k] = 0 (unbiased noise)
- (d) var[eta_k | H_k] <= C for some constant C

Then Delta_k → 0 almost surely (i.e., the algorithm converges to the fixed point w*).

**Application to RL:**
- TD(0) convergence: Set Delta_k = v_k(s) - v_pi(s)
- Sarsa convergence: Set Delta_k = q_k(s,a) - q_pi(s,a)

### Theorem 3: Extended Dvoretzky's Theorem (Theorem 6.3)
**Statement:** Consider the multi-variable iteration. For each component i:
```
Delta_{k+1}(i) = (1 - alpha_k(i)) * Delta_k(i) + beta_k(i) * eta_k(i)
```
where now the noise may be BIASED. If:
- (a) 0 <= alpha_k(i) <= 1, sum(alpha_k(i)) = inf, sum(alpha_k(i)^2) < inf
- (b) ||beta_k||_inf <= beta_hat_k where sum(beta_hat_k^2) < inf
- (c) ||E[eta_k | H_k]||_inf <= gamma * ||Delta_k||_inf (BIASED expectation allowed!)
- (d) var[eta_k(i) | H_k] <= C * (1 + ||Delta_k||_inf^2)

Then Delta_k → 0 almost surely.

**Critical application — Q-learning convergence:**
- Q-learning has biased noise because the target uses max_{a'} q(s',a') instead of q(s',a') under the policy
- The bias is bounded by gamma * ||Delta_k||_inf due to the contraction property of the max operator
- This is exactly condition (c) of Extended Dvoretzky, with the same gamma as the discount factor
- Therefore Q-learning converges despite being off-policy (in the tabular case)

### Theorem 4: Robbins-Monro Convergence
**Statement:** Consider w_{k+1} = w_k - alpha_k * g_tilde(w_k, eta_k) where g_tilde = g(w) + noise. If:
- (a) g(w) is bounded and points toward w* (g(w)*(w-w*) > 0 for w ≠ w*)
- (b) sum(alpha_k) = inf, sum(alpha_k^2) < inf
- (c) E[noise | history] = 0

Then w_k → w* almost surely.

**Application:** SGD for minimizing J(w), any SA-based algorithm.

### Theorem 5: Linear TD Convergence (Function Approximation)
**Statement:** For TD(0) with linear FA, v_hat(s,w) = phi(s)^T * w:
The deterministic update is w_{k+1} = w_k + alpha * (b - A * w_k) where:
- A = Phi^T * D * (I - gamma * P_pi) * Phi
- b = Phi^T * D * r_pi
- D = diag(d_pi) (stationary distribution)

**A is positive definite** because (I - gamma * P_pi) can be decomposed to show diagonal dominance, and D is positive. Therefore:
- Unique fixed point: w* = A^{-1} * b
- The iteration converges (SA conditions apply)
- Error bound: ||Phi*w* - v_pi||_D <= (1/sqrt(1-gamma^2)) * min_w ||Phi*w - v_pi||_D

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
- [ ] Variance bounded: var[eta_k | H_k] <= C

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

**Problem: Non-tabular state space**
- Use projected Bellman equation formulation
- Verify projection operator properties

**Problem: Continuous actions**
- For deterministic PG: gradient exists and is bounded
- For stochastic PG: log-derivative trick ensures unbiased gradient estimate

### Step 5: State the Result
Clearly state:
1. The fixed point the algorithm converges to
2. The conditions required for convergence
3. The convergence rate (if available)
4. Any limitations or caveats

## Quick Reference: Known Convergence Results

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

## Output Format
When proving convergence, produce:
1. **Algorithm:** (name and update rule)
2. **Theorem Applied:** (which convergence theorem)
3. **SA Formulation:** (error form or RM form)
4. **Condition Verification:** (check each condition with justification)
5. **Conclusion:** (converges to what, under what conditions, at what rate)
6. **Caveats:** (limitations, failure modes)
