---
name: rl-innovator
description: >
  Meta-skill that orchestrates other RL skills to explore novel algorithm designs, identify limitations, and propose mathematically grounded improvements.
layer: meta
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [rl-theory-analyzer, rl-convergence-prover, rl-algorithm-designer, rl-implementer]
tags: [orchestration, innovation, meta]
---

# RL Innovator

## Purpose
Meta-skill that orchestrates the other RL skills to explore novel algorithm designs, identify limitations of existing approaches, and propose mathematically grounded improvements.

## When to Use
Invoke this skill when you need to:
- Identify limitations of an existing RL algorithm and propose improvements
- Design a novel algorithm variant for a specific problem
- Conduct a systematic exploration of the algorithm design space
- Validate a proposed algorithm through theory and implementation
- Compare algorithm variants along multiple dimensions

## Innovation Pipeline

### Phase 1: Problem Analysis
**Goal:** Understand the limitation or opportunity

1. **Identify the baseline algorithm** and its properties using `rl-theory-analyzer`
2. **Characterize the problem setting:**
   - What makes the problem hard? (large state space, continuous actions, sparse rewards, partial observability, non-stationarity)
   - What are the baseline's failure modes in this setting?
3. **Literature positioning:** What has been tried before? What gaps remain?

### Phase 2: Design Exploration
**Goal:** Generate candidate modifications

Use `rl-algorithm-designer` to explore modifications along these dimensions:

**Dimension 1: Objective Function**
- Standard: maximize expected discounted return
- Alternatives: risk-sensitive (CVaR), constrained (Lagrangian), multi-objective (Pareto), entropy-regularized (MaxEnt)
- Innovation: Define a new objective that addresses the identified limitation

**Dimension 2: Update Rule**
- Standard: single-step TD, MC return, n-step return
- Alternatives: Retrace(lambda), V-trace, generalized advantage estimation (GAE)
- Innovation: New target construction, multi-timescale updates, auxiliary tasks

**Dimension 3: Approximation Architecture**
- Standard: MLP, linear
- Alternatives: attention, graph neural networks, recurrent, ensemble
- Innovation: Architecture that encodes problem structure (symmetry, locality, hierarchy)

**Dimension 4: Exploration Strategy**
- Standard: epsilon-greedy, Boltzmann
- Alternatives: UCB, posterior sampling, curiosity-driven (ICM, RND), count-based
- Innovation: Exploration that leverages problem structure

**Dimension 5: Data Usage**
- Standard: on-policy (discard after use), off-policy replay buffer
- Alternatives: prioritized replay, hindsight replay (HER), model-based data augmentation
- Innovation: New replay strategies, data weighting schemes

### Phase 3: Theoretical Validation
**Goal:** Verify mathematical soundness of the proposed algorithm

Use `rl-convergence-prover` to:

1. **Formulate as SA:** Write the proposed update in Robbins-Monro or Dvoretzky form
2. **Check convergence conditions:**
   - Learning rate conditions satisfied?
   - Noise conditions met (unbiased or bounded bias)?
   - Contraction property holds?
3. **Identify failure modes:**
   - Deadly triad analysis
   - Stability under function approximation
   - Sensitivity to hyperparameters
4. **Establish guarantees:**
   - What fixed point does it converge to?
   - Error bounds relative to optimal?
   - Sample complexity?

### Phase 4: Implementation & Testing
**Goal:** Validate empirically

Use `rl-implementer` to:

1. **Implement the algorithm** in Python
2. **Set up test environments:**
   - Book's 5x5 grid world (sanity check)
   - Larger grid worlds (scalability)
   - Classic control tasks (CartPole, MountainCar) via Gymnasium
3. **Run experiments:**
   - Compare against baseline on same environment
   - Measure: cumulative reward, convergence speed, final policy quality
   - Ablation: test each modification independently
4. **Analyze results:**
   - Does the theoretical advantage manifest empirically?
   - Are there unexpected failure modes?
   - How sensitive to hyperparameters?

### Phase 5: Synthesis
**Goal:** Consolidate findings

1. **Summarize the novel algorithm:**
   - Name, update rules, pseudocode
   - Theoretical properties (convergence, error bounds)
   - Empirical results
2. **Compare to baselines:**
   - Table of algorithms vs. metrics
   - Qualitative analysis of when the new algorithm helps
3. **Identify next steps:**
   - Remaining limitations
   - Follow-up experiments
   - Potential extensions

## Example Innovation Workflows

### Workflow 1: "Improve Q-learning for Large State Spaces"
1. **Analysis:** Q-learning is tabular; doesn't scale. With FA, the deadly triad emerges.
2. **Design:** Combine Q-learning + target networks + experience replay + linear FA
   → This is essentially DQN (rediscovery validates the framework)
3. **Theory:** Extended Dvoretzky doesn't directly apply to FA case. Use J_PBE minimization with target network stabilization as justification.
4. **Implement:** DQN template from implementer skill
5. **Test:** Compare tabular Q-learning vs. DQN on grid worlds of increasing size

### Workflow 2: "Design Variance-Reduced Policy Gradient"
1. **Analysis:** REINFORCE has high variance due to full-return Monte Carlo estimates
2. **Design:** Add baseline (variance reduction) + critic (bootstrap) + advantage normalization
   → Actor-Critic with advantage function
3. **Theory:** Baseline invariance (E[grad ln pi * b(S)] = 0) ensures unbiasedness. TD error delta_t is an unbiased estimate of advantage A(s,a).
4. **Implement:** A2C template
5. **Test:** Compare REINFORCE vs. A2C on convergence speed

### Workflow 3: "Novel n-step Actor-Critic with Adaptive n"
1. **Analysis:** n-step returns trade off bias and variance. Fixed n is suboptimal.
2. **Design:** Adaptive n based on TD error magnitude. High |delta| → use smaller n (more bootstrap, less variance). Low |delta| → use larger n (less bias).
3. **Theory:** Show this is a valid stochastic approximation with bounded bias. The adaptive mechanism doesn't violate SA conditions if n is bounded.
4. **Implement:** Modified A2C with adaptive n-step returns
5. **Test:** Compare fixed-n vs. adaptive-n on environments with varying noise

## Evaluation Criteria for Novel Algorithms

| Criterion | Weight | How to Assess |
|-----------|--------|---------------|
| Mathematical soundness | High | Convergence proof or clear conditions |
| Empirical improvement | High | Better reward/convergence on benchmarks |
| Simplicity | Medium | Minimal additional hyperparameters |
| Generality | Medium | Works across problem settings |
| Novelty | Medium | Not a trivial variant of existing work |
| Interpretability | Low | Can explain why it works |

## Anti-Patterns to Avoid
1. **Complexity without justification:** Don't add mechanisms that don't address a clear limitation
2. **Theory-practice gap:** Don't claim convergence guarantees that only hold under unrealistic conditions
3. **Overfitting to benchmarks:** Test on diverse environments, not just one
4. **Ignoring baselines:** Always compare against well-tuned standard algorithms
5. **Hyperparameter sensitivity:** An algorithm that only works for specific hyperparameters is fragile

## Output Format
When proposing a novel algorithm:
1. **Motivation:** What limitation are we addressing?
2. **Proposed Algorithm:** Name, update rules, pseudocode
3. **Theoretical Analysis:** Convergence, error bounds, complexity
4. **Experimental Design:** Environments, baselines, metrics
5. **Results:** Empirical comparison (if implemented)
6. **Discussion:** When does this help? When does it fail?
7. **Next Steps:** Further improvements, open questions
