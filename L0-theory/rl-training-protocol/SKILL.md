---
name: rl-training-protocol
description: >
  General-purpose training-protocol hygiene for deep RL with parameterized simulators — env-generator invariance, init/schedule pre-flight assertions, late-training drift handling, multi-seed gating, mechanism-attribution discipline, and PPO/data-pipeline hygiene. Project-agnostic.
layer: L0
domain: [general-rl, ml-training]
source-project: rl-escape-dense-forest
depends-on: [rl-methodology]
tags: [training, protocol, debugging, ablation, reproducibility, ppo, off-policy, actor-critic, vec-envs, curriculum, checkpointing]
---

# RL Training Protocol — Hygiene for Long-Horizon Deep RL

## Purpose & When to Use

Invoke this skill before launching, debugging, or interpreting any deep RL training run with at least one of:

- A **parameterized environment** (procedural generation, randomized obstacles, curriculum, density/scale randomization).
- A **long horizon** (≥100M env steps, multi-day wall-clock, multiple cycles).
- A **schedule** (entropy decay, learning-rate decay, curriculum, cyclic restart).
- A **mechanism claim** ("X reset is what works", "this reward term helps", "this architecture wins").

This skill encodes failure modes that recur in long PPO/SAC/MAPPO training. Read once, then run the **Pre-Flight Checklist** before every training launch and the **Pre-Claim Checklist** before every result-bearing edit (paper, README, commit message).

**The skill states *patterns* and *what to check*. It deliberately does not prescribe numeric thresholds**: thresholds are environment-specific and should be set by the project that owns the env, not borrowed from another project's training log. Where a knob's *direction* is universal (e.g., "decay entropy toward a floor"), that is stated; where the *value* is project-specific (e.g., the floor itself), that is left to the project.

This skill is rigid: when in doubt, follow the checklists. Adapting away the discipline is exactly how the failures recur.

## The Six Failure-Mode Classes

1. **Environment-generator asymmetry / leakage** — the policy learns something true *about your generator*, not your stated task.
2. **Init / schedule typos that satisfy the type system** — the run trains successfully and produces a broken policy.
3. **Late-training drift** — endpoint checkpoints quietly degrade after entropy/curriculum saturate.
4. **Single-seed claims** — paired comparisons where seed variance dominates the effect.
5. **Mechanism misattribution** — naming the cause without ablating each factor.
6. **PPO update / data-pipeline hygiene** — silent corruption of advantages, ratios, normalizers, or env semantics.

The protocol below addresses each class explicitly.

---

## Class 1 — Environment-Generator Invariance

**Rule.** Any procedural world-generator must be tested for the symmetries and contracts you expect, on samples drawn the same way the simulator draws them — not on the generator's spec. The policy *will* find every asymmetry you don't test for.

**Failure pattern.** A generator that nominally produces uniform / symmetric worlds quietly violates that property due to truncation, fill-order, or off-by-one errors. The policy correctly learns the asymmetry and ships it as "behavior." A separately written augmentation that was meant to neutralize the bias may exist but be unwired (dead branch). The bias survives until somebody plots the marginal of sampled worlds.

### Required pre-flight tests

Run all of these on a sample of worlds drawn through the simulator's actual sampler (not the spec). Sample size and tolerances are env-specific contracts — the env owner picks them with confidence intervals; this skill does not.

| Test | Condition |
|------|-----------|
| Marginal density per axis | Empirical density on each axis matches the declared distribution within the env's stated tolerance |
| Symmetry under declared reflections / rotations | Sample-empirical density of a transformed sample matches original within stated tolerance |
| Free-space feasibility | Independent path/feasibility classifier confirms the (start, goal) is reachable above the env's stated probability floor |
| Augmentation actually applied | Toggling the augmentation off changes the empirical sample distribution; an `assert_augmentation_applied` counter is logged at runtime |
| Density matches spec | Sampled obstacle density / count distribution matches the stated parameter |
| Vec-env autoreset semantics | The vec-env returns the post-reset observation on terminal steps if and only if the wrappers and learner expect it; mismatch silently corrupts bootstrap targets |

### How to apply

- Treat the env generator as a **first-class artifact**: it gets unit tests like any code, run in CI on every PR that touches the generator.
- Any "augmentation" or "randomization" branch must include an `assert_was_applied` instrumentation counter, surfaced to logs.
- When a behavioral quirk appears in evaluation ("policy biases left", "agents cluster on one side", "agents avoid the goal corridor"), **first hypothesis is generator asymmetry**, not policy pathology. Sample a batch of worlds and plot the marginals; if they aren't what you declared, fix the generator before touching the policy.
- Re-run the invariance suite whenever generator code changes, including "trivial" refactors. One-line changes (e.g., row-major vs uniform sampling) can introduce or remove an asymmetry.
- Vec-env autoreset is a generator-level contract: a learner that bootstraps over a reset boundary computes garbage advantages. Confirm explicitly which observation `step()` returns at terminals.

---

## Class 2 — Init and Schedule Pre-Flight Assertions

**Rule.** Network initialization values and schedule formulas must be asserted at startup against a known-good reference for *your* architecture and *your* schedule formula. Defaults that compile and train successfully can still be wrong. Most "this run looks weird" debugging traces back to a value here.

**Failure patterns (generic).**

- A `log_std` (or analogous policy-noise) init borrowed from a resume-default is too tight for fresh-start training; the policy never explores.
- A recurrent-cell gate bias is left at the framework default rather than the identity-favoring value the task needs; the network spends large parts of training learning what a better init gives for free.
- An observation-shape constant changes silently in a "cleanup" PR; checkpoints remain loadable in name but produce nonsense.
- A schedule formula divides by a near-zero curriculum fraction; the schedule collapses to ≈0 immediately.
- A schedule's "final" value is set ≥ "initial"; what was meant to be decay becomes ramp-up or constant; for stochastic policies without an entropy floor, action-std runs away.
- A reward term left enabled across a regime shift becomes harmful; the run logs look fine while performance silently degrades.

### Required startup assertions (template)

Run on every launch, gated only by `--smoke_test_only` for fast iteration. Cost should be small (well under a second on a CPU).

```python
def assert_training_config_sane(cfg, network, ckpt_meta=None):
    # Block 1: schedule math sanity
    assert cfg.curriculum_frac > 0.0, "curriculum_frac=0 breaks linear schedules"
    if cfg.ent_coef_final >= cfg.ent_coef:
        assert cfg.entropy_controller == "constant", \
            "ent_coef_final >= ent_coef without explicit constant flag → std runaway risk"
    assert cfg.lr_final <= cfg.lr_init, "non-decaying lr (typo?)"
    assert cfg.total_timesteps > cfg.resume_step, \
        "total_timesteps is absolute; resume_step >= total_timesteps exits immediately"

    # Block 2: init reasonableness against a known-good reference for THIS architecture
    # (Project-specific. The skill prescribes the assertion pattern, not the value.)
    network.assert_init_within_known_good_range()

    # Block 3: shape / checkpoint-metadata compatibility
    network.assert_shapes_match_config(cfg)
    if ckpt_meta is not None:
        for k in network.load_bearing_constants():
            assert ckpt_meta[k] == getattr(cfg, k), f"checkpoint/{k} mismatch"

    # Block 4: 3-point schedule probe (catches off-by-one / sign errors)
    for step_frac in (0.0, 0.5, 1.0):
        step = int(step_frac * cfg.total_timesteps)
        ent = cfg.entropy_at(step)
        lr = cfg.lr_at(step)
        assert 0.0 < ent <= cfg.ent_coef + 1e-6, f"entropy({step_frac:.0%}) = {ent}"
        assert cfg.lr_final - 1e-9 <= lr <= cfg.lr_init + 1e-9, f"lr({step_frac:.0%}) = {lr}"
```

### How to apply

- Run the assertion on every launch, including reruns.
- Treat any **load-bearing constant** (observation dim, action dim, recurrent hidden size, n_agents, sensor counts) as an interface, not a default. Document the set in your project's `CLAUDE.md` (or equivalent) and pin them in checkpoint metadata.
- When you change a default, write a migration plan first: which checkpoints break, which eval scripts must update, what the rollback path is.
- The 1k-step smoke test should compare logged entropy/lr at step 0, ≈mid, ≈end against the schedule formula. Discrepancy ≥ 1% → abort; the formula or the controller is wrong.
- A **known-good init reference** is a project artifact, maintained alongside the network code. It is fine for this artifact to differ across architectures — what matters is that *this project's* assertion file is in sync with *this project's* network.

---

## Class 3 — Late-Training Drift

**Rule.** Once entropy is pinned at its floor and curriculum is at its final difficulty, training enters a regime where multiple drift signals (action-std, KL between successive policies, gradient norm, clip fraction, eval gap) can all move adversely. **The endpoint checkpoint is often not the best checkpoint.** Track best-by-EMA, not last.

**Failure pattern.** After entropy and curriculum saturate, evaluation performance trends downward over tens of millions of steps while the loss curves look fine. Action-std rising is one symptom; it is not the only one. Picking the endpoint checkpoint ships a worse policy than picking by EMA.

### Required tracking

```python
class BestCheckpointTracker:
    """Tracks best-by-EMA episode-success metric and saves snapshot atomically."""
    def __init__(self, save_dir, ema_alpha=0.2, min_step=0):
        self.save_dir = save_dir
        self.ema_alpha = ema_alpha
        self.min_step = min_step  # ignore early checkpoints
        self.best_ema = -float("inf")
        self.ema = None

    def update(self, step, episode_success_metric, save_fn):
        if step < self.min_step:
            return
        self.ema = (episode_success_metric if self.ema is None
                    else self.ema_alpha * episode_success_metric
                         + (1 - self.ema_alpha) * self.ema)
        if self.ema > self.best_ema:
            self.best_ema = self.ema
            save_fn(self.save_dir / "checkpoint_best.pkl",
                    metadata={"step": step, "ema": self.ema,
                              "raw": episode_success_metric})
```

### Required telemetry (every mini-eval)

| Quantity | Why |
|----------|-----|
| Per-**episode** success metric | Per-timestep aggregates can rise while episodes degrade — they are not the same |
| EMA of the above | Saving signal robust to single-eval noise |
| Action-std mean | Drift early-warning for stochastic policies |
| KL between successive policies | Drift signal independent of action-std |
| Gradient norm | Divergence early-warning |
| PPO clip fraction (if PPO) | Update-step magnitude indicator |
| Deterministic-vs-stochastic eval gap | Diverging gap can signal over-sharpening |
| Steps since entropy floor / curriculum max | Context for the above |

### How to apply

- Save `checkpoint_best.pkl`, `checkpoint_last.pkl`, **and** every Nth periodic checkpoint. Cheap; lets post-hoc analysis pick the right one.
- For paper / shipped numbers, evaluate **best-by-EMA** unless you explicitly study drift. State the choice in methods.
- Treat per-timestep metrics as monitoring only. Use a per-episode metric for the saving signal.
- No single drift signal is law. Co-monitor several; an alert fires when **any two** move adversely while the EMA falls.
- For cyclic schedules, track best within each cycle and best across cycles separately.

---

## Class 4 — Multi-Seed Gating

**Rule.** No paired comparison or "X is better than Y" claim leaves the training rig with a single seed. The minimum number of seeds is set by the **effect size you want to claim** and the **noise of your eval** — state and justify it. Do not borrow another paper's number.

**Failure pattern.** A single-seed pilot produces an effect; the team writes a draft around it; later seeds reduce or invert the effect. Without a pre-declared seed budget, the temptation is to retrofit the hypothesis to whichever seeds agree.

### Required protocol

| Claim type | Seed protocol |
|------------|---------------|
| "X works" (existence) | 1 seed sufficient |
| "X reaches Y%" (point estimate) | n seeds (justified); report mean ± SD or CI |
| "X > Y" (paired comparison) | n seeds with **paired** comparison: report Δ_i = X_i − Y_i for each seed i, then mean Δ ± CI |
| "X causes Y" (mechanism) | n seeds × factorial — see Class 5 |
| Held-out validation | Independent seed set, sample size justified by your eval-noise estimate |

### How to apply

- Budget seeds in the experiment plan up front. A factorial with k conditions and n seeds is k×n runs; plan for it.
- "Within error bars" is a publishable finding ("not separable"), not a bug. Report it.
- Use **paired** comparisons when seeds are paired (same seed used for both conditions). The variance of paired Δ_i is typically much smaller than a pooled-SE comparison; use a CI on the mean Δ, not pooled SE alone.
- When seeds disagree with the single-seed pilot, **the seeds win**. Do not retrofit the hypothesis.
- Use **independent** seeds for training and eval. Reusing the training seed for eval is a leak.
- Report **stochastic eval alongside deterministic** when the deployed policy is stochastic. A deterministic-only number can flatter a policy whose value comes from its noise.

---

## Class 5 — Mechanism Attribution Discipline

**Rule.** A method's name must match what was ablated. If you call something "X-reset", you must have ablated X-reset against X-not-reset, holding everything else constant. If the factorial does not isolate the lever, the method is a **bundle**, and the published name should say so.

**Failure pattern (generic, regime-dependent reward).** A reward term is added during early training because it provides a useful gradient signal at the easy end of the curriculum. The team carries the reward term forward as the curriculum advances. At the harder end of the curriculum, the same term encourages a behavior (e.g., excess speed near obstacles) that is now *harmful*. Performance silently caps below the achievable ceiling. A removal ablation **at the deployment regime** (not at the regime where the term was added) reveals a large positive effect from removing it. The fix is a regime-conditional reward composition.

**Failure pattern (generic, bundled "mechanism").** A team names a method by the most salient knob in a multi-knob change ("entropy-warm-restart"). A factorial later finds the named knob's individual effect is non-separable from seed variance, while the *bundle* of knobs has a real end-to-end effect. The honest correction is to retract the mechanism claim and rename the method as a bundle ("the restart procedure"), keeping the empirical end-to-end gain but dropping the false attribution.

### Required protocol

Before claiming a mechanism:

1. **Name the lever explicitly.** One lever per claim.
2. **Construct an ablation that toggles only that lever.** Use config flags. If the lever has multiple knobs, make each a separate flag and run the full factorial.
3. **n seeds per cell, justified.** Buy fewer cells, not fewer seeds.
4. **Report effect ± CI.** If the CI crosses zero, the mechanism is "not separable from noise" at this n. That is a real result; do not name the method after it.
5. **Re-ablate when the regime shifts.** A reward term, schedule, or auxiliary loss that helped at curriculum stage A may hurt at stage B. Re-run reward / loss ablations after curriculum changes, architecture changes, or major hyperparameter shifts.

### Naming hygiene

| Smell | Probable issue |
|-------|----------------|
| Method names a *result* not a *mechanism* ("cyclic improvement", "stability boost") | No isolated lever |
| Method bundles ≥2 knobs but is named for one ("X-warm-restart") | Run a factorial; rename to "the X procedure" if the named knob is non-separable |
| The ablation table reports only end-to-end ("with vs without method") | You have not isolated the mechanism; reviewers will ask |

### How to apply

- Maintain a **mechanism ledger** for every named contribution: lever, factorial design, seeds, effect ± CI, retraction status.
- Reward composition is high risk: every reward term needs a removal ablation at the **deployment regime**, not just the regime where it was added.
- Architecture ablations must hold parameter count fixed, or report both matched and unmatched conditions; a capacity confound is the most common silent contributor to apparent architecture wins.
- When in doubt between "mechanism" and "bundle", choose bundle. It costs nothing to name a bundle correctly; it costs a retraction to name a bundle as a mechanism.

---

## Class 6 — PPO Update and Data-Pipeline Hygiene

**Rule.** The PPO update and the data pipeline that feeds it have several silent corruption modes that can invalidate a training run long before checkpoint selection matters. Each corruption class needs a logged signal and a documented threshold.

**Failure pattern.** Training proceeds; loss curves look fine; eval is flat or worse. The problem is upstream of the policy: GAE λ changed, advantages are unnormalized, IS-ratios sit at the clip boundary every minibatch, observation normalizer was reset on resume so the network sees a shifted input distribution, or the vec-env returns the wrong observation at terminals so bootstrap targets are computed across episode boundaries.

### Sub-rules

1. **GAE λ ablation.** Don't change λ silently across runs. Re-ablate when reward magnitude or episode length changes; both shift the bias/variance trade-off.
2. **Advantage normalization.** Normalize per-batch (or per-minibatch); document which. Inconsistency between train and eval, or between fresh-start and resume, is silent.
3. **PPO IS-ratio clipping & clip fraction.** Log the fraction of samples at the clip boundary. Sustained high clip fraction means the policy is moving too fast per update; reduce learning rate or epochs. Persistently low clip fraction with no learning means the policy is barely moving; check advantages.
4. **KL-divergence early-stop.** Compute `KL(π_old, π_new)` per epoch; abort the epoch on excursion above a documented threshold. Unbounded KL within an update is a divergence path PPO is supposed to prevent.
5. **Gradient clipping.** Log pre-clip gradient norm; clip to a finite value; sudden norm spikes are a divergence signal even when training proceeds.
6. **Vec-env autoreset semantics.** Confirm whether `step()` returns post-reset or pre-reset observations on terminal steps. Bootstrap and advantage computation must agree with this contract; mismatch silently corrupts targets.
7. **Observation normalizer state on resume.** Restore running-mean / running-variance from the checkpoint; never recompute from scratch on the warm-start distribution.
8. **Reward normalizer state on resume.** Same: persisted across resume; not silently re-initialized.

For **off-policy algorithms (DDPG, TD3, SAC, DQN)**: replay-buffer staleness, target-network sync interval, importance-sampling correction when behavior ≠ target, and prioritized-replay correction are analogous concerns. They are not duplicated here; see `rl-algorithms` (L1) and the algorithm-specific platform skills.

### How to apply

- Every PPO trainer logs `clip_fraction`, `approx_kl`, `pre_clip_grad_norm`, and `advantage_mean`/`advantage_std` per update step.
- KL-early-stop and grad-clip thresholds live in the run manifest. Changing them is a config change, not a tweak.
- The vec-env autoreset contract is asserted on the first 1k steps and on every wrapper change.
- The obs/reward normalizer is part of the checkpoint, not the trainer state. A resume that fails to restore it is a known-bad run; abort.

---

## Pre-Flight Checklist

Three sub-lists. Run them at the corresponding event.

### Always run (every launch)

- [ ] Class 2: `assert_training_config_sane` passes; 1k-step smoke test logs match schedule formula at three points.
- [ ] Class 3: Trainer instantiates `BestCheckpointTracker`; per-episode metric, EMA, action-std, KL, grad-norm, eval-gap logged.
- [ ] Class 4: Seed assigned, written to manifest, run is part of an n-seed plan if it will produce a paired comparison.
- [ ] Class 5: If this run will be cited as evidence for a mechanism, the lever and factorial counterpart are identified.
- [ ] Class 6: PPO hygiene logs wired (clip fraction, approx KL, pre-clip grad norm, advantage stats); KL-early-stop and grad-clip thresholds in manifest.

### On env / generator change

- [ ] Class 1: Generator invariance suite green (marginals, declared symmetries, augmentation wired, density matches spec).
- [ ] Class 1 / Class 6: Vec-env autoreset semantics smoke test passes; wrappers and learner agree.

### On resume

- [ ] Class 2: Load-bearing constants pinned in checkpoint metadata; resume-path obs/action dims match.
- [ ] Class 2: `total_timesteps` is absolute (`= resume_step + extra_steps`), not relative.
- [ ] Class 6: Observation normalizer and reward normalizer states restored from checkpoint; not re-initialized.
- [ ] Class 5: If the resume changes any factor in a previously-claimed mechanism, the bundle is re-stated and the prior claim re-tested.

## Pre-Claim Checklist (every result-bearing edit)

Before adding a number to a paper, README, commit message, or report:

- [ ] The metric is **per-episode**, not per-timestep, unless explicitly stated otherwise.
- [ ] The checkpoint is **best-by-EMA** (or endpoint with explicit justification).
- [ ] The selected checkpoint is **independently held-out evaluated** — selection on the same seeds you report numbers from is selection bias.
- [ ] n seeds, justified; report mean ± CI. Paired Δ for paired comparisons.
- [ ] **Stochastic eval reported alongside deterministic** when the deployed policy is stochastic.
- [ ] If a mechanism claim: factorial ablation table cites isolated levers; bundles named as bundles.
- [ ] If parameter count matters (architecture comparison): matched-param condition reported.
- [ ] Normalizer state, config hash, and run manifest recorded; numbers traceable to a specific commit + checkpoint.
- [ ] No stale-generator results re-enter the paper after a generator fix.

## Anti-Patterns

| Anti-pattern | Why it bites |
|--------------|--------------|
| "Train till `total_timesteps`, eval last checkpoint" | Class 3 drift; endpoint is often not best |
| "Single-seed pilot, scale up later" | Class 4; the pilot becomes the headline before n>1 lands |
| "It worked early so we kept it" (rewards, schedules, auxiliary losses) | Class 5; regime-dependent helpers become harmful |
| "Per-timestep success rate looks great" | Not the same as episode success; use the per-episode metric |
| "Cleanup PR — bumped a load-bearing default" | Silent checkpoint break; load-bearing constants are interfaces |
| "We didn't ablate but the gain is obvious" | Class 5; "obvious" mechanisms are routinely wrong on ablation |
| "Resumed without restoring obs/reward normalizer" | Class 6; network sees a shifted input distribution silently |
| "Ignored clip fraction / grad norm because loss looked fine" | Class 6; loss-curve smoothness is not the divergence signal |
| "Changed GAE λ to fit a result" | Class 6 / Class 5; bias/variance trade-off changed silently, no ablation |
| "Reported deterministic eval as the deployment number" | Class 6; deployment is the stochastic policy unless you switched it |
| "Constant entropy because the policy looks fine" | Context-dependent: for stochastic policies without an entropy floor, std often runs away over long horizons; verify in your regime |

## Output Format

When this skill is invoked:

1. State which **Class** the user's situation falls under (1–6), or "Pre-flight" / "Pre-claim".
2. Walk the relevant checklist; identify which items are unmet.
3. For each unmet item, give the concrete check to run *before* proceeding (the diagnostic, not a borrowed numeric threshold).
4. If the user is mid-run with a problem (drift, misattribution, surprise result), state the failure mode by name and the diagnostic that confirms it.
5. Do not propose new training without the pre-flight green.

## Cross-References

- `rl-methodology` (L0): convergence theory, SA learning-rate conditions, deadly triad — upstream of why entropy/lr schedules need to satisfy `sum=inf, sum-sq<inf` properties; also the home of off-policy convergence theory.
- `rl-algorithms` (L1): PyTorch scaffolds where these protocol items get implemented; algorithm-specific replay/target-network/IS hygiene lives here.
- `rl-implementer` (L1): use this protocol when implementing a new algorithm to bake the assertions in from the start.
- Platform skills (`rl-tools`, `isaaclab-rl-training`, `gpd-training-evaluation`, `stable-baselines3`): apply this protocol on top of platform-specific trainers.
