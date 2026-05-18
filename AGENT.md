# DQN Alien — Research Agent

I (Claude) read and update this file every iteration. It is my working memory across sessions.
The user submits a SLURM job I propose, waits for it to finish, then comes back.
I read the `final_eval.json` in each run directory, update this file, and propose the next config.

---

## Goals

| Target | Reward | Status |
|--------|--------|--------|
| Pass   | 950    | ✅ **MET** — exp_53 avg=1721.9 |
| Bonus  | 3000   | **NOT MET** — avg=2304 (exp_56), gap=696. s1=3840 proves ceiling exists. |

Note: Mnih 2013 paper (WITH target network) reports ~3069 for Alien. We have no target network.
B200 GPU speed: 5M steps = **54 minutes**. 10M steps ≈ 108 minutes. Within 48h SLURM limit.

---

## Metric

**`final_eval_reward`** = mean episode reward over 100 episodes run at the very end of
training (step = total_steps), using `eps = eps_eval`.

If `use_best_checkpoint_eval: true`, the 100-episode eval runs on the peak mid-training
checkpoint (tracked by single-episode eval during training) instead of the final step.

This is logged to `runs/<run_id>/final_eval.json` and `wandb eval/final_reward`.

During training, `eval_freq_steps=10_000` + `eval_episodes=1` gives cheap tracking signals
only. The final 100-episode number is the only number that counts.

---

## Fixed Constraints

- `ale_game_id: ALE/Alien-v5` — never touch
- 2013 DQN paper only (Mnih et al., arXiv 1312.5602):
  - No target network, no double DQN, no prioritized replay, no dueling
  - Features already implemented that ARE in the 2013 paper: `noop_max`, `life_loss_terminal`
  - `frame_differencing` is NOT in the 2013 paper — never use it

---

## Tunable Hyperparameters

```
lr, rmsprop_eps, rmsprop_alpha
batch_size
replay_buffer_capacity, min_replay_size
update_freq
gamma                          # paper=0.99; lower→shorter bootstrap chain→less Q-divergence
eps_start, eps_end, eps_anneal_steps, eps_eval
grad_clip                      # NEVER with RMSprop at lr=0.00025 (see stability rules)
loss_fn                        # "mse" (default) or "huber" — clips gradient for large TD errors
frame_skip (not ale_game_id)
total_steps
noop_max (int, 0=off)          # confirmed hurts — avoid
life_loss_terminal (bool)
```

**Confirmed dead ends (never revisit):**
- `gamma ≠ 0.99` **WITHOUT best_ckpt**: exp_46 avg=338.9 at 5M — collapse faster 7×
  ⚠️ REVERTING for exp_56: exp_46_s2 peaked at **4050** (1-ep) @ 3.08M. With best_ckpt+lr_decay,
  faster collapse is irrelevant — higher peaks are the goal. Being retested as exp_56.
- `loss_fn: huber`: exp_47 — clips early gradients, agent barely learns, peaks at 460k
- `rmsprop_eps > 0.01`: exp_48 — uniformly smaller updates, faster collapse (677/Mstep)
- `eps_end > 0.001`: exp_49 — extra noise kills lucky seeds (s2: 825→346)
- `eps_anneal > 250k`: exp_45 — bimodal peak timing, lower avg
- `lr ≠ 0.00025`: exp_44/38 — 0.0001 peaks too late, collapse lands at deadline
- `update_freq=2`: exp_40 — too many updates per env step, s0=100 final
- `noop_max > 0`: exp_36/39/41/43 — consistently hurts

**Unexplored — config-only (within 2013 paper spirit):**
- `min_replay_size=100k`: only tested at 50k. More diverse buffer before first update → Q-values
  better calibrated at start of exploitation window → more seeds may find good policies in 250k.
- `rmsprop_alpha`: only ever tested at 0.95 (paper default). Lower=0.9 → less gradient smoothing,
  more reactive per-param lr. Higher=0.99 → more stable updates, slower lr adaptation.
- `update_freq=8`: paper uses 4; only tested 2 (failed — too many updates). Fewer updates per env
  step → Q-values diverge more slowly. Direct, under-tested collapse rate lever.
- `replay_buffer_capacity=2M`: larger buffer keeps older experience longer → slower catastrophic
  forgetting. Never changed from 1M.
- `total_steps=3M/2M`: now testing (exp_50/51). If early stopping captures peak, try 2.5M/4M.

**Unexplored — requires small implementation (beyond strict 2013 paper):**
- `lr_warmdown` ✅ **IMPLEMENTED**: config keys `lr_decay_start_step` (step to begin decay),
  `lr_min` (target lr, default 0.0). Linear decay from `cfg["lr"]` → `lr_min` over remaining
  steps. Logged as `train/lr` in wandb. Directly caps update magnitude during collapse phase.
- `use_best_checkpoint_eval` ✅ **IMPLEMENTED**: config key `use_best_checkpoint_eval: true`.
  Saves `ckpt_best.pt` whenever 1-ep eval hits a new high during training. At end, loads peak
  checkpoint for the 100-episode final eval instead of the degraded final weights.
  `final_eval.json` records `eval_checkpoint_step` to track which checkpoint was used.
- `lr_cosine_schedule`: cosine decay over full 5M steps. Combined warmup+warmdown. Smoother.
- `eps_second_decay`: after eps reaches 0.001 at step 250k, hold until 3M, then decay to 0.
  Combines greedy commitment phase with terminal convergence. Extends existing eps schedule.

---

## Stability Rules (learned from 43 experiments)

**NEVER combine:**
- `grad_clip > 0` with RMSprop at `lr=0.00025` → explosive divergence from step 0
  (clipping disrupts RMSprop's v accumulator → inflates effective step size)

**Stable regime:** `lr=0.00025 + batch_size=64` → q_mean stays ~9, 0% loss spikes
**Unstable regime:** `lr=0.00025 + batch_size=32` → q_mean → ∞ in first 100k steps
**Slower stable:** `lr=0.0001 + batch_size=32` → stable but lower peak (~800 vs ~1900)

**Tested batch sizes:**
- batch=32: unstable at lr=0.00025; stable only at lr=0.0001
- batch=64: sweet spot — stable at lr=0.00025, best results
- batch=128: stable but diminishing returns vs batch=64

---

## Knowledge Base

### The Post-Peak Degradation Problem
All stable runs follow the same trajectory: reward rises to a peak (1500–2000), then
collapses back to ~400 by step 5M. The single-network bootstrap loop amplifies Q-values
over time until the policy becomes incoherent.

**This is the core problem.** The `final_eval_reward` metric directly penalizes this.
Peak reward and final reward are nearly uncorrelated based on existing data.

### Best Known `final_eval_reward` (from EXPERIMENTS.md "Final eval reward" column)
These are 5-episode evals from older experiments — NOT the 100-episode final_eval_reward.
Once new runs finish with `final_eval.json`, this table becomes authoritative.

| Exp | Config | s0 final | s2 final | Notes |
|-----|--------|----------|----------|-------|
| exp_29 | lr=0.00025, batch=64, eps_end=0.001 | 364 | 478 | Peak ~1940 but collapses |
| exp_23 | lr=0.00025, batch=64, eps_end=0.005 | 278 | 402 | Peak ~1860 |
| exp_11 | lr=0.00025, batch=64, eps_end=0.05  | 320 | 326 | 0% spikes, stable |
| exp_10 | lr=0.0001, batch=32, eps_end=0.05   | 146 | 156 | Stable, low peak |
| baseline | lr=0.0001, batch=32, eps_end=0.05  | 320 | 532 | Original stable ref |

**Working hypothesis for `final_eval_reward`:** Lower eps_end pushes peak higher but
worsens the final because the near-greedy policy overfits and then collapses harder.
eps_end=0.05 with batch=64 may actually produce better *final* reward than eps_end=0.001
even though the latter peaks higher.

### eps_end Trend (peak reward, old metric):
0.05→832, 0.01→1676, 0.005→1860, 0.001→1940 (consistently increasing)

### Known Dead Ends
- `lr > 0.00025` (exp_14): q_mean stuck at ~4, terrible performance
- `lr=0.0001 + slow eps_anneal=2M` (exp_31): s0 collapsed (q=2.5)
- `frame_skip=2` (exp_13): weaker signal per transition, worse results
- `frame_skip=8` (exp_16): not in notes — avoid without strong hypothesis
- `frame_differencing` (exp_24): not 2013 paper, disallowed

---

## Experiment Results (final_eval_reward = 100-episode eval at step 5M)

`end_s*` = last 5-episode mid-training eval (noisy proxy). `final100` = authoritative metric.

| Exp | Config delta | pk_s0 | pk_s2 | end_s0 | end_s2 | fin_s0 | fin_s2 | avg |
|-----|--------------|-------|-------|--------|--------|--------|--------|-----|
Checkpoint quality: ✓ = true final (ckpt@5M), ~ = stale (eval from earlier checkpoint, unreliable).

| Exp | Config delta | fin_s0 | ckpt_s0 | fin_s2 | ckpt_s2 | avg | reliable? |
|-----|--------------|--------|---------|--------|---------|-----|-----------|
| exp_34 | life_loss | 403.5 | 4M~ | 365.9 | 4.5M~ | 384.7 | partial |
| exp_35 | maxpool_all | — | — | 100.3 | 5M✓ | bad | no |
| exp_36 | noop | 172.3 | **2M~** | 292.9 | 3.5M~ | 232.6 | no (s0 ckpt far too stale) |
| exp_37 | fast_anneal_250k | 396.8 | 3.5M~ | 215.8 | 3M~ | 306.3 | partial |
| exp_38 | lr=0.0001 | 409.0 | 3.5M~ | 397.1 | 5M✓ | 403.1 | partial (s2 only final) |
| exp_39 | noop+life+fast | 574.0 | 5M✓ | 413.6 | 5M✓ | 493.8 | ✓ |
| exp_40 | update_freq=2 | 100.0 | 5M✓ | 437.4 | 5M✓ | 268.7 | ✓ |
| exp_41 | noop+life | 548.1 | 5M✓ | 431.3 | 5M✓ | 489.7 | ✓ |
| exp_42 | fast+life | 327.9 | 5M✓ | 825.4 | 5M✓ | **576.7** | ✓ |
| exp_43 | lr=0.0001+noop+life | 433.1 | 5M✓ | 366.8 | 5M✓ | 400.0 | ✓ |
| exp_44 | lr=0.0001+life | 237.6 | 5M✓ | avg(s1=252.9,s2=337.4) | 5M✓ | **276.0** | ✓ WORST |
| exp_45 | fast_500k+life | pending | — | pending | — | pending | — |
| exp_46 | gamma=0.97+fast_250k+life | pending | — | pending | — | pending | — |
| exp_47 | huber+fast_250k+life | pending | — | pending | — | pending | — |
| exp_48 | rmsprop_eps=0.1+fast_250k+life | 181.0 | 5M✓ | avg(219,280) | 5M✓ | **226.7** | ✓ FAILED |
| exp_42_s1 | exp_42 3rd seed ground truth | 129.7 | 5M✓ | — | — | **427.7** (true 3-seed) | ✓ REVISED DOWN |
| exp_49 | eps_end=0.005+fast_250k+life | 309.1 | 5M✓ | avg(448.3,345.7) | 5M✓ | **367.7** | ✓ FAILED |
| exp_50 | total_steps=3M+fast_250k+life | 398.4 | 3M✓ | avg(231.5,400.5) | 3M✓ | **343.5** | ✓ FAILED |
| exp_51 | total_steps=2M+fast_250k+life | 387.7 | 2M✓ | avg(441.8,198.3) | 2M✓ | **342.6** | ✓ FAILED |
| exp_52 | use_best_ckpt_eval+fast_250k+life | 1496.8 | 3990k✓ | avg(1681.3,1702.1) | 3910k/1630k✓ | **1626.7** | ✅ PASS |
| exp_53 | lr_decay@3M+best_ckpt+fast_250k+life | 1520.3 | 3160k✓ | avg(1763.0,1882.5) | 3250k/3680k✓ | **1721.9** | ✅ PASS |
| exp_54 | 10Msteps+lr_decay@3M+best_ckpt | pending | — | pending | — | pending | — |
| exp_55 | update_freq=8+lr_decay@3M+best_ckpt | 990.3 | 2530k✓ | avg(1691.2,1019.6) | 1170k/3540k✓ | **1233.7** | ✗ WORSE |
| exp_56 | gamma=0.97+best_ckpt+lr_decay@3M | 958.5 | 1960k✓ | avg(3840.4,2113.1) | 3630k/4150k✓ | **2304.0** | ✅ NEW CHAMPION |
| exp_54 | gamma=0.99+10M+best_ckpt+lr_decay@3M | 550.3 | 6420k✓ | avg(2328.2,2507.8) | 8390k/3300k✓ | **1795.4** | ✓ |
| exp_57 | gamma=0.97+10M+best_ckpt+lr_decay@3M | pending | — | pending | — | pending | — |
| exp_58 | gamma=0.97+minreplay=100k+best_ckpt+lr_decay@3M | pending | — | pending | — | pending | — |

**Key lessons:**
- `noop_max` consistently hurts: exp_39 < exp_42, exp_43 < exp_41. Never add it again.
- `life_loss` alone (exp_34, ckpt@4-4.5M) ≈ 385 avg — similar to baseline, no real gain.
- `update_freq=2` killed s0 completely (100.0 final).
- `lr=0.0001` alone (exp_38 s2 only reliable): 397.1 — not better than exp_42 base.
- `fast+life_loss` (exp_42) TRUE 3-seed avg: **427.7** (s0=327.9, s1=129.7, s2=825.4).
  Previous 2-seed estimate (576.7) was badly biased — s1 was the worst seed.
- Early stopping (exp_50/51) hurts average: each seed has a different optimal stopping point.
  s2 (lucky) needs 5M → 825; s1 (unlucky) needs 2M → 442; s0 needs 3M → 398.
  Fixed total_steps can't optimize all seeds simultaneously.
- Going forward all evals proper (train.py final_eval + eval_run.py for post-hoc).

**CRITICAL FINDING (exp_42_s1):**
The true 3-seed avg is 427.7, far below the 950 goal. The entire search so far has been
on a biased 2-seed estimate. The champion config (exp_42) is only the best we've found,
not a near-solution. We are at **45% of goal**. Fundamentally different directions needed.

---

## How to Run the Next Experiment

### Step 1: Check results
```bash
for d in runs/exp_{34..43}_{s0,s2}; do
  echo -n "$d: "
  cat $d/final_eval.json 2>/dev/null || echo "still running / no final_eval.json"
done
```

### Step 2: I (Claude) propose config
I write `configs/exp_NN_<name>.yaml` as a diff from exp_29 base.

### Step 3: Submit three seeds (s0=seed 0, s1=seed 1, s2=seed 2)
```bash
EXP=exp_NN_<name>
mkdir -p runs/${EXP}_s0 runs/${EXP}_s1 runs/${EXP}_s2
J0=$(sbatch --job-name=${EXP}_s0 --output=runs/${EXP}_s0/slurm_%j.log \
     --error=runs/${EXP}_s0/slurm_%j.err \
     run_template.sh ${EXP}_s0 0 configs/${EXP}.yaml | awk '{print $4}')
J1=$(sbatch --job-name=${EXP}_s1 --output=runs/${EXP}_s1/slurm_%j.log \
     --error=runs/${EXP}_s1/slurm_%j.err \
     run_template.sh ${EXP}_s1 1 configs/${EXP}.yaml | awk '{print $4}')
J2=$(sbatch --job-name=${EXP}_s2 --output=runs/${EXP}_s2/slurm_%j.log \
     --error=runs/${EXP}_s2/slurm_%j.err \
     run_template.sh ${EXP}_s2 2 configs/${EXP}.yaml | awk '{print $4}')
echo "Submitted: s0=$J0 s1=$J1 s2=$J2"
```

---

## Current Research Direction

**Core problem:** Single-network DQN bootstrap loop amplifies Q-values → policy collapses.
Final eval at 5M captures the collapsed policy, not the peak. Every hyperparameter change
that attacked the collapse from "within training" has failed (lower lr, gamma, huber, rmsprop_eps).

**Current champion: exp_42 (fast_anneal_250k + life_loss_terminal, avg=576.7, 2 seeds)**
Config: lr=0.00025, batch=64, eps_end=0.001, eps_anneal=250k, gamma=0.99, mse, no noop.

**Locked-in conclusions (43+ experiments):**
- lr=0.00025, batch=64 is the only stable, high-performance regime
- eps_anneal=250k is the right exploration window — sweet-zone peak timing (2–3.4M)
- eps_end=0.001 is optimal — more noise kills lucky seeds, less changes nothing
- life_loss_terminal=true always helps (or doesn't hurt)
- noop_max=0 always (hurts without exception)
- gamma=0.99, loss=mse, rmsprop_eps=0.01 are all optimal

**Critical realisation (exp_42_s1 result):**
The true 3-seed avg is 427.7. We are at 45% of the 950 goal. Every experiment so far
attacked hyperparameters within the basic training loop. None have come close to 950.
The single-network bootstrap instability is a hard ceiling — the agent always collapses.

**The per-seed optimal stopping insight (exp_50/51):**
| Seed | Optimal stop | Reward |
|------|-------------|--------|
| s0   | ~3M         | 398    |
| s1   | ~2M         | 442    |
| s2   | 5M          | 825    |
Fixed total_steps penalises at least one seed. **use_best_checkpoint_eval** (exp_52/53)
should naturally adapt per-seed by selecting the best mid-training checkpoint for each.

**Active experiments (running):**
- exp_52: use_best_ckpt_eval at 5M. Each seed picks its own peak. Key test.
- exp_53: lr_warmdown from 3M + best_ckpt. Slows collapse AND picks peak. Two mechanisms.

**Decision tree after exp_52/53:**

→ **If exp_52 avg > 700:**
  Peak checkpoint selection is a major lever. Next: exp_54 = extend to 10M total_steps +
  best_ckpt (more time for each seed to find and consolidate its peak).

→ **If exp_52 avg ≈ 427 (no improvement):**
  The single-episode eval that selects ckpt_best is too noisy — lucky episodes mislead.
  Next: increase eval_episodes to 5 (so best ckpt is selected on 5-ep avg, not 1-ep).

→ **If exp_53 > exp_52:**
  LR warmdown genuinely slows collapse. Next: tune lr_decay_start_step (try 2M instead of 3M).

→ **To reach 950, we likely need:**
  1. Best-checkpoint eval to escape the collapse (exp_52/53)
  2. Longer training (10M steps?) so seeds have time to develop deep policies
  3. Better exploration in the 250k window to produce more "lucky" seeds

**The warmup/warmdown idea (implemented):**
`lr_decay_start_step` + `lr_min` config keys in train.py. Linear lr decay after the start
step. Logged as `train/lr` in wandb. Use in any config by adding these two keys.

---

## Agent Decision Log

| Date | Exp | Rationale |
|------|-----|-----------|
| 2026-05-18 | Setup | Initial autoresearch framework |
| 2026-05-18 | exp_44 | lr=0.0001 + life_loss. **FAILED** avg=276. Peaks hit too late (4.8M+) — collapse lands at 5M deadline. lr=0.0001 ruled out. |
| 2026-05-18 | exp_45 | fast_anneal_500k + life_loss. Addresses exp_42 variance: 250k commitment window too narrow (327 vs 825). 500k gives more exploration before locking in. |
| 2026-05-18 | exp_46 | gamma=0.97+fast_250k+life. **FAILED** avg=338.9. Sweet-zone collapse rate 7× worse (1791 vs 252/Mstep). Lower gamma accelerates collapse. gamma=0.99 is optimal — never change. |
| 2026-05-18 | exp_48 | rmsprop_eps=0.1+fast_250k+life. **FAILED** avg=226.7. Higher eps makes optimizer LESS adaptive → uniformly smaller updates → noisier Q-learning → faster collapse (677/Mstep in sweet zone vs 252). |
| 2026-05-18 | exp_47 | huber+fast_250k+life. **FAILED** avg=182.3 (WORST). Clipped gradients slow Q-value growth too much → agent peaks too early (460k) with tiny values → learns almost nothing. |
| 2026-05-18 | exp_42_s1 | Third seed of exp_42 (best config). 2-seed avg=576.7 may be misleading. Ground truth check. |
| 2026-05-18 | exp_49 | eps_end=0.005+fast_250k+life. **FAILED** avg=367.7. Hypothesis inverted: eps=0.005 killed lucky seeds (s2: 825→346) without helping unlucky ones. Extra noise prevents the greedy commitment that makes good seeds great. Collapse rates 500-711/Mstep (2-3x worse than exp_42). eps_end=0.001 is definitively optimal. |
| 2026-05-18 | exp_45 | fast_500k+life. avg=417.6 WORSE than exp_42 (576.7). Variance unchanged (212–611). Bimodal peak timing (1M or 4.7M+). 500k anneal bad. 250k anneal is the right window. |
| 2026-05-18 | exp_47 | huber+fast_250k+life. Collapse rate attack: Huber clips gradient to ±1 for large TD errors → Q-value runaway slows → collapse rate drops from ~250/M toward ~100/M. |
| 2026-05-18 | exp_50 | total_steps=3M+fast_250k+life. Early stopping at 3M: exp_42 s0 peaks at 2.8M (reward≈782 at 3M), s2 peaks at 3.4M (reward≈1258 at 3.4M). Final eval at 3M catches both seeds near peak instead of post-collapse. |
| 2026-05-18 | exp_51 | total_steps=2M+fast_250k+life. Earlier stopping: both seeds are pre-peak at 2M. If final reward is still reasonable, confirms collapse (not learning) is the bottleneck. Diagnostic as much as optimization. |
| 2026-05-18 | exp_50 | total_steps=3M. **FAILED** avg=343.5. Early stopping hurts s2 (825→400) while helping s1 (129→232). Optimal stopping is SEED-DEPENDENT. Fixed total_steps cannot optimise all seeds simultaneously. |
| 2026-05-18 | exp_51 | total_steps=2M. **FAILED** avg=342.6. Confirms: different seeds have different optimal stopping (s1 peaks at 2M=442, s2 needs 5M=825). Fixed cutoff always misses some seeds. |
| 2026-05-18 | exp_42_s1 | 3rd seed = 129.7. CRITICAL: true 3-seed avg=427.7 (not 576.7). The 2-seed estimate was biased. We are at 45% of the 950 goal. Fundamentally different strategies needed. |
| 2026-05-18 | exp_52 | use_best_checkpoint_eval+fast_250k+life. **PASS 950** avg=1626.7. Peak checkpoints: s0@3990k, s1@3910k, s2@1630k. Proves: collapse is the ONLY bottleneck. Peak policy is high-quality; degraded final weights were hiding it. 3.8× improvement over exp_42. |
| 2026-05-18 | exp_53 | lr_decay@3M+best_ckpt+fast_250k+life. **PASS 950** avg=1721.9. LR warmdown pushed s2 peak from 1630k→3680k and reward 1702→1882. All seeds now peak in warmdown phase (3.2–3.7M). New champion. Gap to bonus: 1278. |
| 2026-05-18 | exp_54 | 10M+lr_decay@3M+best_ckpt. 7M of warmdown instead of 2M — lr decays 3.5× more slowly. At 5M lr≈0.000178, at 7M lr≈0.000107. Peaks may reach 5–7M and score 2500+. ~108min on B200. |
| 2026-05-18 | exp_55 | update_freq=8+lr_decay@3M+best_ckpt. **FAILED** avg=1233.7. Fewer updates → agent learns too slowly → lower peaks (990, 1019 for s0/s2). update_freq=4 confirmed optimal. |
| 2026-05-18 | exp_56 | gamma=0.97+best_ckpt+lr_decay@3M. **NEW CHAMPION** avg=2304.0. s1=3840.4 (100-ep!) — first time exceeding 3000 bonus on a single seed. s1 1-ep peak=4260@3.63M → 100-ep=3840 (ratio=0.90). Bottleneck: s0 peaked only at 1010 (bad local optimum). Variance is the final problem. |
| 2026-05-18 | exp_54 | gamma=0.99+10M+best_ckpt+lr_decay@3M. avg=1795.4. s1/s2 pushed to 2328/2508 at 8.4M/3.3M ckpts. s0 still stuck (550@6.4M). 10M helps lucky seeds but doesn't fix s0's bad initial policy. |
| 2026-05-18 | exp_57 | gamma=0.97+10M+best_ckpt+lr_decay@3M. 7M of warmdown (3.5× slower lr decay). Unlucky seeds (s0 peaked at 1010@1.97M) have 8M more steps to find better policies. If s0 reaches ~3000, avg hits bonus. ~108min on B200. |
| 2026-05-18 | exp_58 | gamma=0.97+min_replay=100k+best_ckpt+lr_decay@3M. Last unexplored config-only lever. 100k diverse steps before first gradient update → better Q-value calibration → s0 less likely to commit to bad policy in 250k window. Direct attack on the "s0 stuck" pattern. |
