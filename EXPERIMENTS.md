# DQN Alien — Experiment Log

Stats files: `runs/<run_id>/training_stats.jsonl` (previous run archived as `training_stats_prev.jsonl`)
Key metrics: `loss`, `grad_norm`, `q_mean`, `q_max`, `q_std`, `target_mean`, `td_error`, `eval_reward`.

**Eval protocol change (exp_31+):** eval_episodes increased from 5 → 50 per checkpoint.
Previous experiments used 5 episodes — peaks are noisier. From exp_31 onward, peaks are averages over 50 episodes (much more reliable).
Primary metrics: **Peak** (max eval), **Top20avg** (avg of best 20 evals), **%>1000** (fraction of training above 1000).

---

## Baseline — lr=0.0001, grad_clip=10, eps_end=0.05
**Config:** original `configs/base.yaml` (before lr was changed to 0.00025)
**Runs:** run_0 (seed 0), run_2 (seed 2) — data in `training_stats_prev.jsonl`
**Status:** COMPLETE

| Metric | run_0 | run_2 |
|---|---|---|
| Peak eval reward | 792 @ 4.5M | 864 @ 3.9M |
| Final eval reward | 320 | 532 |
| Loss spikes >1 after 1M | 31 | 36 |
| Spike rate after 1M | 3.3% | 3.5% |
| Max q_mean | 159 | 86 |

**Verdict:** Good peaks, acceptable stability. Q-values diverge early (step 206-207k)
but grow gradually (0.006→0.05→0.5→42→137→159) and self-correct. The gradual growth
allows training to proceed. This is the reference target.

---

## exp_01 — Same lr=0.0001, eps_end changed to 0.1
**Config:** `configs/exp_01_lower_lr.yaml`
**Runs:** run_0 (seed 0), run_2 (seed 2) — data in `training_stats.jsonl`
**Status:** COMPLETE — SLIGHTLY WORSE than baseline

| Metric | run_0 | run_2 |
|---|---|---|
| Peak eval reward | 746 | 670 |
| Final eval reward | 162 | 228 |
| Loss spikes >1 after 1M | 53 | 61 |
| Spike rate after 1M | 5.1% | 6.0% |
| Max q_mean | 77 | 79 |

**Verdict:** Worse than baseline across the board. The change from eps_end=0.05→0.1
appears to hurt. Note: lower q_mean (77 vs 159) is a different divergence pattern — less
extreme bootstrap amplification but more instability in the long run.

---

## exp_02 — lr=0.00025, grad_clip=5.0
**Config:** `configs/exp_02_lower_clip.yaml`
**Status:** SUPERSEDED — not separately evaluated; ablation sweep revealed lr=0.00025 causes divergence

---

## Ablation sweep exp_03–09 — FAILED (wrong lr=0.00025)

**Root cause:** All 7 ablation configs used `lr: 0.00025`. With this lr, Q-values diverge
explosively from the **first training step** (step 202k): q_mean jumps 0.03 → 123 → 1176
in two logging intervals. The bootstrap loop amplifies too fast to self-correct.

With `lr: 0.0001`, the same bootstrap instability grows slowly (0.006 → 0.05 → 0.5 → 42 → 137)
and stabilizes. The faster learning rate (0.00025) turns a gradual process into an explosion.

| ID | Change | peak s0 | peak s2 | spikes/total | spike% | verdict |
|---|---|---|---|---|---|---|
| exp_03 | eps_end 0.1→0.05 | 792 | 424 | 1222/1222, 1138/1338 | 100%, 85% | FAILED |
| exp_04 | buffer 200k→500k | 586 | 622 | 1109/1302, 1132/1326 | 85%, 85% | FAILED |
| exp_05 | batch_size 32→64 | 640 | 820 | 1167/1362, 855/1382 | 86%, 62% | FAILED |
| exp_06 | lr 0.00025→0.0002 | 598 | 496 | 1037/1221, 399/1347 | 85%, 30% | FAILED |
| exp_07 | eps_anneal 1M→500k | 488 | 518 | 1140/1358, 1138/1342 | 84%, 85% | FAILED |
| exp_08 | clip=5 + eps_end=0.05 | 508 | 716 | 1114/1342, 1178/1371 | 83%, 86% | FAILED |
| exp_09 | batch64+buf500k+eps05 | 698 | 594 | 1214/1414, 818/1322 | 86%, 62% | FAILED |

Note: exp_06_s2 at lr=0.0002 had only 30% spike rate vs 85%+ elsewhere — approaching the
stable regime. Confirms the stability threshold is somewhere between lr=0.0001 and lr=0.0002.

---

## exp_10 — CORRECT lr=0.0001 + eps_end=0.05 (root cause fix)
**Config:** `configs/exp_10_lr1e4_eps05.yaml`
**Runs:** run_0 (seed 0), run_2 (seed 2)
**Status:** COMPLETE

### What changed
- `lr: 0.0001` — restored to original stable value
- `eps_end: 0.05` — matches the original stable run (slurm log confirmed eps=0.050 at end)
- `grad_clip: 10.0` — same as baseline

### Hypothesis
The original stable runs used lr=0.0001. All ablations with lr=0.00025 failed immediately.
This experiment should replicate the original stable result (peak ~800+) and confirm that
lr=0.0001 is the controlling variable.

### Results

| Metric | run_0 | run_2 |
|---|---|---|
| Peak eval reward | 588 | 524 |
| Final eval reward | 146 | 156 |
| Loss spikes >1 after 1M | 79 | 53 |
| Spike rate after 1M | 7.3% | 4.8% |
| Max q_mean | ~19 | ~19 |

**Verdict:** Stable training (q_mean stable at ~19 vs baseline's inflated 159), confirming
lr=0.0001 prevents catastrophic divergence. Peak rewards (588/524) are below the baseline
(792/864) — likely due to stochastic variation in this seed/run rather than a fundamental
difference, since the early training trajectory is identical to the baseline (same q_mean,
loss, grad_norm at every step). Spike rate (4.8–7.3%) is comparable to baseline (3.3–3.5%).
The lr=0.0001 + eps_end=0.05 setting is confirmed stable and is the correct base for future experiments.

---

---

## exp_11 — Larger batch size (64) ★ BEST SO FAR
**Config:** `configs/exp_11_batch64.yaml`
**Runs:** exp_11_s0 (seed 0), exp_11_s2 (seed 2)
**Status:** COMPLETE

| Change | batch_size 32 → 64 |
|---|---|
| Hypothesis | Half the gradient variance per update → smoother Q-value estimates |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 832 | 782 |
| Final eval reward | 320 | 326 |
| Spike rate after 1M | 0% | 0% |
| Final q_mean | 9.3 | 8.7 |

**Verdict:** Best result across all experiments. batch=64 eliminates Q-value divergence
entirely (0% spikes) even with lr=0.00025 — the larger batch keeps gradients stable enough
that no clipping is needed. Peak 832 beats the original baseline. q_mean ~9 is much more
accurate than baseline's ~19. Sweet spot: 64 provides stability without the diminishing
returns of 128.

---

## exp_12 — Larger batch size (128)
**Config:** `configs/exp_12_batch128.yaml`
**Runs:** exp_12_s0 (seed 0), exp_12_s2 (seed 2)
**Status:** COMPLETE (s0/s2 at ~4.4M steps)

| Change | batch_size 32 → 128 |
|---|---|
| Hypothesis | 4x batch for very stable gradients — checks if batch=64 gains hold or plateau |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 748 | 678 |
| Final eval reward | 352 | 204 |
| Spike rate after 1M | 0% | 0% |
| Final q_mean | 9.0 | 9.9 |

**Verdict:** Stable (0% spikes, same q_mean as exp_11) but strictly lower peaks than
batch=64. Diminishing returns: doubling from 64→128 costs performance, likely because
each update is more conservative and learning slows down.

---

## exp_13 — Smaller frame skip (2)
**Config:** `configs/exp_13_frameskip2.yaml`
**Runs:** exp_13_s0 (seed 0), exp_13_s2 (seed 2)
**Status:** RUNNING (~3.1M/5M steps)

| Change | frame_skip 4 → 2, update_freq 4 → 2 (keeps one train per agent action) |
|---|---|
| Hypothesis | Finer action control, 2× agent decisions per 5M frames |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 602 | 640 |
| Final eval reward | 266 | 200 |
| Spike rate after 1M | 1% | 0% |
| Final q_mean | 2.3 | 1.4 |

**Verdict (partial):** Lower performance so far. q_mean ~2 suggests the agent gets very
little signal per actual frame — with only 2-frame repeats, each transition captures less
meaningful state change, making Q-value learning shallower. Still running.

---

---

## exp_14 — Higher lr (0.0005) + batch=64
**Config:** `configs/exp_14_lr5e4_batch64.yaml`
**Status:** STOPPED EARLY (at ~4.2M steps) — DID NOT HELP

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 272 | 324 |
| Spike rate after 1M | 0% | 0% |
| q_mean at stop | 3.9 | 3.9 |

**Verdict:** Doubling lr to 0.0005 killed performance even with batch=64. q_mean stuck at
~4 (vs ~9 for exp_11) — updates too large, Q-values never consolidate. lr=0.00025 is the
ceiling for this setup. Stopped early, no recovery in sight.

---

## exp_15 — Huber loss + batch=64
**Config:** `configs/exp_15_huber_batch64.yaml`
**Status:** STOPPED EARLY (at ~4.3M steps) — DID NOT HELP

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 450 | 594 |
| Spike rate after 1M | 0% | 0% |
| q_mean at stop | 3.9 | 3.9 |

**Verdict:** Huber loss underperforms MSE here. Same low q_mean (~4) as exp_14 —
Huber's capped gradient for large TD errors is too conservative; MSE's stronger signal
on large errors is what drives bootstrap convergence in this setting. Stopped early.

---

## exp_16 — frame_skip=8 + batch=64
**Config:** `configs/exp_16_skip8_batch64.yaml`
**Status:** RUNNING (~4.4M/5M steps) — OK

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 702 | 622 |
| Last eval reward | 282 | 340 |
| Spike rate after 1M | 6% | 10% |
| q_mean now | 12.9 | 11.4 |

**Verdict:** Decent but below exp_11 (832/782). Longer repeats introduce more instability
(6-10% spikes vs 0%) and slightly lower peaks. frame_skip=4 remains the sweet spot.

---

---

## exp_17 — Faster epsilon anneal (500k) + batch=64
**Config:** `configs/exp_17_fast_anneal_batch64.yaml`
**Status:** COMPLETE — HIGH VARIANCE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 854 | 414 |
| Last eval reward | 264 | 200 |
| Spike rate after 1M | 0% | 0% |
| q_mean at end | 10.9 | 3.9 |

**Verdict:** Highest single peak so far (854) but catastrophic seed variance — s2 collapsed
(q=3.9). Fast annealing locks in a mediocre policy if the agent hasn't learned enough by
500k steps. Promising direction but needs lower eps_end to give a fallback (→ exp_21).

---

## exp_18 — Lower eps_end (0.01) + batch=64 ★ NEW BEST
**Config:** `configs/exp_18_eps01_batch64.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | **1676** | 730 |
| Last eval reward | 328 | 280 |
| Spike rate after 1M | 0% | 0% |
| q_mean at end | 11.4 | 11.2 |

**Verdict:** s0 hit **1676** at steps 4,915k–4,916k (two consecutive evals — confirmed real).
That's 2× the previous best (854). The agent clearly has a 1676-capable policy but hasn't
fully stabilised at that level — it drops back to 274–652 afterward. s2 is lower (730) but
healthy. eps_end=0.01 is the key: near-greedy exploitation lets a good policy execute
without noise disrupting it. exp_22 (8M steps) targets making this sustained.

---

## exp_19 — Intermediate lr (0.0003) + batch=64
**Config:** `configs/exp_19_lr3e4_batch64.yaml`
**Status:** KILLED (~3.6M steps) — DID NOT HELP

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 428 | 598 |
| Spike rate after 1M | 0% | 4% |
| q_mean at kill | 3.9 | 3.9 |

**Verdict:** Same failure mode as exp_14 — q_mean stuck at ~4, same as the 0.0005 run.
Even a 20% lr increase above 0.00025 is enough to prevent Q-values from developing
properly. lr=0.00025 is a hard ceiling; don't go higher.

---

---

## exp_20 — Slower epsilon anneal (2M) + batch=64
**Config:** `configs/exp_20_slow_anneal_batch64.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 644 | 642 |
| Last eval reward | 398 | 248 |
| Spike rate after 1M | 0% | 0% |
| q_mean now | 10.1 | 7.5 |

**Verdict:** Below exp_11. Exploring longer before exploiting doesn't help — earlier policy
refinement beats more random exploration. Slow anneal is a dead end.

---

---

## exp_21 — Fast anneal (500k) + eps_end=0.01 + batch=64
**Config:** `configs/exp_21_anneal500k_eps001.yaml`
**Status:** COMPLETE — MIXED (seed collapse again)

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 264 | 1040 |
| Last eval reward | 140 | 204 |
| Spike rate after 1M | 0% | 0% |
| q_mean at end | 2.3 | 11.8 |

**Verdict:** s0 collapsed (q=2.3 — same failure pattern as every fast-anneal seed failure).
s2 reached 1040 which is decent. Fast anneal continues to be unreliable regardless of eps_end.
Abandon this direction — the variance is too high to be useful.

---

## exp_22 — eps_end=0.01 + 8M steps + batch=64
**Config:** `configs/exp_22_eps001_8M.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 1676 @ 4.92M | 1022 @ 6.73M |
| Final eval reward | 534 | 310 |
| Spike rate after 1M | 0% | 0% |

**Trajectory (1M intervals):**
| Step | s0 | s2 |
|---|---|---|
| 1M | 186 | 158 |
| 2M | 258 | 270 |
| 3M | 248 | 206 |
| 4M | 492 | 396 |
| 5M | 506 | 280 |
| 6M | 248 | 258 |
| 7M | 476 | 264 |
| 8M | 534 | 310 |

**Verdict:** Longer training (8M vs 5M) didn't prevent or reverse policy degradation.
s0 peaked at 1676 @ 4.92M — same as exp_18, not better than exp_23 (1860). After peaking,
both seeds crashed hard and never recovered. The degradation is fundamental to the single-network
bootstrap instability, not a data-starvation issue. eps_end=0.005 (exp_23) beats eps_end=0.01
even though it peaked earlier. **8M steps not worth the compute cost.**

---

## exp_23 — eps_end=0.005 + batch=64 ★ NEW BEST
**Config:** `configs/exp_23_eps0005.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | **1860** | 1128 |
| Last eval reward | 278 | 402 |
| Spike rate after 1M | 0% | 0% |
| q_mean at end | 11.6 | 12.6 |

**Verdict:** New all-time best — 1860 at steps 2,922k–2,923k (two consecutive evals).
Above 800 for 13 evals across 2.3M–4.0M steps (sustained high-scoring window).
Clear trend: lower eps_end → higher peaks (0.05→832, 0.01→1676, 0.005→1860).
Policy fades after 4M — the agent learns a great policy then slowly unlearns it.
Next step: try eps_end=0.001 and/or longer training to lock in the peak.

---

---

## exp_24 — Frame differencing + eps_end=0.005 + batch=64
**Config:** `configs/exp_24_framediff.yaml`
**Runs:** exp_24_s0 (job 14375681), exp_24_s2 (job 14375682)
**Status:** RUNNING (~1M/5M steps)
*Note: original jobs 14375627/628 cancelled due to replay buffer bug (always returned absolute frames
during training while FrameStack used diff during collection). Fixed and resubmitted.*

| Change | frame_stack_mode stack→diff (4 motion-encoded channels) |
|---|---|
| Hypothesis | Alien's aliens and bullets are moving targets — explicit Δframe encoding helps the agent track velocity without inferring it from two absolute frames |

| Metric | s0 | s2 |
|---|---|---|
| Best so far (@ ~1M) | 162 @ 202k | 158 @ 386k |
| Current | 74 @ 989k | 100 @ 928k |

**Verdict:** KILLED at ~2.9M steps. Max ~208 across both seeds, q_mean stuck at 4.3–4.7.
Frame diffs discard the absolute positional information the agent needs for localization.
**Dead end — consistent with exp_26.**

---

## exp_25 — 8 stacked frames + eps_end=0.005 + batch=64
**Config:** `configs/exp_25_8frames.yaml`
**Runs:** exp_25_s0 (job 14375683), exp_25_s2 (job 14375684)
**Status:** RUNNING (~1.1M/5M steps)
*Note: original jobs cancelled due to replay buffer bug (frame_history hardcoded to 4 → network
expected 8 channels but got 4). Fixed and resubmitted.*

| Change | num_frames 4→8 (network first conv 4→8 input channels) |
|---|---|
| Hypothesis | 8 frames = ~0.5s of history at skip=4. Longer temporal context for tracking alien trajectories and planning multi-step dodges |

| Metric | s0 | s2 |
|---|---|---|
| Best so far (@ ~1.1M) | 566 @ 929k | 300 @ 428k |
| Current | 292 @ 1.1M | 56 @ 1.14M |

**Final verdict:** s0 peaked at **1232 @ 4.93M** (very late peak — crashed to 118 @ 4M then
recovered!). s2 collapsed early (q=2.5). 8 frames peaks much later than 4 frames but lower
(1232 vs 1860/1940). Interesting recovery behavior but not competitive with best 4-frame configs.

---

## exp_26 — 8 diff frames + eps_end=0.005 + batch=64
**Config:** `configs/exp_26_8frames_diff.yaml`
**Status:** KILLED (~750k steps) — CONFIRMED BAD

| Metric | s0 | s2 |
|---|---|---|
| Best eval reward | 140 @ 261k | 140 @ 302k |
| At kill | 140 @ 734k | 104 @ 759k |

**Verdict:** Same failure as exp_24 — diff mode hurts regardless of frame count.
Both seeds stuck at ~100–140, no improvement trend. Frame differencing discards too much
absolute positional information that the agent needs for localization. **Diff mode is a dead end.**

---

## exp_27 — Huber loss + eps_end=0.005 + batch=64
**Config:** `configs/exp_27_huber_eps0005.yaml`
**Status:** KILLED (~2M steps) — CONFIRMED BAD

| Metric | s0 | s2 |
|---|---|---|
| Best eval reward | 252 @ 471k | 322 @ 912k |
| At kill | 140 @ 1.93M | 32 @ 2.06M |
| q_mean at kill | 18.1 | 1.48 (collapsed) |

**Verdict:** Consistent with exp_15 — Huber loss is harmful regardless of eps_end.
s2 collapsed (q=1.48), s0 drifting high (q=18). Max reward ~322 vs exp_23's 1860.
Huber's capped gradient on large TD errors weakens the bootstrap signal that single-network
DQN needs to drive convergence. **MSE is correct for this setting. Abandon Huber entirely.**

---

## exp_29 — eps_end=0.001 + batch=64 ★ NEW BEST
**Config:** `configs/exp_29_eps0001.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | **1906** @ 3.42M | **1940** @ 2.05M |
| Final eval reward | 364 | 478 |
| Spike rate after 1M | 0% | 0% |

**Trajectory:**
| Step | s0 | s2 |
|---|---|---|
| 1M | 340 | 0 |
| 2M | 190 | 190 |
| 3M | 242 | 852 |
| 4M | 336 | 416 |
| 5M | 364 | 478 |

**Verdict:** New all-time best — **1940** (s2) and 1906 (s0). Both seeds above 1900 for the
first time — much better seed consistency than exp_23 (1860/1128). The eps trend holds:
0.05→832, 0.01→1676, 0.005→1860, 0.001→1940. s2 peaked extremely early (2.05M) which means
the near-greedy policy is locking in faster. Post-peak degradation pattern persists — both seeds
slowly recovering by 5M (364, 478) but far below the peak. Next: try slower anneal to delay
the greedy phase, and eps=0.0001 to push the trend.

---

## exp_30 — replay_buffer=2M + eps_end=0.005 + batch=64
**Config:** `configs/exp_30_bigbuf_eps0005.yaml`
**Status:** COMPLETE

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | 1860 @ 2.92M | 1128 @ 3.02M |
| Final eval reward | 266 | 312 |

**Verdict:** Identical to exp_23 in every way — same peaks, same timing, same degradation.
2M buffer has zero effect on performance. Buffer diversity is not the bottleneck for this
training setup. **Dead end — don't pursue larger buffers.**

---

## exp_31 — eps_end=0.001 + eps_anneal=2M + batch=64
**Config:** `configs/exp_31_eps0001_slowanneal.yaml`
**Status:** KILLED (~2.25M steps) — HIGH SEED VARIANCE, NOT BETTER

| Metric | s0 | s2 |
|---|---|---|
| Best eval reward | 210 @ 0.4M | 832 @ 1.1M |
| At kill | 7 @ 2.25M | 110 @ 2.25M |
| q_mean at kill | 2.5 (collapsed) | 7.7 |
| Top20avg | 155 | 475 |

**Verdict:** s0 collapsed (q=2.5 — dead). s2 peaked at 832 which is below exp_29's 1906.
Slower anneal hurts: the agent is still partially random at 2M steps, so it never commits to
the near-greedy exploitation window that drives exp_29's peaks. Fast anneal (1M) is better —
getting to eps=0.001 quickly and staying there is what produces 1940. Slow anneal is a dead end.

---

## exp_32 — eps_end=0.0001 + batch=64
**Config:** `configs/exp_32_eps00001.yaml`
**Runs:** exp_32_s0 (job 14376224), exp_32_s2 (job 14376225)
**Status:** RUNNING (resubmitted, eval_episodes=50, eval_freq=100k)

| Change | eps_end 0.001→0.0001 (vs exp_29) |
|---|---|
| Hypothesis | Push the eps trend to its limit: essentially deterministic (1 random action per 10k steps). |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_33 — update_freq=8 + eps_end=0.001 + batch=64
**Config:** `configs/exp_33_update8_eps0001.yaml`
**Runs:** exp_33_s0 (job 14375769), exp_33_s2 (job 14375770)
**Status:** DIED (preempted, no checkpoint saved — results lost)

| Change | update_freq 4→8 (vs exp_29) |
|---|---|
| Hypothesis | 1 gradient update every 8 env steps instead of 4 → Q-values drift half as fast → the self-referential bootstrap loop (q≈target, no target network) accumulates instability more slowly → longer sustained high-performance window. |

---

## exp_28 — grad_clip=10 + eps_end=0.005 + batch=64
**Config:** `configs/exp_28_gradclip_eps0005.yaml`
**Status:** KILLED (~256k steps) — DIVERGED IMMEDIATELY

| Metric | s0 | s2 |
|---|---|---|
| Best eval reward | 162 | 164 |
| q_mean at kill | 1192 | 564 |

**Verdict:** Explosive Q-value divergence from the start — same pattern as lr=0.00025 + batch=32.
Surprising given that exp_23 (same lr/batch, no clip) is perfectly stable. Mechanism: with
RMSprop's adaptive scaling, gradient clipping disrupts the v (mean-square) accumulation —
clipped gradients produce a smaller v, which inflates the effective step size and tips the
borderline-stable system into divergence. **Never combine grad_clip with RMSprop + lr=0.00025.**

---

## exp_34 — life_loss_terminal only + eps_end=0.001
**Config:** `configs/exp_34_lifeloss.yaml`
**Runs:** exp_34_s0 (job 14376087), exp_34_s2 (job 14376088)
**Status:** RUNNING

| Change | `life_loss_terminal=true` (vs exp_29) |
|---|---|
| Hypothesis | Dying gives done=True in replay buffer → Q(s_death) gets no future bootstrap. Agent genuinely learns survival has value. Previously Q-values looked past death to the respawn state. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_35 — maxpool_all_frames only + eps_end=0.001
**Config:** `configs/exp_35_maxpool.yaml`
**Runs:** exp_35_s0 (job 14376524), exp_35_s2 (job 14376525)
**Status:** RUNNING (resuming from 500k checkpoint)

| Change | `maxpool_all_frames=true` (vs exp_29) |
|---|---|
| Hypothesis | Max-pool ALL 4 skip frames instead of just the last 2. Alien bullets are single pixels on alternating frames — current code misses sprites in skip frames 0-1 entirely. Better visibility = better state representation. |
| Warning | Previously diverged at ~900k steps (q=502). Watching for recurrence. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_36 — noop_start only + eps_end=0.001 + batch=64
**Config:** `configs/exp_36_noop.yaml`
**Runs:** exp_36_s0 (job 14376518), exp_36_s2 (job 14376519)
**Status:** RUNNING

| Change | `noop_max=30` — 1–30 random no-ops at episode start (vs exp_29) |
|---|---|
| Hypothesis | Prevents agent overfitting to a fixed starting state. Mnih 2013 standard. Randomising start position forces the agent to generalise across more initial configurations. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_37 — fast eps anneal (250k steps) + eps_end=0.001 + batch=64
**Config:** `configs/exp_37_fastanneal.yaml`
**Runs:** exp_37_s0 (job 14376520), exp_37_s2 (job 14376521)
**Status:** RUNNING

| Change | `eps_anneal_steps=250_000` — paper schedule: 1M frames = 250k agent steps (vs 1M steps in exp_29) |
|---|---|
| Hypothesis | Reaches eps_end=0.001 after only 250k steps → exploits greedily for 4.75M of 5M steps. Our 1M-step anneal is 4× too slow vs the paper. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_38 — lower lr (0.0001) + eps_end=0.001 + batch=64
**Config:** `configs/exp_38_lowlr.yaml`
**Runs:** exp_38_s0 (job 14376522), exp_38_s2 (job 14376523)
**Status:** RUNNING

| Change | `lr=0.0001` (vs lr=0.00025 in exp_29) |
|---|---|
| Hypothesis | Lower lr reduces update magnitude → slower Q-value bootstrap growth → longer stable window before post-peak degradation. Trade-off: may also slow down peak acquisition. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_39 — ALL 3 paper features + eps_end=0.001 + batch=64
**Config:** `configs/exp_39_all_paper.yaml`
**Runs:** exp_39_s0 (job 14376526), exp_39_s2 (job 14376527)
**Status:** RUNNING

| Change | noop_max=30 + life_loss_terminal + eps_anneal_steps=250k (vs exp_29) |
|---|---|
| Hypothesis | Full Mnih 2013 alignment: all three paper-specified improvements combined. If each helps independently, combination should compound. The "ideal paper run." |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_40 — update_freq=2 + eps_end=0.001 + batch=64
**Config:** `configs/exp_40_updatefreq2.yaml`
**Runs:** exp_40_s0 (job 14376528), exp_40_s2 (job 14376529)
**Status:** RUNNING

| Change | `update_freq=2` — gradient update every 2 env steps instead of 4 (vs exp_29) |
|---|---|
| Hypothesis | 2× more gradient updates per episode → faster Q-value refinement. Risk: more frequent bootstrap updates may also amplify single-network instability faster. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_41 — noop + life_loss combined + eps_end=0.001 + batch=64
**Config:** `configs/exp_41_noop_lifeloss.yaml`
**Runs:** exp_41_s0 (job 14376530), exp_41_s2 (job 14376531)
**Status:** RUNNING

| Change | noop_max=30 + life_loss_terminal (vs exp_29, no fast anneal) |
|---|---|
| Hypothesis | The two most principled Mnih 2013 features: start-state randomisation + survival incentive. Tests whether their combination beats each alone (exp_34/exp_36). |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_42 — fast anneal + life_loss + eps_end=0.001 + batch=64
**Config:** `configs/exp_42_fastanneal_lifeloss.yaml`
**Runs:** exp_42_s0 (job 14376532), exp_42_s2 (job 14376533)
**Status:** RUNNING

| Change | eps_anneal_steps=250k + life_loss_terminal (vs exp_29, no noop) |
|---|---|
| Hypothesis | Fast anneal locks in a greedy policy early; life_loss teaches survival simultaneously. Combines the two features that interact with the epsilon schedule most directly. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## exp_43 — lr=0.0001 + noop + life_loss + eps_end=0.001 + batch=64
**Config:** `configs/exp_43_lowlr_noop_lifeloss.yaml`
**Runs:** exp_43_s0 (job 14376534), exp_43_s2 (job 14376535)
**Status:** RUNNING

| Change | lr=0.0001 + noop_max=30 + life_loss_terminal (vs exp_29) |
|---|---|
| Hypothesis | Lower lr may stabilise the survival signal from life_loss_terminal — each Q-update is more conservative, reducing runaway bootstrap. Tests if lr=0.0001 unlocks features that diverge with lr=0.00025. |

| Metric | s0 | s2 |
|---|---|---|
| Peak eval reward | | |
| Last eval reward | | |

---

## How to add a new experiment

1. Copy a stable config: `cp configs/exp_10_lr1e4_eps05.yaml configs/exp_11_name.yaml`
2. Change exactly one or two fields. Keep `lr: 0.0001` unless specifically testing lr.
3. Update `run_0.sh` and `run_2.sh` — change `--config` flag.
4. Add an entry here before submitting.
5. After runs finish, fill in the results table and update verdict.
