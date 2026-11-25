# Reward and teacher design (Phase 2, items 5–6): proposal

Status: approved and implemented 2026-11-22 (`dap/reward.py`,
`dap/teacher.py`, `scripts/train_teacher.py`). The 2M-step pilot from
scratch failed. The teacher is now trained by DAgger toward the servo expert
and matches it (section 2b).
It covers
the reward (`docs/phase2_plan.md` item 5) and the PPO teacher (item 6). Labels
as in `docs/setup.md`: **V** verified in the paper or code, **E** measured
here, **Ours** our choice. Measurements marked **E** use training-bank
scenarios only; no validation or evaluation seed was looked at.

## 0. What the design has to respect

- **Score is the harness's.** H1 is success rate; the tiered loss is the
  secondary measure (`docs/setup.md` 1b). A reward that ranks outcomes
  differently from the evaluation trains for the wrong thing.
- **The teacher is distilled.** Its actions must be smooth and must not
  depend on information the student cannot infer (imitation gap, Weihs et
  al. 2021). The plan fixes its input: the latency-matched palm pose and
  velocity, no endpoint, speed profile or switch time.
- **Task geometry** (**E**, 500 bank scenarios): episodes last 32–133
  policy steps (median 49, i.e. 2.5 s; 95th percentile 101). The partner
  reaches its meeting point after a median 1.0 s. At reset our palm is a
  median 0.51 m from the partner palm and **98° from antiparallel** (range
  28–179°), so most of the task is reorienting, not reaching.

## 1. Reward

Computed per 20 Hz step by a wrapper (`dap/reward.py`, new; the env's
`reward_fn(info)` hook is stateless, and two terms below need the previous
step). Everything comes from `info`, which is ground truth at the current
time. The reward is for learning only; results stay the harness scores.

    r_t = R_end(s_t+1)  +  [γ Φ(s_t+1) − Φ(s_t)]  −  w_a ‖a_t − a_t−1‖²
          outcome           shaping                   smoothness

### 1a. Outcome term: the evaluation loss, shifted

At the episode's event, `R_end = 12.5 − L`, where `L` is that trial's
`dap.scoring.dap_loss` (the same function the analysis uses, so there is one
definition). With the settled weights:

| Outcome | Loss `L` | `R_end` |
| --- | --- | --- |
| success | 0–3.95 (contact needs `pos_err` ≤ 5 cm) | 8.55 to 12.5 |
| failed contact (`rot_err` > 15°) | 7.5–10 | 2.5 to 5.0 |
| timeout | 10–15 | none: truncated, value bootstrapped (below) |
| collision | 20 | −7.5 |

- **Why:** the teacher then optimizes the ordering we score, every success
  over every failure, and within successes a centered, well-aligned
  contact. That margin matters for distillation, since a student that
  imitates with some error still lands inside the 15° tolerance. **Ours.**
- **Why 12.5:** it puts a timeout, whose loss is 10–15, at about 0. Timeouts
  are truncations, so they get no terminal reward, and the ordering
  success > failed contact > timeout > collision matches the loss.
- **Timeouts are bootstrapped, not terminal.** SB3 2.9.2 adds `γ V(s_T)` on
  truncation (`on_policy_algorithm.py`, **V**), which is partial-episode
  bootstrapping (Pardo et al. 2018, **V**). The alternative, a timeout
  penalty, would need the remaining time in the observation to keep the task
  Markov (Pardo, **V**). But the horizon depends on the partner's motion
  length, which the teacher may not see.
- **Discounting prefers earlier events.** That is harmless here, because
  success needs no timing.

### 1b. Shaping: potential-based, so it cannot change the optimal policy

    Φ(s) = −k [ θ / 45°  +  ρ / 0.25 m ],      Φ(terminal) = 0

- **θ** is `rot_err` (deg).
- **ρ** is our face center's distance to an approach point on the partner
  palm's normal: `ρ = sqrt(in_plane² + (gap − s(θ))²)`.
- **The approach point** sits `s(θ) = 5 cm · min(1, θ/15°)` in front of the
  partner face. It is 5 cm out (the servo's standoff) while misaligned, and
  slides onto the face as the palms align. This builds the servo's
  standoff-then-close approach into the shaping. A state function cannot use
  the servo's latch, so the point moves smoothly instead. **Ours.**
- **Scales:** at a median start each term is about 2, so `Φ(s0) ≈ −4` with
  `k = 1`. The total shaping is then smaller than the success reward. **Ours**;
  `k` is the main knob to check on validation seeds.
- **Invariance:** `F = γΦ(s') − Φ(s)` keeps the optimal policy (Ng, Harada &
  Russell 1999, Thm. 1, **V**).
  - With episodes that end, `Φ` must be 0 at the states where they end, or
    the last term `γ^N Φ(s_N)` "can modify the policy" (Grzes 2017, AAMAS,
    **V**). So `Φ = 0` after a contact or collision.
  - The shaped return of a terminated episode is then the unshaped return
    minus `Φ(s0)`, a constant per scenario. That makes the ranking of
    trajectories exact.
  - At a timeout `Φ` keeps its value and the critic bootstraps. I found no
    theorem for shaping under partial-episode bootstrapping (searched, **U**),
    so that case rests on the infinite-horizon view.
- **Not used:** non-potential dense terms such as `exp(−d)` or a reward for
  being close. They can make hovering near the partner pay. Catch It!, Agile
  Catching and REACT all use them (**V**); we don't need them.

### 1c. Smoothness

- **Action-rate penalty:** `w_a ‖a_t − a_{t−1}‖²` on the [−1, 1] joint-speed
  fractions, with `w_a = 0.005` (**Ours**). The proposal had 0.05.
  - Precedents use the same form: legged_gym's `action_rate` (−0.01 × dt in
    code), Isaac Lab reach (−1e-4 × dt, raised to −0.005 by a curriculum)
    and REACT (−0.03) (all **V**). Their scales are not comparable to ours,
    because they multiply by dt and their other rewards differ.
  - The penalty applies to the sampled actions, so exploration noise pays
    it too. Independent Gaussian noise with std 0.5 alone gives
    E‖Δa‖² = 2 · 7 · 0.25 = 3.5 per step.
  - At 0.05 a smoke run's episodes paid about −8 (**E**), a whole success
    reward, and the noise std began to shrink. At 0.005 the noise costs about
    0.9 per 50-step episode.
  - This weight barely smooths the policy's mean actions. If the trained
    teacher's deterministic actions turn out jerky (mean ‖Δa‖ is logged on
    validation seeds), the options are a penalty on the policy mean instead
    of the sampled action (CAPS's temporal loss, Mysore et al. 2021, **V**),
    or raising the weight as the noise shrinks (Isaac Lab's reach curriculum,
    **V**).
- **Why smoothness matters:** Catch It! needed a low-pass filter on the real
  robot (**V**), and gSDE's authors report jerky step-based exploration
  (**V**).
- **No source shows that smoothness helps distillation** (searched, **U**).
  It is a plausible benefit, not a cited one.
- **Not used:**
  - A control-magnitude penalty (Catch It!'s `‖a‖`), because a 0.9 m/s
    partner needs speed.
  - Joint-limit penalties, because the harness clips commands at the limits.
  - A contact-speed term, because the loss leaves it unweighted.

## 2. Teacher

| Choice | Value | Source |
| --- | --- | --- |
| Algorithm | PPO, Stable-Baselines3 2.9.2 | plan; SB3 handles truncation (**V**) |
| Input | `privileged` obs (43-D), `privileged_delay=True` | plan; Weihs 2021 |
| Extra features | `partner_p − palm_p`, and the rotation from our palm to the antiparallel target as its 9 matrix entries (continuous, where an axis-angle vector flips at 180°), computed by an observation wrapper before normalization | **Ours**; Visual Dexterity's teacher also gets the rotation difference (V via ar5iv summary) |
| Networks | separate actor and critic MLPs, 2 × 256, tanh | Andrychowicz et al. 2021: tanh, two layers, separate wide value net (**V**); Catch It! [256, 128] (**V**); zoo Reacher [256, 256] (**V**) |
| Init | last policy layer × 0.01 (SB3 default); initial action std 0.5 (`log_std_init = −0.69`) | Andrychowicz: 0.01 scaling, std 0.5 "performing best" (**V**) |
| Normalization | `VecNormalize` on observations (clip 10) and rewards; stats saved with the teacher and reused for DAgger labels | Andrychowicz "Always use observation normalization" (**V**); Engstrom 2020 reward scaling (**V**); SB3 docs on saving stats (**V**) |
| γ, λ | 0.99, 0.95 | effective horizon 100 steps ≈ the 95th-percentile episode (**E**); SB3, legged_gym and Isaac Lab defaults (**V**); Andrychowicz suggests λ = 0.9 (**V**) |
| PPO | lr 3e-4, clip 0.2, 256 steps per env per iteration, minibatch 256, 5 epochs, entropy 0, grad clip 0.5 | Andrychowicz "0.0003 is a safe default" (**V**); Rudin et al. 2021: PPO "struggles when we provide fewer than 25 consecutive steps" (**V**) |
| Exploration | Gaussian (gSDE off) | smoothness comes from the reward, and DAgger labels use the deterministic mean |
| Warm start | none by default | fallback below |

**Throughput (E, today's machine).** The plan assumed 620 steps/s per core,
about 6,200 in total. It isn't so:

- **Measured:** one env does 410 steps/s with random actions. Ten envs in
  SB3's `SubprocVecEnv` do 530–620 in total, and with PPO updates 560.
- **Cause:** a bank reset costs 49 ms, 94% of it rebuilding the Drake
  diagram, against about 1 ms per step. SB3 steps all environments in
  lockstep, so one reset stalls the other nine.
- **Without lockstep:** eight independent single-env processes did 220
  steps/s each, 1,770 in total. Chrome was using more than 3 cores at the
  time.

Options:

- **(a) Per-worker rollout collection (chosen, `dap.teacher.ParallelPPO`).**
  Each worker runs its own copy of the actor for a whole rollout, and the
  learner does SB3's PPO update on the merged batch. It needs no env change.
  - **Measured:** about 2,000 steps/s steady state with 8 workers,
    including PPO updates (16,384 steps in 8 s, **E**). That is 3.6× stock
    SB3 on the same machine.
  - It also helps the DAgger student (8 ms steps, 130 ms resets).
- **(b) Reset without rebuilding the scene.** Move the partner base and
  pedestals in the context rather than building a new diagram. This would
  remove most of the remaining cost, but it changes `dap/scene.py`. It would
  need the regression lock to show identical behavior, and Drake support for
  every part of it is unverified.
- **(c) Accept 600 steps/s.** Then 10M steps take about 4.6 h.

**Budget.** The steps needed are unknown until the teacher trains.

- **Precedents** (**V**): SB3 zoo reacher runs 1M steps; Isaac Lab Franka
  reach runs about 98M (1,000 iterations × 4,096 envs × 24 steps); legged_gym
  about 147M. Ours is harder than a static reach.
- **Plan:**
  - A **2M-step pilot**: about 20 min with (a), plus about 20 s per
    validation checkpoint.
  - Then up to **20M steps** if success is still rising: about 3 h with (a).
    This replaces the plan's estimate of 50M steps in 2.2 h.

**Checkpoints and the bar (validation seeds only).**

- **Selection:** every 0.5M steps, score the deterministic teacher with
  `run_policy_trial(..., allow_privileged=True)` on validation seeds 0–99.
  Pick by success rate, then mean loss.
- **Bar (item 6):** check the teacher against `servo_gt_cam` on disjoint
  validation seeds 100–299, paired. Checking on the selection seeds would
  bias the check upward, since the best of many checkpoints looks better on
  the seeds it was picked on. Report success rates with Wilson intervals and
  McNemar, labelled as a check.
- **Diagnostics:**
  - Mean ‖Δa‖ (smoothness).
  - How much the teacher's actions change on frames where the partner palm is
    hidden from the head camera. Those are the states where the student
    cannot follow.

**Fallbacks, in order, if the pilot does not learn:**

1. Raise `k` (more shaping).
2. Warm start: behavior-clone `servo_gt_cam`'s actions on training scenarios,
   then PPO.
   - Precedents: DAPG (Rajeswaran et al., RSS 2018), kickstarting (Schmitt
     et al. 2018) and jump-start RL (Uchendu et al. 2022), all arXiv-checked.
     Christen et al. 2023 pretrain from a privileged planner (**V**).
   - The risk is a teacher biased toward the servo's style.
3. An asymmetric critic that also sees the current (not delayed) state and
   the time to the horizon (Pinto et al. 2018).

## 2b. Results so far (validation seeds 0–99, development only)

These are selection-set numbers for choosing what to run next, not reported
results.

**The 2M-step pilot from scratch** (`results/teacher/pilot_2m/`, **E**):
0/100 at every checkpoint.

- It learned to avoid collisions (20% → 0% of training episodes) and then
  timed out about 99% of the time. Validation loss went 13.2 → 12.6.
- Median closest distance plateaued at about 20 cm, and median rotation
  error at the closest point fell from 128° to 46°.
- Exploration std collapsed from 0.5 to 0.15.
- Reading: a cautious local optimum. Hovering near the partner beats risking
  a collision, and random exploration rarely finds an antiparallel touch.
  This is the exploration failure that fallback 2 addresses.

**The expert** (`dap.teacher.ServoExpert`, **E**): the diff-IK servo run at
20 Hz on the teacher's own privileged observation, aiming LOOKAHEAD ahead of
the latency-matched partner palm.

- Validation: 94/100 (97 contacts, 3 of them over 15°; 3 collisions) with
  the servo's latched closing rule. With the memoryless rule it now uses
  (below): 96/100 (97 contacts, 1 over 15°; 3 collisions).
- Training scenarios: 96.2% (latched) and 96.5% (memoryless) of 2,000
  episodes.
- It beats `servo_gt_cam` (11/12 vs 8/12 on seeds 0–11) because it uses the
  partner's true velocity, not one estimated from camera frames.
- It is the fair reference for what the teacher's information is worth;
  `servo_gt_cam` remains the bar for the student.
- It replaces fallback 2's `servo_gt_cam`, which needs camera frames the
  teacher's observation does not have. The expert sees exactly what the
  teacher sees, and with the memoryless rule it keeps no history.

**Warm start** (`dap.teacher.collect_expert` / `warm_start`;
`train_teacher.py --warm-start 2000`). Precedents: DAPG, kickstarting and
jump-start RL (section 2).

- *Actor:* the Gaussian mean regressed on the expert's actions, 30 epochs,
  Adam at lr 1e-3 (Ours).
- *Critic:* regressed on the expert's discounted returns, in the scaled
  reward units PPO uses.
- *Normalization:* observation and return statistics fitted on the expert
  data.
- *Exploration:* std reset to 0.2 (Ours).
- *Fit* (**E**): actor MSE 0.014, critic R² 0.87 on 44,726 steps.

**A 300k-step check** (`results/teacher/warm_check_300k/`, **E**):

| Steps | Success | Contact | Collision | Timeout | Loss | Mean ‖Δa‖ |
|---|---|---|---|---|---|---|
| 0 (BC only) | 46 | 50 | 44 | 6 | 10.97 | 0.35 |
| 100k | 43 | 57 | 34 | 9 | 10.09 | 0.41 |
| 200k | 68 | 83 | 12 | 5 | 5.84 | 0.47 |
| 300k | 79 | 90 | 8 | 2 | 4.30 | 0.60 |
| 300k + 1 update | 81 | | | | 4.43 | |

In this run, checkpoints were taken before each rollout's PPO update. The
final policy, one update later, was saved under the same name
(`ckpt_301056`) and appended to `val.jsonl` under the same step count. Later
runs checkpoint at the start of a rollout, so `ckpt_<n>` is the policy after
the update on all n steps.

What the check shows:

- BC alone loses half the expert's successes, mostly to collisions. This is
  the usual compounding error of cloning (Ross et al. 2011), plus the latch.
  PPO then recovers most of that in 300k steps and was still rising.
- **PPO steps are large.** Approx-KL is about 0.12–0.18 per update and the
  clip fraction about 0.55, against typical values of about 0.01–0.02 and
  0.1–0.2 (Huang et al. 2022). With std 0.2, a small shift in the mean is a
  large KL, since each joint contributes Δμ²/2σ². It still learns, but
  lowering the learning rate or setting SB3's `target_kl` is the first knob
  if it becomes unstable.
- **Smoothness is getting worse** (mean ‖Δa‖ 0.35 → 0.60). Watch this for
  the student.

**The 2M-step warm-started run** (`results/teacher/warm_2m/`, same
settings, **E**):

| Steps | Success | Contact | of which > 15° | Collision | Timeout | Mean ‖Δa‖ |
|---|---|---|---|---|---|---|
| 0 (BC only) | 46 | 50 | 4 | 44 | 6 | 0.35 |
| 250k | 72 | 87 | 15 | 7 | 6 | 0.51 |
| 500k | 80 | 91 | 11 | 7 | 2 | 0.75 |
| 750k | 77 | 88 | 11 | 10 | 2 | 1.04 |
| 1.0M | 78 | 89 | 11 | 8 | 3 | 1.17 |
| 1.25M | 66 | 79 | 13 | 10 | 11 | 1.48 |
| 1.5M | 75 | 85 | 10 | 12 | 3 | 1.44 |
| 1.75M | 75 | 89 | 14 | 9 | 2 | 1.43 |
| 2.0M | 79 | 90 | 11 | 8 | 2 | 1.44 |

The expert's own mean ‖Δa‖ is 0.49 (400 training episodes).

- **Success plateaus at about 75–80 from 500k steps on**, below the expert's
  94. PPO fixed cloning's collisions but has not caught up with the expert
  it started from. The remaining failures are mostly contacts over 15° (about
  11) and collisions (about 8).
- **The deterministic policy chatters.** Mean ‖Δa‖ is about 3× the expert's
  by 1M steps.
- **Updates grow as exploration shrinks.** Std goes 0.20 → 0.12, approx-KL
  0.12 → 0.35, and clip fraction 0.53 → 0.67.
- **The plumbing is not the cause.** Worker and learner log-probs on the
  same rollout agree to 1e-6 (**E**), so the KL comes from the updates
  themselves.
- Training rollouts with exploration noise contact about 90% of the time.
  The noise averages over a jittery mean.
- **Next lever:** SB3's `target_kl` (`train_teacher.py --target-kl`), which
  stops an update's epochs once approx-KL passes 1.5× the target (Ours).

**The same run with `--target-kl 0.02`** (`results/teacher/warm_2m_kl02/`,
**E**): collapsed, and stopped at about 265k steps.

- The cap worked as intended. Approx-KL was about 0.015, clip fraction
  about 0.2, and about one epoch per update.
- Contacts in training rollouts fell from 14% to 0 within about 60k steps,
  and timeouts rose to about 89%.
- Validation at 250k: 0/100 (92 timeouts, 8 collisions), with a median
  closest distance of 25 cm. This is the pilot's hovering optimum again.
- **Reading:**
  - Under std-0.2 exploration noise, the warm-started policy collides 60–80%
    of the time near the partner.
  - Mean outcome reward in this run's training episodes (**E**): contact
    +6.6, collision −7.5, timeout 0. So for the noisy policy, approaching is
    worth less than hovering, and small steps follow that gradient to
    hovering.
  - SB3's early stop also skips the critic's update (one shared loss), and
    the warm-started critic starts out mismatched (explained variance −0.25).
  - Without the cap, large updates sharpened the mean faster than this
    drift; that is not a fix.
- **Conclusion:** PPO from the expert needs an anchor to the expert, not
  just smaller steps.

**Kickstarting** (`ParallelPPO.train`, `train_teacher.py --kick-coef`;
Schmitt et al. 2018).

- *Labels:* each worker runs the ServoExpert alongside the policy and records
  its action at every state the policy visits.
- *Loss:* PPO's loss gains −λₖ · mean log π(a_expert | s) on a random
  minibatch of the rollout. For a deterministic expert this is kickstarting's
  cross-entropy.
- *Schedule:* λₖ falls linearly from `kick_coef` to 0 over `kick_steps`.
- *Cost:* about 14% of throughput (2,556 → 2,196 steps/s, **E**).

**Check, 300k steps, kick 1.0 annealed over 1M** (`results/teacher/kick_check_300k/`,
**E**): collapsed to hovering, 0/100 at 100k, 200k and 300k (about 92
timeouts, with mean ‖Δa‖ 0.02–0.04: the deterministic policy barely moves).

- **The labels are right, and the gap is real.** Driving with the expert,
  the cloned policy matches it to MSE 0.02. Driving with the cloned policy,
  disagreement grows to 0.05 by step 2, 0.14 by step 10 and 0.31 by step 20
  (20 training episodes). This is compounding error (Ross et al. 2011).
- **The cross-entropy's std gradient reads that disagreement as "explore
  more".** Std rose 0.20 → 0.55, so the noisy policy collided more, which
  strengthens PPO's push toward hovering.
- **It swamped PPO without fitting the expert.** At σ = 0.2 its gradient on
  the mean is large, so one update moved the policy a long way (approx-KL up
  to about 2.3) and about 87% of PPO's samples were clipped. Yet the mean
  never closed in: kick MSE stayed about 0.31–0.33, because each rollout
  lands in new states and only that rollout's labels are used.
- **Reading:** what the evidence points to is DAgger (Ross et al. 2011):
  relabel the policy's own states with the expert and fit an aggregated
  dataset by imitation alone, before any PPO. A milder fix is to keep the
  kick term off the std (mean-only), which removes the inflation but not the
  chasing.

**DAgger toward the expert** (`dap.teacher.dagger`, `train_teacher.py
--dagger-rounds`; Ross et al. 2011).

- *Round 0* is the warm start on 2,000 expert episodes.
- *Each round:*
  - Roll out the current policy (std 0.2) for 40k steps in the workers, with
    the expert labelling every state.
  - Add those states to one aggregated dataset.
  - Refit the actor's mean on all of it: MSE, 10 epochs, Adam at lr 1e-3,
    continuing from its weights.
- *Normalization:* observation statistics stay those of the expert data.
- *No PPO is involved.*

**Check: 5 rounds** (`results/teacher/dagger_check/`, **E**). The gap is
the policy's MSE to the expert on its own states, before refitting.

| Round | Gap | Fit | Success | Contact (> 15°) | Collision | Timeout | Loss | Mean ‖Δa‖ |
|---|---|---|---|---|---|---|---|---|
| 0 (BC) | – | 0.014 | 46 | 50 (4) | 44 | 6 | 10.97 | 0.35 |
| 1 | 0.23 | 0.028 | 77 | 78 (1) | 7 | 15 | 4.22 | 0.32 |
| 2 | 0.08 | 0.030 | 54 | 54 (0) | 8 | 38 | 6.21 | 0.26 |
| 3 | 0.08 | 0.030 | 65 | 66 (1) | 6 | 28 | 4.76 | 0.27 |
| 4 | 0.06 | 0.032 | 82 | 82 (0) | 5 | 13 | 3.31 | 0.32 |
| 5 | 0.05 | 0.031 | 84 | 84 (0) | 7 | 9 | 3.29 | 0.33 |

- **The best teacher so far,** and the cheapest: 84/100 with loss 3.29,
  against 80/100 and loss 4.13 for 500k PPO steps. Its contacts are
  accurate (none over 15°), and its actions are as smooth as the expert's
  (0.33 against 0.49).
- **What is left is hovering:** timeouts that stop a median 1.4–2.5 cm short
  of the partner, peaking at round 2.
- **First guess: the expert's closing latch.** The servo holds a 5 cm
  standoff until it is within 2 cm of it and aligned, then closes and never
  switches back. When the policy drives a different path, the shadow expert
  may never latch, which would give conflicting labels near contact. The
  rerun below shows this was at most part of the cause.
- **A memoryless closing rule does as well**
  (`DiffIKServo(latch=False)`, now the ServoExpert's rule, **E**).
  - The rule: close whenever the palm is within 2 cm of the partner's normal
    line in-plane, within 10° in rotation, and no further out than the
    standoff plus 2 cm.
  - It does not chatter, because advancing along the normal does not break
    the check.
  - As the expert on validation seeds 0–99 it scores 96/100 (97 contacts, 1
    over 15°, 3 collisions), against 94/100 for the latched rule.
  - Its labels are a function of the state alone, which is what DAgger
    needs. A test checks that a fresh expert labels any state of an
    episode exactly as the one that drove it.

**DAgger rerun with the memoryless expert** (`results/teacher/dagger_memoryless/`,
same settings, **E**):

| Round | Gap | Fit | Success | Contact (> 15°) | Collision | Timeout | Loss | Mean ‖Δa‖ |
|---|---|---|---|---|---|---|---|---|
| 0 (BC) | – | 0.016 | 37 | 38 (1) | 53 | 9 | 12.77 | 0.34 |
| 1 | 0.23 | 0.029 | 69 | 71 (2) | 8 | 21 | 4.94 | 0.30 |
| 2 | 0.08 | 0.028 | 68 | 69 (1) | 2 | 29 | 4.40 | 0.29 |
| 3 | 0.07 | 0.026 | 83 | 84 (1) | 4 | 12 | 2.91 | 0.34 |
| 4 | 0.045 | 0.024 | 86 | 87 (1) | 2 | 11 | 2.34 | 0.35 |
| 5 | 0.036 | 0.022 | 87 | 89 (2) | 5 | 6 | 2.72 | 0.37 |

- **The best teacher so far:** 86–87/100 with loss 2.3–2.7, smooth actions,
  and still improving. Gap and fit fall every round, where with the latched
  expert the fit rose.
- **Hovering remains, and it is not label conflict.** On 60 training-bank
  episodes with the round-2 policy (26 timeouts), the last 10 steps of each
  timeout show:
  - The palm is aligned: in-plane offset about 1 cm, rotation about 2°.
  - It is 1–3 cm from the partner face.
  - The expert consistently labels "close", with action norm about 0.8–1.1,
    yet the policy outputs about 0.1.
- **Copying the previous action explains a little of it.** The privileged
  vector holds both `prev_action` and `qdot` (= previous action × v_max in
  this kinematic sim), and the expert's actions are smooth. Swapping in the
  expert's own previous action at those states raises the policy's action
  norm from 0.14 to 0.25, against the expert's 0.9.
- **Mostly the network underfits the last few cm.** The expert's closing
  command is sharp there (gain 10 on a 1–4 cm error), and those states are
  a small share of an MSE fit over everything.

**15 rounds** (`results/teacher/dagger_15/`, same settings and seed, so
rounds 1–5 repeat the run above exactly, **E**):

| Round | Gap | Fit | Success | Contact (> 15°) | Collision | Timeout | Loss | Mean ‖Δa‖ |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.036 | 0.022 | 87 | 89 (2) | 5 | 6 | 2.72 | 0.37 |
| 6 | 0.031 | 0.019 | 87 | 89 (2) | 6 | 5 | 2.89 | 0.39 |
| 7 | 0.026 | 0.018 | 89 | 92 (3) | 4 | 4 | 2.32 | 0.39 |
| 8 | 0.023 | 0.016 | 86 | 88 (2) | 4 | 8 | 2.65 | 0.40 |
| 9 | 0.022 | 0.015 | 92 | 93 (1) | 4 | 3 | 2.09 | 0.41 |
| 10 | 0.019 | 0.015 | 89 | 91 (2) | 4 | 5 | 2.44 | 0.41 |
| 11 | 0.019 | 0.014 | 95 | 96 (1) | 3 | 1 | 1.61 | 0.45 |
| 12 | 0.016 | 0.013 | 93 | 94 (1) | 4 | 2 | 1.90 | 0.43 |
| 13 | 0.017 | 0.014 | **97** | 97 (0) | 3 | 0 | 1.52 | 0.46 |
| 14 | 0.015 | 0.012 | 96 | 97 (1) | 1 | 2 | **1.40** | 0.45 |
| 15 | 0.014 | 0.012 | 95 | 96 (1) | 3 | 1 | 1.59 | 0.44 |

- **The teacher matches the expert.** From round 11 on it scores 93–97/100
  against the expert's 96/100 on the same seeds.
- **Hovering is gone:** 0–2 timeouts from round 11. More rounds were what
  the last few cm needed.
- **Smooth:** mean ‖Δa‖ 0.44–0.46, against the expert's 0.49.
- **Converging:** the gap on its own states falls to 0.014 and the fit
  error to 0.012, while success plateaus at about 95.
- **Selected:** `ckpt_dagger13` (97/100, loss 1.52), by success then mean
  loss on these selection seeds. Choosing the best of 16 checkpoints on
  these seeds is optimistic, so the bar check (section 2) uses validation
  seeds 100–299.

**Bar check** (`scripts/check_teacher.py`, config `teacher_bar`: validation
seeds 100–299, paired; `results/teacher_bar/dagger_15_ckpt_dagger13/`,
**E**). A check, not a result: the teacher and the expert see ground truth.

| | Success [Wilson 95%] | Collision | Timeout | Mean loss [95%] | Median rot err at contact |
|---|---|---|---|---|---|
| Teacher (`ckpt_dagger13`) | 93.0% [88.6, 95.8] | 5.0% | 1.5% | 2.11 [1.55, 2.74] | 1.6° |
| Expert (ServoExpert) | 94.0% [89.8, 96.5] | 6.0% | 0 | 1.89 [1.30, 2.57] | 0.9° |
| `servo_gt_cam` (the bar) | 83.5% [77.7, 88.0] | 16.0% | 0 | 4.86 [3.94, 5.81] | 0.6° |

- **The teacher clears the bar.** Against `servo_gt_cam` it succeeds where
  the bar fails on 24 seeds, and the reverse on 5: exact McNemar p = 0.0005
  (Holm 0.003). Its loss is 2.75 lower [1.85, 3.69] (paired bootstrap).
- **The teacher matches the expert.** Discordant pairs are 2 vs 4 (p =
  0.69), and the loss difference is 0.22 [−0.18, 0.62]. Its contacts are
  slightly less exact: rotation error 0.75° larger and position error 3 mm
  larger, both significant and both far inside the 15° tolerance.
- **Selection bias was small:** 97/100 on the selection seeds, 93% here.
- **Fastest partners:** the teacher's failures are mostly partners peaking
  at 0.75–0.90 m/s (29/38 succeed), exactly where the expert fails too
  (29/38). There it is limited by what the expert can do, not by imitation.

## 3. Distillation risks the teacher should anticipate (not in the default)

- **Occlusion near contact.** Our own hand hides the partner palm from the
  head camera for a few hundred ms, while the teacher still sees it. For
  the servo this costs little: 8/12 with the true pose on every frame vs.
  7/12 on visible frames only (**E**, validation, `docs/setup.md` 1b).
  Options, if the student's gap concentrates there:
  - **A visibility-gated teacher input:** hold the partner state at the last
    visible frame. No paper does exactly this (searched, **U**). In the
    nearest evidence, a badminton policy trained with a fitted perception
    model, including field-of-view dropout, learned to keep the target in
    view, and one trained on ground truth did not (Ma et al., Science
    Robotics 2025, **V**).
  - **Student-informed teacher training (SITT):** penalize the teacher by its
    KL to a proxy student, `r' = r − 0.05 · KL` (Messikommer et al., ICLR
    2025, **V**). In a drawer task where the arm could hide the handle, the
    teacher learned to avoid self-occlusion. Student success was 0.34 with
    DAgger and 0.88 with SITT.
- **Expect a real gap.** REACT, the closest task we found, has a humanoid
  high-fiving a person. Its PPO teacher sees the partner's pose and its
  DAgger depth student sees 192×108 frames. Contact recall at 15 cm was 69.5
  for the teacher and 23.9 for the student (Shibata et al., CoRL 2026,
  **V**). Its reward is direction and distance terms, a contact term and an
  action-rate penalty of −0.03.

## 4. Decisions for the team

1. ~~Reward form~~ and 2. ~~throughput~~: approved 2026-11-22 and built,
   with the action-rate weight lowered to 0.005 (section 1c).
3. ~~The 2M-step pilot~~: ran, 0/100 (section 2b). Warm start from the
   servo expert: approved and built; 2M steps plateau at 75–80/100 with a
   chattering policy; capping the update size collapses it to hovering, and
   so does kickstarting as built. DAgger toward the expert reaches 84/100,
   smooth, in 5 rounds and no PPO, and 87/100 with the memoryless expert.
   After 15 rounds it matches the expert: 93–97/100 from round 11 on;
   `ckpt_dagger13` selected. Bar check on validation seeds 100–299:
   93.0% against `servo_gt_cam`'s 83.5% (McNemar p = 0.0005) and the
   expert's 94.0% (section 2b). The teacher is ready for item 7.
4. Occlusion: keep the plan's plain latency-matched teacher for now
   (recommended), and revisit with the student's measured gap.

## References (new here; others in `docs/setup.md`)

- Ng, Harada, Russell. Policy invariance under reward transformations: theory and application to reward shaping. ICML 1999.
- Grzes. Reward shaping in episodic reinforcement learning. AAMAS 2017, pp. 565–573.
- Pardo, Tavakoli, Levdik, Kormushev. Time limits in reinforcement learning. ICML 2018, arXiv:1712.00378.
- Andrychowicz et al. What matters in on-policy reinforcement learning? A large-scale empirical study. ICLR 2021, arXiv:2006.05990.
- Engstrom et al. Implementation matters in deep policy gradients: a case study on PPO and TRPO. ICLR 2020, arXiv:2005.12729.
- Huang et al. The 37 implementation details of proximal policy optimization. ICLR Blog Track 2022.
- Raffin, Kober, Stulp. Smooth exploration for robotic reinforcement learning (gSDE). CoRL 2021, arXiv:2005.05719.
- Rudin, Hoeller, Reist, Hutter. Learning to walk in minutes using massively parallel deep RL. CoRL 2021, arXiv:2109.11978; code legged_gym @ 8fa29ac.
- Isaac Lab reach task and Franka PPO config, isaac-sim/IsaacLab @ 37ddf62 (v2.3.2).
- Zhang, Liang, Chen, Ze, Xu. Catch It! Learning to catch in flight with mobile dexterous hands. arXiv:2409.10319 (ICRA 2025 per the authors' repo); code @ b6af28c.
- Abeyruwan et al. Agile catching with whole-body MPC and blackbox policy learning. L4DC 2023, arXiv:2306.08205.
- Rajeswaran et al. Learning complex dexterous manipulation with deep RL and demonstrations (DAPG). RSS 2018, arXiv:1709.10087.
- Schmitt et al. Kickstarting deep reinforcement learning. arXiv:1803.03835, 2018.
- Uchendu et al. Jump-start reinforcement learning. arXiv:2204.02372.
- Ross, Gordon, Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning (DAgger). AISTATS 2011, arXiv:1011.0686.
- Messikommer, Xing, Aljalbout, Scaramuzza. Student-informed teacher training. ICLR 2025, arXiv:2412.09149.
- Ma, Cramariuc, Farshidian, Hutter. Learning coordinated badminton skills for legged manipulators. Science Robotics 10, eadu3922, 2025; arXiv:2505.22974.
- Shibata et al. VLM-guided contact retargeting and generative partner modeling for humanoid-human interaction (REACT). CoRL 2026, arXiv:2610.28523.
