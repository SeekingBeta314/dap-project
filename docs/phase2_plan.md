# Phase 2 plan: perceiving the partner from depth

Status: written 2026-11-22, at the end of Phase 1. This is the handoff for
whoever picks up Phase 2. It says what Phase 1 established, what Phase 2 is
for, what to build in what order, how it will be evaluated, and what the
team still has to decide.

**The focus is perception.** The trajectory planner (`dap/planner.py`,
`dap/meeting.py`) and the experiments built around it (H1, replanning vs.
planning once; H2, success vs. partner speed) are no longer part of the
project. Phase 2 asks whether a policy that learns its own perception from
the head camera's depth frames meets the partner better than a classical
perception pipeline on the same frames.

## 1. Where Phase 1 left us

The problem setup is fixed and documented, with a source for every
parameter (`docs/setup.md`):
- **Scene and randomization:** two iiwa arms, randomized layouts, start
  configurations and meeting points. An IK check confirms that a dap exists.
- **Partner:** a human-like min-jerk reach at a peak speed drawn per
  scenario (0.15–0.9 m/s), executed by a real iiwa within its limits.
- **Camera:** a D435-style depth camera at 30 fps with one frame of latency
  and modelled noise.
- **Scoring:** from ground truth only. Success means the palms touch
  antiparallel within 15° (a placeholder).
- **Seeds:** paired across methods and split into train, validation and
  evaluation streams (`docs/setup.md` section 9).
- **RL environment:** `dap/PalmMeet-v0` with a `pixels` mode (the student),
  a latency-matched `privileged` mode (the teacher), a training scenario bank
  and a scorer for learned policies.

The baseline ladder on validation seeds 0–11 (`first_look`,
`first_look_wide`; a check, not a result):

| Controller | Partner from | `nominal` | `wide` |
| --- | --- | --- | --- |
| `blind` | nothing (a fixed pose fit on training scenarios) | 0/12 | 0/12 |
| `servo_depth` | **classical perception on the depth frames** (`dap/perception.py`) | 0/12 | 3/12 |
| servo, true pose at camera timing (scratch diagnostic) | ground truth, 30 Hz, one frame late | 8/12 | 9/12 |
| `servo_gt` | ground truth, current, 100 Hz | 10/12 | 11/12 |
| `servo` | stand-in pose sensor (noise only, no latency or occlusion) | 11/12 | 11/12 |

The same servo controller succeeds most of the time with a good pose
estimate, and rarely with the pose it perceives from depth. The camera's
timing costs a few successes; the perception step costs most of the rest
(`docs/setup.md` section 1b). On training frames the classical pipeline
gives a fix in 65% of frames, gets the finger direction wrong in 15% of
fixes, and sometimes locks onto the partner's arm
(`docs/perception_pipeline.md`). That gap is what Phase 2 is about.

## 2. Questions

Fixed now, before any evaluation run, so the evaluation tests them rather
than whatever looks good afterwards.

- **H1 (primary).** A depth student distilled from a privileged teacher
  daps more often than `servo_depth` on paired evaluation seeds, on the
  `nominal` preset. Test: exact McNemar on per-seed success; the paired
  bootstrap of the loss as the secondary measure.
- **Q2: what does perceiving from depth cost a learned policy?** Report the
  teacher − student difference in success (paired, with a CI) and the same
  difference for the classical pair (servo at camera timing −
  `servo_depth`). This is an estimate, not a directional hypothesis.
- **Q3: does it generalize?** H1's comparison repeated on `wide` (wider
  layouts and mid-course target switches, not seen in training if we train
  on `nominal` only, section 6), and both presets broken down by realized
  partner speed. Following Abeyruwan et al. (2023), who found a learned
  catcher matched MPC in distribution but collapsed under distribution
  shift, out-of-distribution results are reported separately.
- Holm's correction across the confirmatory tests (H1 on `nominal`; H1 on
  `wide` if Q3 is run as a test). Everything else is exploratory and
  labelled so.

**Not claimed:** that RL improves timing over a rendezvous planner. Every
classical catcher picks an intercept time explicitly (Kim et al. 2014;
Bäuml et al. 2010), and our servo does not. A timing claim would need an
intercept-aware comparator, which this project no longer builds
(`reports/Dap task research landscape.md`). Timing error is still logged
and reported as a score.

**Novelty, stated carefully.** As far as our searches found, learned
interception systems route perception through a separate tracker, and
learned handover policies use segmented point clouds from a wrist camera.
Raw head-camera depth to joint velocities for an oriented contact with a
moving partner, compared on paired seeds against a classical pipeline on
the same frames, has no verified precedent. The searches were not
exhaustive, so write "to our knowledge".

## 3. Method

The standard route to depth policies (Chen et al. 2019; Agarwal et al.
2022; Zhuang et al. 2023): train a teacher with RL on compact privileged
state, then distill it into a depth student with on-policy DAgger (Ross et
al. 2011). Direct RL from depth failed or was judged intractable in both
locomotion papers, so it is not the main method.

- **Teacher:** PPO on the `privileged` observation, which is our arm state
  plus the partner palm's true pose and velocity as of the student's newest
  frame. The latency matching is deliberate. A teacher acting on
  information the student cannot infer is imitated only on average (the
  imitation gap; Weihs et al. 2021), and the privileged vector holds no
  endpoint, speed profile or switch time.
- **Student:** the `pixels` observation (arm state plus the last 3 depth
  frames at 212×120). Defaults from the literature, to settle on validation
  seeds:
  - a small CNN, with a GRU on top if the frame stack is not enough
    (every successful student we reviewed had memory);
  - random-shift augmentation, the same shift across the stacked frames
    (DrQ-v2, Yarats et al. 2021);
  - a behavior-cloning warm start on teacher rollouts before DAgger (Chen
    et al. 2019).
- **Action smoothness:** an action-rate penalty in the teacher's reward.
  Catch It!'s real-robot palm-touch rate depended on low-pass filtering its
  policy's actions (Zhang et al. 2024), and a smooth teacher is easier to
  distill.
- **Budget, an estimate:** at about 95 `pixels` steps/s per core and 620
  `privileged` steps/s per core (`docs/setup.md` section 8, **E**), 6M
  DAgger steps on 10 cores take about 1.75 h, and 50M teacher steps about
  2.2 h. Agarwal et al. used 6M student samples; the teacher budget is
  unknown until it trains.

## 4. Work items, in order

Each item says what is done when it is done. Owners are for the team to
fill in.

| # | Item | Done when | Owner |
| --- | --- | --- | --- |
| 0 | **Settle scoring.** The success tolerance (15°), whether success needs timing, and the loss weights (`docs/setup.md` section 11). The reward and H1 both depend on them. | Values written into `dap/scoring.py` and `docs/setup.md`, decided without looking at evaluation seeds | |
| 1 | **Retire the planner.** Remove `replan_vs_once`, `speed_limits` and the `planner` default from `dap/experiment.py` and `scripts/run_experiment.py`. Keep the controller contract in `dap/planner.py` (the baselines implement it); delete or clearly park the `ReplanningController` and `dap/meeting.py` stubs. | `make test` passes; the docs no longer mention H1/H2 | |
| 2 | **Promote the camera-timing reference.** Turn the scratch diagnostic (the servo fed the true pose at the camera's capture times) into a named reference, `servo_gt_cam`. It is the classical counterpart of the teacher for Q2. | Listed in `dap/baselines.py` and the docs, covered by the regression lock | |
| 3 | **Install training dependencies.** torch and stable-baselines3 (`docs/setup.md` section 8 gives the version constraints). | `requirements.txt` updated; `make setup` works on a clean clone | |
| 4 | **Build the training bank.** `scripts/build_bank.py --n 20000 --workers 8` on `nominal` (about 1.5 h: 40 scenarios took 10 s with 8 workers). | `banks/train_nominal_20000.pkl`, failed seeds listed | |
| 5 | **Reward.** A `reward_fn(info)` (`docs/setup.md` section 8 lists the fields): success bonus, collision penalty, shaping on face distance and rotation error, action-rate penalty. Our choice, marked **Ours**. | The teacher learns on a small bank; reward documented | |
| 6 | **Teacher.** PPO on `privileged`, trained on the bank, checkpoints chosen on validation seeds. | Teacher success on validation seeds at least that of `servo_gt_cam` (otherwise the student has nothing to learn from) | |
| 7 | **Student.** DAgger on `pixels` with teacher labels from `info["privileged"]`, BC warm start, augmentation. Architecture and checkpoint chosen on validation seeds. | Student and teacher frozen, with the choices written down | |
| 8 | **Evaluation configs.** One config per preset on evaluation seeds from 1000, with every method on the same seeds: student, `servo_depth`, `servo_gt_cam`, `servo_gt`, `servo`, teacher (a reference), `blind`, `oracle`. Seed counts from a power calculation on validation pilot rates (Connor 1987), fixed before the run. | Configs committed before the evaluation run | |
| 9 | **One evaluation run and the analysis.** `scripts/analyze.py`: success with Wilson intervals, McNemar with Holm, paired bootstrap of the loss, success by speed bin, failures by type (collision, wrong orientation, miss, timeout). | Results table and figures for the report | |
| 10 | **Report.** Update the Overleaf report and the proposal doc to the results. | Team sign-off | |

Items 0–3 can run in parallel. Items 5–7 are the long pole, and the
validation split exists so that they can be iterated freely.

## 5. Rules that keep the result honest

- **Seeds.** Training uses `train_rngs`. Everything chosen by looking at
  results uses validation seeds: reward terms, hyperparameters, architecture,
  checkpoints, early stopping. Evaluation seeds start at 1000 and are run
  once per frozen method; if a method changes after that, the next
  evaluation uses fresh seeds and the report says so (`docs/setup.md`
  section 9). The splits stay disjoint at any number of seeds.
- **Scoring** comes only from the harness's ground-truth log. A policy is
  scored by `dap/policy_eval.py` exactly like a controller.
- **The deployed student sees only `pixels`.** Privileged data is for the
  teacher, the critic and labels. A privileged policy is a reference, never
  a method.
- **Parameters** keep the **V** / **E** / **Ours** labels of
  `docs/setup.md`, and citations are checked against the source.
- **Behavior changes** that are intended get a re-recorded regression lock
  (`REGRESSION_WRITE=1 pytest tests/test_regression.py`), named in the
  commit message.

## 6. Decisions for the team

| Decision | Default proposed here | Why it matters |
| --- | --- | --- |
| Scoring: tolerance, timing, loss weights | Keep 15°, no timing requirement, weights as now | Defines success for H1 and shapes the reward |
| Training presets | `nominal` only; `wide` stays out of distribution | Q3 is only a generalization test if `wide` is unseen |
| Student memory | 3-frame stack first, GRU if validation shows the need | Mid-course switches may need memory beyond 3 frames |
| Teacher information | Latency-matched current pose and velocity only | Richer inputs widen the imitation gap |
| Is `servo_depth` frozen? | Yes, as documented | Tuning it further on validation seeds is allowed but must be recorded, and it is not the project's focus |

## 7. Risks and fallbacks

- **The student does not learn.** Check the teacher first (item 6's bar).
  Then try, in order: a GRU, a lower resolution (`camera_scale=0.125`), and
  an auxiliary loss predicting the partner palm pose (GenH2R adds a
  forecasting loss to its point-cloud student; Wang et al. 2024). A negative
  result with the teacher − student gap measured is still a result.
- **The comparison looks unfair.** `servo_depth` is a simple pipeline, so
  beating it alone is weak. The ladder shows where the student sits between
  `servo_depth` and the references with a perfect pose, and Q2 reports the
  learned gap next to the classical one.
- **Compute.** Rendering dominates. Fallbacks are a smaller image, fewer
  DAgger steps, or more workers.
- **Overfitting to the partner model.** The partner is a min-jerk reach;
  real handover reaches are mostly curved (Chan et al. 2020). The project
  makes no claim about real people.

## 8. Getting started

```bash
make setup && make test                                           # 101 tests, ~4 min
python scripts/run_experiment.py first_look --controller servo_depth   # the baseline on validation seeds
python scripts/show_scene.py --help                               # look at a scenario
```

Read first: `README.md`, `docs/setup.md` (sections 1b, 8, 9, 11),
`docs/perception_pipeline.md`, and `reports/Dap task research landscape.md`
(the literature review behind this plan).

## References

Details as checked in `docs/setup.md` and `research_notes/Dap task research landscape/`.

- Abeyruwan et al. Agile catching with whole-body MPC and blackbox policy learning. L4DC 2023 (per the arXiv comments), arXiv:2306.08205.
- Agarwal, Kumar, Malik, Pathak. Legged locomotion in challenging terrains using egocentric vision. CoRL 2022, arXiv:2211.07638.
- Bäuml et al. Kinematically optimal catching a flying ball with a hand-arm-system. IROS 2010.
- Chan, Tran, Sheikholeslami, Croft. An experimental validation and comparison of reaching motion models for unconstrained handovers. IEEE-RAS Humanoids 2020, arXiv:2108.12780.
- Chen, Zhou, Koltun, Krähenbühl. Learning by cheating. CoRL 2019, arXiv:1912.12294.
- Connor. Sample size for testing differences in proportions for the paired-sample design. Biometrics 43(1):207–211, 1987.
- Kim, Shukla, Billard. Catching objects in flight. IEEE T-RO 30(5):1049–1065, 2014.
- Ross, Gordon, Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning (DAgger). AISTATS 2011, arXiv:1011.0686.
- Wang et al. GenH2R. CVPR 2024, arXiv:2401.00929.
- Weihs et al. Bridging the imitation gap by adaptive insubordination. NeurIPS 2021, arXiv:2007.12173.
- Yarats, Fergus, Lazaric, Pinto. Mastering visual continuous control: improved data-augmented reinforcement learning (DrQ-v2). arXiv:2107.09645.
- Zhang, Liang, Chen, Ze, Xu. Catch It! Learning to catch in flight with mobile dexterous hands. arXiv:2409.10319 (venue unverified).
- Zhuang et al. Robot parkour learning. CoRL 2023, arXiv:2309.05665.
