# Intercepting a moving palm from depth

A robot that learned to locate the partner palm in depth images reached aligned contact in **114 of 172 nominal trials (66.3%)**. A classical depth tracker and servo reached it in **21 (12.2%)** on the same scenarios. The advantage persisted on wider, previously unseen layouts. Both used the same simulated camera and fixed checkpoint or controller throughout the evaluation. The learned system also had fewer collisions, although 48 of its 172 wide trials still ended in collision.

The study covers **2,752 controller trials** across eight methods and 344 preset–seed combinations. Each method used the same 172 reserved seeds on each preset. Every trial completed with a recorded first-event outcome. The [protocol](../perception_protocol.md) fixes the sample size, endpoints and two-test family before evaluation. The [scenario archive](scenarios/) identifies the exact layouts, initial joint configurations and partner trajectories.

## Does the depth student improve aligned contact?

| Evaluation preset | Depth student | Classical depth | Student-only / classical-only successes | Absolute improvement, pointwise 95% paired interval | Holm-adjusted exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 114/172 (66.3%) | 21/172 (12.2%) | 97 / 4 | +54.1 percentage points [45.9, 62.2] | 6.71 × 10⁻²⁴ |
| Wide | 93/172 (54.1%) | 17/172 (9.9%) | 81 / 5 | +44.2 percentage points [36.0, 52.3] | 9.58 × 10⁻¹⁹ |

Success requires the first front-side palm touch to be within 15° of the target relative rotation. A time window is assessed separately below. The student improved the secondary benchmark loss as well. With lower scores preferred, its paired mean difference from the classical system was −9.65 points on nominal (95% interval −10.91 to −8.33) and −7.92 on wide (−9.33 to −6.49). [Paired comparisons](paired.csv) · [IEEE table](paired_table.tex)

## How much performance is lost when the controller must perceive the partner?

The teacher receives the partner's true pose and velocity at camera timing. The student estimates them from depth and acts through the frozen teacher network. The classical comparison substitutes true camera-time pose for the depth tracker while keeping the classical servo.

| Approach | Nominal success with privileged input | Nominal gap to depth | Wide success with privileged input | Wide gap to depth |
| --- | ---: | ---: | ---: | ---: |
| Learned teacher and student | 163/172 | 28.5 points [21.5, 35.5] | 145/172 | 30.2 points [22.1, 38.4] |
| Classical servo | 145/172 | 72.1 points [64.5, 79.1] | 133/172 | 67.4 points [59.9, 74.4] |

The gaps are absolute success-rate differences on matched trials with pointwise 95% paired bootstrap intervals. The classical controller's command rate is 100 Hz. The student acts at 20 Hz. The classical tracker receives the static layout and camera calibration so it can render the expected background and locate the partner's shoulder. The spatial student uses the fixed camera calibration and receives depth and its own arm state. The trial layout is absent from the student observation. The teacher receives the initial partner pose while the student holds its arm until its first usable image. These differences matter when interpreting each whole-system comparison. [Paired estimates](paired.csv) · [Observation definitions](../../dap/envs/dap_env.py) · [Classical perception](../perception_pipeline.md)

## What changed on wide trajectories, and when did contact happen?

- **Generalization.** The student's success rate fell from 66.3% to 54.1% on the wide preset. The wide sample reached 0.447 m of lateral offset and 44.5° of base yaw in magnitude, compared with 0.298 m and 29.9° on nominal. The partner switched target mid-reach in 58 of the 172 wide trajectories and none of the nominal trajectories. The student succeeded in 29/58 switched and 64/114 unswitched wide trials. Classical depth succeeded in 3/58 and 14/114, respectively. The switch split is exploratory. [Switch breakdown](by_switch.csv) · [Frozen trajectories](scenarios/)
- **Speed.** In the fastest realized-speed bin (at least 0.75 m/s), the student succeeded in 22/44 nominal and 11/36 wide trials. The slowest bin had 22/34 and 21/32. Speed, feasible layouts and the accepted partner trajectory vary together, so these are descriptive trends. [Speed bins and confidence intervals](by_speed.csv)
- **Failures.** The nominal student had 18 misaligned touches, 34 collisions and six timeouts. On wide it had 24 misaligned touches, 48 collisions and seven timeouts. The classical depth system collided on 121 trials in each preset. [Outcome counts](outcomes.csv)
- **Motion and timing.** Of the student's 114 nominal successes, 86 happened before the partner's final trajectory knot and 28 at or after it. The corresponding wide counts were 66 and 27. With the additional requirement of contact within 100 ms of the intended meeting time, student successes fell to 22/172 nominal and 15/172 wide. Classical depth achieved 5/172 and 4/172 in that window. Aligned touch and precisely timed interception are different performance goals. [Motion at successful contact](motion_at_success.csv) · [Timing sensitivity](timing.csv)

The limited number of tightly timed successes motivates a follow-up experiment. Fix a timing window before training, compare against a controller that explicitly predicts an intercept time, and report both alignment and collision rates on fresh scenarios. This would test whether a stronger timing objective improves contact precision without increasing collisions.

## Reproduction on another machine

We repeated all 344 student trials after the evaluation closed on a single additional CPU host. Every binary aligned-contact label agreed. One wide timeout became a collision. Continuous contact errors and timing can vary across machines. In the most divergent nominal trial the arm still made misaligned contact, but the measured rotation error shifted by 42.70°. The original 2,752 trials remain the basis of the primary results. [Reproduction records, hardware and instructions](portability/)

## Figures, tables and raw records

| Artifact | How to use it |
| --- | --- |
| [Aligned first contacts (PDF)](figures/success.pdf) · [PNG](figures/success.png) | Aligned contact on matched trials with 95% Wilson intervals. Of the two depth methods, the classical tracker receives the trial layout for background rendering and shoulder location. The student acts at 20 Hz and the classical servo at 100 Hz. |
| [First-event outcomes (PDF)](figures/outcomes.pdf) · [PNG](figures/outcomes.png) | The classical depth system collides in 121 trials per preset, versus 34 and 48 for the student. Bars account for every scored trial. |
| [Speed (PDF)](figures/speed.pdf) · [PNG](figures/speed.png) | Student and classical rates within bins of the partner's realized peak speed. The wide student succeeds in 11 of 36 trials in the fastest bin. The bin counts and interval limits are in [by_speed.csv](by_speed.csv). |
| [IEEE table](paired_table.tex) | Include this file with `\input` in an IEEE conference manuscript. It contains both prespecified comparisons and the Holm-adjusted p-values. |
| [Absolute rates](rates.csv) · [Outcome counts](outcomes.csv) | All eight methods and both presets, with sample counts, intervals, continuous contact metrics and mean loss. |
| [Paired contrasts](paired.csv) | Each comparison records both discordant counts, paired differences (first method minus second), intervals and the exact McNemar p-value. Only the two student–classical comparisons enter the Holm-adjusted family. |
| [Timing](timing.csv) · [Contacts before the final knot](motion_at_success.csv) · [Target switches](by_switch.csv) | Sensitivity to timing, reach completion and the subset of wide trajectories with a mid-course switch. The switch split is exploratory. |
| [Nominal records](nominal/trials.jsonl.gz) · [Wide records](wide/trials.jsonl.gz) | One strict JSON object per seed and method, with ground-truth outcome, timing, geometry, scenario SHA-256 and controller measurements. |
| [Nominal manifest](nominal/config.json) · [Wide manifest](wide/config.json) | Seed range, scoring tolerances, method, code and checkpoint hashes, software versions, camera settings and the frozen scenario-archive digest. |
| [Evaluated scenarios](scenarios/) · [Teacher fit](training/teacher/) · [Student fit](training/student/) | Exact robot layouts and joint trajectory knots, training instructions, validation selections and provenance. |

## Reproduce

The tables and figures can be rebuilt from the published measurements without refitting policies or rerunning simulation. Run the command from the root of the study branch with the project Python environment.

```bash
.venv/bin/python scripts/analyze_perception.py docs/perception_results --out results/perception_rebuilt
```

The following runs all 2,752 trials from the two frozen checkpoints and 344 archived scenarios. Use the [recorded software versions](constraints.txt) to reproduce the evaluated environment. Each method can run independently. The runner reuses completed records after an interruption, checks source and checkpoint identity when resuming, and supports `--offset` and `--count` for disjoint seed shards.

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
out=results/perception_rerun
for preset in nominal wide; do
    for method in servo_depth servo_gt_cam servo_gt servo blind oracle; do
        "$PY" scripts/run_perception_study.py "$preset" "$method" --out "$out" --workers 8
    done
    "$PY" scripts/run_perception_study.py "$preset" teacher --out "$out" --workers 8 \
        --checkpoint models/policies/teacher
    "$PY" scripts/run_frozen_student.py "$preset" --out "$out" --workers 8 \
        --checkpoint models/policies/student.pt
done
"$PY" scripts/analyze_perception.py "$out" --out results/perception_rerun_tables
```

A single checkpoint for each learned method was evaluated. The uncertainty intervals describe variation over simulated scenarios. The cameras share the same calibration, timing, noise model and per-seed noise stream. Different actions can cause different occlusions after the robots begin moving. The partner follows a preprogrammed reach. The first-contact score uses geometry alone. The cupped-hand choreography and these benchmark trials serve different purposes. [Protocol](../perception_protocol.md) · [Camera agreement test](../../tests/test_perception.py) · [Motion reference and contact audit](../hand_dap.md)
