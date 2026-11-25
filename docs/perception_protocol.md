# Held-out depth perception study

## Design fixed before evaluation

How often can a frozen controller meet a moving palm, with its fingers antiparallel, using only a head-mounted depth camera and its own arm state? The primary comparison is the depth student against the classical depth-perception servo. Both use the same camera specification (212 × 120, 30 Hz, one-frame latency, medium depth noise) and the same per-seed noise stream. Each closes its own feedback loop. The arm poses, occlusions and resulting images can diverge after their first different command. The student acts at 20 Hz and the servo at 100 Hz. The comparison measures complete controllers.

- Run the eight methods listed below on evaluation seeds **1000–1171 inclusive** on each of two presets. Each seed fixes the randomized partner, layout and arm start. Pair methods within a preset. The two presets are separate experiments. Training uses the nominal preset and tagged training streams. Checkpoints and hyperparameters are chosen on separate validation streams and frozen before evaluating each method.
- **Primary outcome.** Ground-truth first contact from the front, within the benchmark palm-distance and overlap tolerances, with full relative-rotation error at most 15°. Timing is recorded separately. See [contact and scoring](setup.md).
- **H1 and generalization.** Test student versus classical depth on nominal and repeat on wide (wider layouts and possible mid-course target switches). Use exact two-sided McNemar tests on matched successes. Adjust these two p-values together with Holm at familywise 0.05. Report both discordant counts, the success-rate difference, and the secondary difference in the benchmark's tiered loss.
- **Cost of perception.** Estimate teacher minus student and camera-timed true-pose servo minus classical-depth servo separately for each preset. Report the absolute difference in success rates and its paired 95% interval. Bootstrap per-seed differences with 10,000 resamples and fixed RNG seed 0. Interval columns are pointwise. Loss differences use the same paired bootstrap.
- **Descriptive results.** Report counts and 95% Wilson intervals, first-event outcomes, and performance by realized peak palm speed. Report aligned contacts within 100 and 200 ms of the intended meeting as sensitivity analyses. A timeout with closest face centers at most 5 cm is labeled a near miss for the failure breakdown. The threshold is a descriptive choice. All timeouts keep their original score.
- A controller exception is a failure with the collision-tier loss. Scenario sampling failures are excluded for all methods on that seed and counted. A harness exception, missing seed, inconsistent sampled scenario or inconsistent checkpoint invalidates the affected comparison until resolved.

| Method label | Information and purpose |
| --- | --- |
| student | Depth and own state. Distilled student. |
| servo_depth | Classical palm tracker on the same depth sensor and a differential IK servo. |
| teacher | Frozen teacher using latency-matched privileged partner state. Learned reference. |
| servo_gt_cam | Same servo given true pose at the camera's capture times and delay. |
| servo_gt | Same servo with current true pose. |
| servo | Same servo with the stand-in noisy pose sensor. |
| blind | Fixed pose prior in the robot's approach frame. In the existing baseline the prior is fit from 30 training-stream scenes of the corresponding preset, including wide for the wide reference. |
| oracle | Knows the intended meeting configuration and time. |

## Sample size

The validation baseline achieved 0/12 aligned touches on nominal and 3/12 on wide. Before opening evaluation results, we selected an absolute improvement of 0.20 (one additional aligned touch in five) as the smallest effect targeted by the design. At the pilot baseline proportions, the assumed student rates are therefore 0.20 and 0.45. For each pair of marginal rates we used the largest possible probability of discordance (0.20 and 0.70). We powered each two-sided exact test at the conservative Holm first-step level of 0.025.

For each candidate number of seeds, the number of discordant pairs D is binomial with parameters n and q (the discordance probability). Conditional on D, wins favoring the student are binomial with parameters D and (1 + 0.20/q)/2. Summing the probability of both exact McNemar rejection tails over D gives minimum sizes of 44 and 172 at 80% power for nominal and wide. We use **172 in both presets**. Under these assumptions, nominal power exceeds 99.99% and wide power is 80.07%. The 12-seed pilots leave considerable uncertainty about the true marginals. Smaller effects, or different discordance rates, can produce substantially less power. See [the original study plan and pilot](phase2_plan.md).

The cupped-hand choreography is a separately authored illustration. All scored methods above use the same rigid-hand, first-contact benchmark.
