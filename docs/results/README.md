# Trial records and comparisons

Each experiment directory includes a run manifest (`config.json`), one compressed JSON record per trial (`trials.jsonl.gz`), and tables derived from those records. The manifests specify the scenario distribution, seeds, tolerances, simulator versions, and hashes of the code used by the sweep. Inference manifests also identify the bundled weights. The [report](../report.pdf) interprets the results.

| Question | Data |
| --- | --- |
| How did the controllers perform on the same scenarios at nominal noise? | [Matched rates and intervals](matched_summary.csv), [paired comparisons](matched_paired.csv) |
| What changed when the controller replanned? | [Three pose-noise levels, 150 paired seeds each](replan_vs_once/paired.csv) |
| How did replanning perform across partner speeds? | [300 trials grouped by realized peak speed](speed_limits/by_speed.csv) |
| How often did an aligned touch fall within 100 or 200 ms of the intended time? | [Timing sensitivity](matched_timing.csv) |
| Did aligned contact occur before the partner stopped? | [Motion at successful contact](motion_at_success.csv), [reconstructed stop times](motion_times.csv) |

Success is an aligned front-side palm contact. Contact counts include aligned and misaligned touches. Timing error is signed in seconds, relative to the intended meeting time. The row labeled `pixels` is the depth policy. The `proprio` policy observes only the arm state and trial time. The `oracle` receives the intended meeting configuration and time. The `once` and `replan` methods use the noisy stand-in pose sensor. Noise level 1 means 5 mm and 2° standard deviations for that sensor.

The six-method comparison uses the intersection of evaluated seeds at nominal noise. Rows from the same seed across conditions are paired observations. The 300-seed speed study includes the first 150 seeds used in the noise study. Adding those sample sizes would count some scenarios twice. To regenerate tables from the archived records, see the [repository instructions](../../README.md#experiments).
