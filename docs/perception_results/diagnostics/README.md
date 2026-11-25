# How accurately could the student see the partner's palm?

The camera captures at 30 Hz and the student acts at 20 Hz. At every decision with an available image, the replay records the palm center, orientation and velocity estimated by the frozen student. The reference is the simulator pose at the newest delivered image's capture time. This accounts for camera latency. The measurements come from the same estimates the actor used to choose its actions. They are compared with ground truth after inference.

| Held-out reaches | Trials with estimates | Median center error | Median orientation error |
| --- | ---: | ---: | ---: |
| Nominal, aligned | 114/114 | 1.97 cm | 6.51° |
| Nominal, failed | 58/58 | 4.41 cm | 21.57° |
| Wide, aligned | 93/93 | 2.21 cm | 7.16° |
| Wide, failed | 79/79 | 4.33 cm | 21.92° |

On both presets, the typical terminal orientation error was roughly three times as large in failed trials. These numbers measure the network's estimate of the partner's orientation. The benchmark scores a different quantity, the actual relative orientation of the two palms at their first touch.

The [figure](pose_errors.pdf) groups the 344 held-out student trials by preset and eventual outcome. For each reach, it takes the median error over the final 200 ms before termination. Each dot is then the median of those trial values, so a long timeout and a short contact each contribute one observation. The horizontal lines span the 10th to 90th percentiles. Position is the Euclidean distance between the corresponding palm centers. Orientation is the angle of the full relative rotation, so reversing the fingers counts as a large error. Each trial ends at its own first event or timeout. These groups describe an association with the outcome.

[Figure for video](pose_errors.png) · [Group counts and medians](pose_summary.csv) · [All nominal trial measurements](nominal.csv) · [All wide trial measurements](wide.csv)

## Reproduce

From a Git clone of the study branch, run the following commands with the [recorded dependencies](../constraints.txt). The replay checks the scenario digest, outcome, event time and contact measurements against each archived trial before publishing its CSV. The cross-CPU study documents how small numerical differences can change individual closed-loop measurements. [Cross-CPU results](../portability/)

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/student_pose_replay
for preset in nominal wide; do
  .venv/bin/python scripts/audit_student_observations.py --preset "$preset" \
    --rows "docs/perception_results/$preset/trials.jsonl.gz" --workers 8 \
    --out "$out/$preset.csv"
done
.venv/bin/python scripts/plot_student_diagnostics.py "$out"
```
