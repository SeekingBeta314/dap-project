# Training the depth student

The released model is the heatmap-and-finger-sign student after its on-policy DAgger round. It achieved 47 aligned contacts in 64 scenarios on the fresh nominal validation cohort, compared with 24 for the whole-image regression model. Both had been nominated using a separate 64-scenario cohort. The figure and records below cover model development. The main study reports performance on separate held-out evaluation seeds.

[Validation-stage figure](validation_stages.png) · [Frozen selection and hashes](selection.json) · [All 512 validation trial records](validation.jsonl.gz) · [Rollout manifests](datasets.json) · [Fitting settings and traces](fits.json.gz) · [Initial diagnosis](initial_diagnostic.json) · [Released model](../../../../models/policies/student.pt)

The student runs at 20 Hz. It receives the three latest delayed 212 × 120 depth frames and the robot's 31 proprioceptive values. It holds still until a frame arrives. Each 2 × 2 image block contributes its nearest valid depth. Both candidate architectures use the resulting images and normalized proprioception to estimate the partner palm's position, velocity, normal and finger axis at the newest frame's capture time. They orthogonalize the two estimated axes. A frozen copy of the selected teacher's actor turns the estimate and the robot's current state into seven bounded joint velocities. The deployed checkpoint contains both networks.

The simulator supplies the true partner pose and velocity at the capture time for offline supervision. For action distillation, the same frozen actor runs twice on each training sample. One pass receives the inferred partner state and the other receives the training label. We minimize the mean squared difference between these actions plus a smooth L1 error on the standardized partner state. Pose-only warm-up uses the second term. The final closed-loop scores use the deployed student's estimates throughout.

The first architecture concatenates features from four stride-two convolutional layers (16, 24, 32 and 40 channels) with arm state and predicts the twelve partner-state values through two hidden layers (192 and 128 units). The encoder and pose head contain 270,636 trainable parameters. The actor contains 81,927 frozen parameters in both architectures. We kept spatial features to preserve image location and used supervised pose error to give the camera network a training signal even when different partner poses yield the same clipped action. The first architecture uses random image shifts of up to one pooled pixel, applied to all frames together. Both architectures hide the normalized proprioceptive input to the pose head for 35% of fitting samples. The actor always receives the robot's actual arm state. This reduces the incentive to infer the partner's position solely from the robot's own wrist on successful demonstrations. The architecture, augmentation magnitude, proprioception masking rate, optimization settings and rollout counts are project design choices.

An episode uses only the nominal training RNG stream or the nominal training bank. Collection stores the delivered image stack and the matched privileged vector for offline labeling. Closed-loop validation uses an independent validation stream. Checkpoint selection maximizes aligned first contacts on nominal validation seeds 300–363, with ties resolved by lower mean benchmark loss. The held-out evaluation stream is separate.

[Student and inference code](../../../../dap/student.py) · [Collection, fitting and validation](../../../../scripts/train_student.py) · [Teacher](../teacher/) · [Observation and seed definitions](../../../setup.md)

The planned selection candidates use the same selected teacher: a fit on the initial demonstration corpus, a fit after adding trajectories visited by an early student, and a fit after a further round driven by the updated student. The initial data combine privileged servo demonstrations and trajectories from the earlier IK-goal policy, which expose the vision network to arm configurations away from expert demonstrations. A provisional student with the teacher's warm-start actor collects the first learner trajectories. Its validation check is diagnostic. The selected teacher labels all final candidate training sets.

## Follow-up to the initial validation failure

The provisional student achieved 2 aligned contacts on the 64 selection scenarios. A diagnostic sampled every 32nd trajectory from the completed set of 1,536 learner rollouts. Across the 877 middle steps of these 48 training episodes, median palm-position error was 9.1 cm and median finger-axis error was 54.5°. The 80th-percentile finger error was 127.3°. This motivated a second architecture that learns an image heatmap for the palm, uses camera calibration to convert the estimated image location and depth into a world position, and reads an image patch centered on the palm. It learns the finger axis and its sign separately. All camera observations, teacher labels and evaluation rules are shared with the first architecture.

The spatial model has 184,988 trainable parameters. It retains the image coordinates by dispensing with random shifts. Its heatmap loss uses a Gaussian target centered at the projected, latency-matched true palm position, and a separate binary loss supervises the finger-axis sign. The head sees the surrounding 20 × 20 pooled pixels and the full-image context. The follow-up evaluates a fit using the initial demonstration and learner-trajectory sets and another fit after collecting more training visits with the spatial model. We choose within each architecture on nominal validation seeds 300–363. The winners are compared on the same fresh nominal validation seeds 400–463. We freeze the model with more aligned contacts there, resolving ties by lower mean benchmark loss. These two cohorts serve model selection and remain separate from the final evaluation.

## Training trajectories

| Rollout policy | Nominal training seeds | Episodes | Labeled control steps |
| --- | --- | ---: | ---: |
| Privileged servo expert | 0–2,303 | 2,304 | 49,537 |
| Earlier IK-goal depth policy | 10,000–10,767 | 768 | 15,565 |
| Provisional depth student | 20,000–21,535 | 1,536 | 36,418 |
| Global student after its first round | 30,000–31,151 | 1,152 | 23,747 |
| First spatial student | 40,000–40,767 | 768 | 17,358 |

The final global fit used the first four corpora (125,267 distinct labeled steps). The final spatial fit used the first three corpora and the spatial student's own visits (118,878 labeled steps). Each fitting epoch revisits the same corpus. All five collections used the training RNG tag.

## Reproduce the training

Run the following commands from the repository root after installing the project requirements. The collection and validation commands use Drake. The fitting commands can use the same Python environment or another environment with NumPy and PyTorch. GPU fitting used PyTorch 2.5.1+cu121. Set `device=cpu` to fit without a GPU. Floating-point kernels can vary across hardware. The released model and archived rollout policies allow direct replay.

```bash
set -e
S=banks/reproduce-student
PY=.venv/bin/python
FITPY=.venv/bin/python
device=cuda
workers=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$S"

collect() {
  "$PY" scripts/train_student.py collect --workers "$workers" --out "$S/$1" \
    --rollout "$2" --first "$3" --count "$4" "${@:5}"
}
fit() {
  "$FITPY" scripts/train_student.py fit --out "$S/$1.pt" --device "$device" "${@:2}"
}
validate() {
  "$PY" scripts/train_student.py validate --checkpoint "$S/$1.pt" --workers "$workers" \
    --first 300 --count 64 --out "$S/$2.json"
}

collect warm expert 0 2304
collect ik ik 10000 768 --checkpoint models/policies/depth.npz
cp docs/perception_results/training/student/behavior/initial-actor.pt "$S/initial-actor.pt"
"$PY" scripts/train_student.py export-teacher --checkpoint models/policies/teacher --out "$S/final-actor.pt"

fit pose --data "$S/warm" --epochs 24 --batch-size 192
fit warm --data "$S/warm" "$S/ik" --init "$S/pose.pt" \
  --teacher "$S/initial-actor.pt" --epochs 8 --batch-size 192 --lr .0004
validate warm initial-val
collect dagger student 20000 1536 --checkpoint "$S/warm.pt"
fit finalwarm --data "$S/warm" "$S/ik" --init "$S/warm.pt" \
  --teacher "$S/final-actor.pt" --epochs 6 --batch-size 192 --lr .0003
fit round1 --data "$S/warm" "$S/ik" "$S/dagger" --init "$S/finalwarm.pt" \
  --teacher "$S/final-actor.pt" --epochs 12 --batch-size 192 --lr .00035
collect second student 30000 1152 --checkpoint "$S/round1.pt"
fit round2 --data "$S/warm" "$S/ik" "$S/dagger" "$S/second" --init "$S/round1.pt" \
  --teacher "$S/final-actor.pt" --epochs 10 --batch-size 192 --lr .0002

fit spatial --architecture spatial --data "$S/warm" "$S/ik" "$S/dagger" \
  --teacher "$S/final-actor.pt" --epochs 26 --batch-size 160 --lr .00065 --shift 0
collect spatialdata student 40000 768 --checkpoint "$S/spatial.pt"
fit spatial2 --architecture spatial --data "$S/warm" "$S/ik" "$S/dagger" "$S/spatialdata" \
  --init "$S/spatial.pt" --teacher "$S/final-actor.pt" --epochs 12 --batch-size 160 --lr .00025 --shift 0

for name in finalwarm round1 round2 spatial spatial2; do validate "$name" "val-$name"; done
"$PY" scripts/select_student.py confirm --runs "$S" --workers "$workers"
"$PY" scripts/select_student.py publish --runs "$S" --checkpoint "$S/selected.pt" --archive "$S/archive"
```

To regenerate a recorded collection with the exact policy that generated it, give `collect` the corresponding released checkpoint from `behavior/` or `models/policies/student.pt` if that policy won. The published demonstration checkpoint used for `ik` is `models/policies/depth.npz`. The original runs partitioned the same seed ranges into disjoint parallel jobs and merged their per-episode files in seed order.

The two nominees can also be replayed on validation seeds 400–463. The winner is in `models/policies/student.pt`. The runner-up is either the named checkpoint in `behavior/` or `runner_up.pt` in this directory.
