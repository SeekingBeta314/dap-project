# Learning a dap from depth

The depth controller learns to predict the seven joint angles that bring its own palm to the partner's intended meeting pose. The simulator supplies an inverse-kinematics solution as a training label. During evaluation, the controller receives the robot's initial and current joint angles, time since the start of the trial, and the newest delivered depth image. Changes in the predicted joint angles change its commanded velocity each control cycle. The comparison policy trains with the same joint-angle labels, split and optimizer, using only initial and current joint angles and time.

The selected policy uses the convolutional encoder and two wrist branches described below. On the 32 development scenarios it achieved 8 aligned contacts. The two coarse-depth models each achieved 3. The separate held-out evaluation is reported in [the study results](results/matched_summary.csv). The [development comparison](results/development_models.csv) records every candidate's success count and loss.

## Generate training examples

Run from the repository root with the simulator environment active. The cluster sweeps use one BLAS thread per simulator worker. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before parallel collection or evaluation.

```bash
python scripts/collect_pixel.py --first 0 --count 1024 --workers 8
```

Each training seed uses the training RNG namespace. The simulator draws a nominal scenario and computes its closest collision-free meeting configuration from the arm's starting posture. Twenty snapshots per accepted scenario span the partner's motion and the following 0.85 s. Own-arm poses vary around trajectories with randomized progress toward the expert posture. Partner poses in the snapshots come from 34–67 ms before the nominal snapshot time to approximate the delivered camera view. Training applies the same medium depth-noise preset as evaluation. Snapshot generation samples static views independently and does not model the own arm's motion within that latency window. Evaluation uses the environment's actual camera buffer and control loop.

## Fit and replay

The fitting scripts require PyTorch and consume NumPy data files, so training may run in a separate environment. The first command below fits the direct-regression development candidate in `banks/direct`. The two evaluation commands load the selected, versioned checkpoints in `models/policies`.

```bash
python scripts/train_pixel.py --epochs 80 --seed 761
python scripts/run_experiment.py speed_limits --seeds 150 --workers 8 \
  --policy dap.pixel_policy:pixel_factory --policy-name pixels
python scripts/run_experiment.py speed_limits --seeds 150 --workers 8 \
  --policy dap.pixel_policy:proprio_factory --policy-name proprio
```

The direct-regression candidate takes the closest valid return in each 4 × 4 patch of the 120 × 212 depth image. Missing patches take the sensor's 3 m far-range value. In this candidate two ReLU layers of 192 and 128 units predict the normalized goal configuration. Its two networks minimize smooth L1 loss with AdamW. The split groups examples by scenario, reserving seeds divisible by five for validation. Normalization uses training examples, and the validation mean absolute error in standardized joint coordinates selects an epoch. Each saved model includes the feature and target scales. A neighboring JSON file records collection seeds, snapshot counts, model selection and the checkpoint hash.

At 20 Hz the policy subtracts the current joint angles from the estimated goal and divides by a 0.22 s servo time constant. The environment clips commands to the actual iiwa joint-speed and position limits. The depth policy holds position until the first camera image arrives. The arm-state comparison can move on its first step. The choice of network, pooling, augmentation, servo gain and optimizer settings are project engineering choices.

## Development comparison

We trained three candidates and reserved nominal scenarios 900001–900032 for closed-loop development. These seeds use the evaluation scenario generator and lie outside the final comparison (0–149). In each candidate, the blind comparison gets the same examples, labels, wrist output representation and training objective as the sighted model. We select the candidate on its depth policy's development success count, breaking ties with mean dap loss.

| Candidate | Training scenarios | Depth representation | Goal representation |
| --- | ---: | --- | --- |
| Direct regression | 1,024 | Closest return in 4 × 4 patches | Seven joint angles |
| Wrist branches | 4,096 | Closest return in 4 × 4 patches | Two possible goal configurations and a learned branch score |
| Convolutional wrist branches | 4,096 | 56 × 68 central depth crop, three learned convolutions | Two goal configurations, branch score, additional local palm-pose loss |

The two wrist targets address a property of the sampled demonstrations. Across the first 1,024 training scenarios, the sign of the sum of expert joint angles 5 and 7 agreed with the direction of the target's finger axis along world y in 99.1% of cases. Regressing both postures with one head can average them into a poorly aligned hand. During training we select the regression head by the sign of the expert joint sum and supervise the branch classifier with binary cross entropy. During control we select it by the sign of its learned logit, smoothing each new logit with weight 0.25. These outputs use only deployment observations.

For the second candidate, first collect the 1,024 direct-regression training scenarios described above. In the additional 3,072 scenarios, 65% of snapshots sample a reaching posture independently of the demonstrated endpoint. This weakens the artificial correlation between own-arm position and the label in expert trajectories. Use a distinct output directory to keep the augmentations identifiable.

```bash
python scripts/collect_pixel.py --first 1024 --count 3072 --workers 8 \
  --independent-poses --out banks/pixel_more
python scripts/train_pixel.py --data banks/pixel_data banks/pixel_more \
  --split-wrist --out banks/wrist
```

The convolutional candidate renders the same 4,096 scenario seeds with independent arm-pose augmentation throughout. It takes rows 0–111 and columns 38–173 from the 120 × 212 camera image and pools to 56 × 68. Three stride-two convolutions (16, 24 and 24 channels, 5 × 5 then 3 × 3 kernels) encode that crop. The blind comparison omits the image features. Both models use the same two hidden layers of 192 and 128 units. Their loss adds a term for the local change in palm pose induced by a joint prediction error. Drake's spatial Jacobian, evaluated offline at the expert joint configuration, maps joint errors to first-order angular and linear errors, scaled by 0.26 rad and 0.05 m. Model selection across epochs minimizes the validation objective using the predicted branch. The NumPy export implements the trained convolutions and joint policy for deployment. All architectures, loss scales, and training hyperparameters are project choices.

```bash
python scripts/collect_pixel.py --count 4096 --workers 8 \
  --independent-poses --fine-view --out banks/pixel_fine
python scripts/goal_jacobians.py --data banks/pixel_fine --out banks/goal_jacobians.npz
python scripts/train_visual.py --data banks/pixel_fine --jacobians banks/goal_jacobians.npz \
  --out banks/convolutional/depth.npz
python scripts/train_visual.py --data banks/pixel_fine --jacobians banks/goal_jacobians.npz \
  --out banks/convolutional/proprio.npz --blind
```

The direct evaluation API can replay development trials from any saved checkpoint. The example below runs one seed; the development comparison iterates through 900001–900032.

```python
from dap.experiment import CONFIGS, Condition
from dap.pixel_policy import PixelPolicy
from dap.policy_eval import run_policy_trial

policy = PixelPolicy("banks/convolutional/depth.npz")
row = run_policy_trial(CONFIGS["speed_limits"], Condition("pixels", 1.0), 900001, policy)
print(row.outcome, row.success, row.rot_err)
```

After fitting, the selected files from the convolutional run can be copied from `banks/convolutional/` to `models/policies/` to evaluate the rebuilt policies through their normal factories. The repository already includes frozen versions of `depth.npz`, `proprio.npz`, and each associated JSON manifest. Validation uses 820 of the 4,096 training-distribution scenarios (16,400 snapshots), leaving 3,276 scenarios (65,520 snapshots) for gradient updates. The compressed development trial records preserve the outcomes of all 96 candidate rollouts.
