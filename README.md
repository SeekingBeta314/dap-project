# Dap: Angle and Alignment Matching for Robotic Hands

6.4210 final project — Josh Nix, Oliver Wang, Samuel Zhou.

Two simulated KUKA iiwa arms meet palm to palm in Drake. A learned controller watches the partner through a fixed head camera and commands our arm while the partner follows a randomized reach. The goal is a front-side touch with the fingers pointing in opposite directions. The simulator ends the trial on contact, collision, or timeout and measures alignment, timing, and relative speed from ground truth.

## Study

On 150 matched held-out trials the depth policy made **43 aligned touches (28.7%)**, compared with **1 (0.7%)** for the arm-state-only policy and **33 (22.0%)** for classical replanning. All 43 depth successes occurred before the partner stopped, and 18 were within 100 ms of the intended meeting time. The depth policy collided in 86 trials. Its success-rate difference from the replanner was inconclusive.

[Watch the first successful trial, seed 0](docs/success.gif) · [Watch the first collision, seed 2](docs/collision.gif)

- [Final report](docs/report.pdf) and [audited trial records and tables](docs/results/)
- [Simulation, scoring and parameter sources](docs/setup.md)
- [Living proposal](https://claude.ai/artifact/QUBj973spHWSZKigTEf1Cz)

The depth policy and its independently trained arm-state-only comparison are in `dap/pixel_policy.py`. The classical controllers in `dap/meeting.py` and `dap/planner.py` use a noisy pose sensor, inverse kinematics, and joint trajectory optimization. The oracle in `dap/baselines.py` knows the intended meeting time and joint positions. Each method is evaluated on the same scenario seeds.

## Setup

Needs Python 3.12 or 3.13 and Drake 1.57+ (tested on 1.57 and 1.58). `make
setup` picks `python3.13`, then `python3.12`, then `python3`, and stops if
that is older than 3.12 (Anaconda's default `python3` is often 3.11, which
Drake does not support). Override with `make setup PYTHON=/path/to/python3.13`.
The cluster runs used Drake 1.58.0 in an Ubuntu 24.04 container with Mesa EGL for headless rendering. The wheel requires glibc 2.34 or newer on x86-64 Linux. From the repo root:

```bash
make setup     # creates .venv and installs requirements.txt (once)
```

## Run

The `make` targets use `.venv/bin/python` directly, so there's no need to
activate anything:

```bash
make test                          # pytest
make scene SEED=0                  # Meshcat view of a sampled scenario (SCENARIO=fixed|nominal|wide)
make dry-run CONFIG=replan_vs_once # list an experiment's conditions
make experiment CONFIG=smoke       # run a sweep
```

To run scripts by hand, activate the venv first (`source .venv/bin/activate`,
once per terminal); then `python` means the venv's Python. Without it, macOS
only has `python3`, which is your system Python and not the project's.

> Drake downloads its robot models on first use and that download breaks if the
> venv path contains a space (e.g. `MIT Stuff/`). If you hit `returncode == 512`,
> clone the repo somewhere without spaces and `make setup` there.

## Run the depth policy

The fitted policies are stored in `models/policies/`. Inference uses NumPy. The simulator provides a stack of three delayed depth frames and our arm state at 20 Hz. The fitted depth policy uses the newest available frame.

```bash
.venv/bin/python scripts/run_experiment.py speed_limits --seeds 3 --policy dap.pixel_policy:pixel_factory --policy-name pixels --out results/demo
.venv/bin/python scripts/show_policy.py dap.pixel_policy:pixel_factory dap.gif --seed 0
```

Each GIF shows an external camera beside the depth frame available to the controller. For the arm-state-only comparison, use `dap.pixel_policy:proprio_factory` and the label `proprio`.

## Train and reproduce

The training collector renders nominal scenarios with dedicated training random streams and labels them with inverse-kinematics solutions. A validation split selects the model checkpoint. The [training instructions](docs/pixel_policy.md) give the commands and dependencies. The [Gymnasium interface](docs/setup.md) also accepts a scenario bank and a custom reward function for other learning experiments.

## Experiments

Each named config in `dap/experiment.py` is a grid of conditions (planner x
sensor-noise scale) times seeds, on a scenario preset
(`ExperimentConfig.scenario`, default `nominal`). Every condition reuses the
same seeds, and a seed fixes the whole scenario (layout, partner path and
speed, our start) and the sensor noise, so comparisons are paired. Partner
speed is not a condition: it is drawn per scenario and analysed against the
realized speed.

**Scoring** (docs/setup.md sections 1a–1b): a trial ends at the first
contact (the palm boxes touch, from the front) or collision. Success =
contact with the palms antiparallel (rotation error ≤ 15°). Position, alignment, timing and contact speed are
continuous scores, and `dap.scoring.dap_loss` combines them over a whole
results table using the weights specified in the setup document. Everything is computed from the
ground-truth log; nothing the controller reports is scored.

| Config | Tests | Conditions |
| --- | --- | --- |
| `smoke` | Plumbing | replan only, 3 seeds |
| `replan_vs_once` | H1 | planner {once, replan} x noise {0.5, 1, 2}, 150 seeds |
| `speed_limits` | H2 | replan, 300 seeds, analysed by partner speed |

```bash
python scripts/run_experiment.py smoke --dry-run           # list conditions, run nothing
python scripts/run_experiment.py replan_vs_once --workers 4
python scripts/run_experiment.py smoke --controller oracle # scripted reference, same seeds
python scripts/analyze.py results/replan_vs_once           # summary, paired tests, by speed
```

Sweeps resume completed trials. Exceptions are recorded with their traceback and counted according to the [crash policy](docs/setup.md). The study uses the repository's 15° orientation tolerance. Timing is a separate continuous score.

To regenerate analysis tables and plots from the archived raw measurements:

```bash
.venv/bin/python scripts/publish_study.py --runs docs/results --out results/rebuilt --policies speed_limits_pixels speed_limits_proprio
.venv/bin/python scripts/plot_outcomes.py --runs results/rebuilt --out results/rebuilt/outcomes.png
.venv/bin/python scripts/motion_audit.py
```

## Working together

- Branch per workstream (`scene/partner-arm`, `meeting-point`, `planner`, ...),
  open a PR into `main`, one other person reviews.
- Keep `main` green: `pytest` passes before merging.
- Owners for each module are tracked in section 6 of the proposal doc.
