# Dap: Angle and Alignment Matching for Robotic Hands

6.4210 final project — Josh Nix, Oliver Wang, Samuel Zhou.

Two simulated iiwa arms face each other in Drake, in a layout randomized per
trial. The partner arm's hand approaches along a randomized, human-like path;
our arm sees it through a head-mounted RGB-D camera (perception pipeline
still to come; a noisy pose sensor stands in for it) and has to meet it palm
to palm: palms parallel with fingers opposite. The task is purely kinematic:
rigid hands, no clasp, and the trial ends at the touch. Each scenario draws
its own partner speed (peak palm speed 0.15–0.9 m/s).

**Problem setup, every parameter and its source:** [docs/setup.md](docs/setup.md).
Planned perception pipeline: [docs/perception_pipeline.md](docs/perception_pipeline.md).

**Proposal (living draft):** https://claude.ai/artifact/QUBj973spHWSZKigTEf1Cz

## Layout

| Path | What | Status |
| --- | --- | --- |
| `models/dap_hand.sdf` | Rigid flat hand: palm + fixed fingers (`package://dap/...`) | done |
| `dap/scene.py` | Two iiwa7s on pedestals, partner pose from a `SceneLayout`; `Scene` helpers (posing, palms, distances, camera) | done |
| `dap/scenario.py` | Per-trial sampler: layout, meeting point, feasibility, partner motion, start configs; presets `fixed`/`nominal`/`wide` | done |
| `dap/feasibility.py` | Two-arm IK check that a palm-to-palm dap exists (+ reach-in-time check) | done |
| `dap/partner.py` | Min-jerk Bezier palm path at a sampled peak speed (overshoot, optional target switch), executed by the partner iiwa within its limits; noisy pose sensor | done |
| `dap/ik.py` | Single-arm IK helper | done |
| `dap/camera.py` | D435-like head camera (848x480, 58 deg vFOV, 30 fps) + frame-rate/latency buffer | done |
| `dap/depth_noise.py` | D435-style depth noise (disparity noise, jitter, fixed pattern, dropout, shadow, holes) | done |
| `dap/envs/` | Gymnasium env `dap/PalmMeet-v0` (state or state+depth obs; reward hook only) | done; no training |
| `dap/estimator.py` | Constant-acceleration Kalman filter + palm-normal filter | done (first pass) |
| `dap/meeting.py` | Meeting time + palm pose via IK sweep | TODO |
| `dap/planner.py` | Controller contract (`ControllerModel`, `reset`/`step`); `KinematicTrajectoryOptimization` + replanning loop | contract done; planner TODO |
| `dap/baselines.py` | Scripted references: oracle (ground truth) and hold-still | done |
| `dap/scoring.py` | The one definition of contact, collision, success, scores and the (vectorized) loss | done |
| `dap/metrics.py` | Per-trial result record (harness measurements only) | done |
| `dap/stats.py` | Wilson intervals, exact McNemar, paired bootstrap, Holm | done |
| `dap/experiment.py` | Conditions (planner x sensor noise) + named configs (`smoke`, H1, H2) | done |
| `dap/trial.py` | Causal kinematic trial loop: arm limits enforced, ground truth logged, scored afterwards | done; needs the controller |
| `scripts/show_scene.py` | View a sampled scenario in Meshcat, animate it, or save head-camera depth | done |
| `scripts/run_experiment.py` | Run a config, append to `results/<name>/trials.jsonl` | done |
| `scripts/analyze.py` | Paired analysis: rates with Wilson CIs, McNemar, paired bootstrap of the loss, success vs. partner speed | done |
| `tests/` | Scene, camera, depth noise, feasibility, scenario, scoring, stats, trial harness, env (gymnasium `check_env`) | passing |
| `docs/` | Setup (with sources), the planned perception pipeline, and pending fixes for the report draft (`report_fixes.md`) | |
| `notebooks/` | Scratch notebooks (clear outputs before committing) | |
| `results/` | Experiment outputs (git-ignored) | |

Hand frame convention: palm `+z` is the palm normal (the face that meets the
partner), `+x` points toward the fingers.

## Setup

Needs Python 3.12 or 3.13 and Drake 1.57+ (tested on 1.57 and 1.58). `make
setup` picks `python3.13`, then `python3.12`, then `python3`, and stops if
that is older than 3.12 (Anaconda's default `python3` is often 3.11, which
Drake does not support). Override with `make setup PYTHON=/path/to/python3.13`.
From the repo root:

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

## RL environment

```python
import gymnasium as gym
import dap.envs                      # registers dap/PalmMeet-v0
env = gym.make("dap/PalmMeet-v0", obs_mode="state+depth", scenario="nominal")
obs, info = env.reset(seed=0)
```

No reward or training yet: pass `reward_fn=lambda info: ...`. See
docs/setup.md section 8.

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
results table (placeholder weights). Everything is computed from the
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

Sweeps resume where they left off (`--fresh` to start over); crashed trials
are logged with their traceback in `error` and counted by the analysis
(`--fail-fast` to stop instead). The trial loop calls
`ReplanningController.reset/step` in `dap/planner.py`, so the planner configs
don't run end to end until that's implemented; `--controller oracle|hold`
runs the scripted references now. The success tolerance and loss weights
are placeholders until we settle them.

## Working together

- Branch per workstream (`scene/partner-arm`, `meeting-point`, `planner`, ...),
  open a PR into `main`, one other person reviews.
- Keep `main` green: `pytest` passes before merging.
- Owners for each module are tracked in section 6 of the proposal doc.
