# Dap: Angle and Alignment Matching for Robotic Hands

6.4210 final project — Josh Nix, Oliver Wang, Samuel Zhou.

Two simulated iiwa arms face each other in Drake, in a layout randomized per
trial. The partner arm's hand approaches along a randomized, human-like path;
our arm sees it through a head-mounted RGB-D camera (perception pipeline
still to come; a noisy pose sensor stands in for it) and has to meet it palm
to palm, on time, palms parallel with fingers opposite, and then clasp.

**Problem setup, every parameter and its source:** [docs/setup.md](docs/setup.md).
Planned perception pipeline: [docs/perception_pipeline.md](docs/perception_pipeline.md).

**Proposal (living draft):** https://claude.ai/artifact/QUBj973spHWSZKigTEf1Cz

## Layout

| Path | What | Status |
| --- | --- | --- |
| `models/dap_hand.sdf` | Flat palm + hinged finger plate (`package://dap/...`) | done |
| `dap/scene.py` | Two iiwa7s on pedestals, partner pose from a `SceneLayout`; `Scene` helpers (posing, palms, distances, camera) | done |
| `dap/scenario.py` | Per-trial sampler: layout, meeting point, feasibility, partner motion, start configs; presets `fixed`/`nominal`/`wide` | done |
| `dap/feasibility.py` | Two-arm IK check that a palm-to-palm dap exists (+ reach-in-time check) | done |
| `dap/partner.py` | Min-jerk Bezier palm path (overshoot, optional target switch), tracked by the partner arm via IK; noisy pose sensor | done |
| `dap/ik.py` | Single-arm IK helper | done |
| `dap/camera.py` | D435-like head camera (848x480, 58 deg vFOV, 30 fps) + frame-rate/latency buffer | done |
| `dap/depth_noise.py` | D435-style depth noise (disparity noise, jitter, fixed pattern, dropout, shadow, holes) | done |
| `dap/envs/` | Gymnasium env `dap/PalmMeet-v0` (state or state+depth obs; reward hook only) | done; no training |
| `dap/estimator.py` | Constant-acceleration Kalman filter + palm-normal filter | done (first pass) |
| `dap/meeting.py` | Meeting time + palm pose via IK sweep | TODO |
| `dap/planner.py` | `KinematicTrajectoryOptimization` + replanning loop | TODO |
| `dap/clasp.py` | Compliant follow + finger close | TODO |
| `dap/metrics.py` | Per-trial result record | done |
| `dap/checkpoints.py` | Random ordered checkpoints between start and meeting | done |
| `dap/experiment.py` | Sweep conditions + named configs (`smoke`, H1–H3) | done |
| `dap/trial.py` | Kinematic trial loop (partner arm driven, arm distance + collisions logged) + scoring | done; needs the controller |
| `scripts/show_scene.py` | View a sampled scenario in Meshcat, animate it, or save head-camera depth | done |
| `scripts/run_experiment.py` | Run a config, append to `results/<name>/trials.jsonl` | done |
| `scripts/analyze.py` | Summary table + success/error figures | done |
| `tests/` | Scene, camera, depth noise, feasibility, scenario, env (gymnasium `check_env`), scoring | passing |
| `docs/` | Setup (with sources) and the planned perception pipeline | |
| `notebooks/` | Scratch notebooks (clear outputs before committing) | |
| `results/` | Experiment outputs (git-ignored) | |

Hand frame convention: palm `+z` is the palm normal (the face that meets the
partner), `+x` points to the finger hinge.

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

Each named config in `dap/experiment.py` is a grid of conditions (planner,
partner speed, noise, checkpoint count, velocity matching) times seeds, on a
scenario preset (`ExperimentConfig.scenario`, default `nominal`). Every
condition reuses the same seeds, and a seed fixes the scenario, sensor noise,
and checkpoints, so comparisons across planners and noise levels are paired.

| Config | Tests | Sweeps |
| --- | --- | --- |
| `smoke` | Plumbing | replan only, 3 seeds |
| `replan_vs_once` | H1 | planner x speed x noise x checkpoints (0, 2), 50 seeds |
| `speed_limits` | H2 | speed (0.5–3x) x checkpoints (0, 2, 4), 30 seeds |
| `velocity_match` | H3 | velocity matching on/off x speed x noise, 50 seeds |

```bash
python scripts/run_experiment.py smoke --dry-run          # list conditions, run nothing
python scripts/run_experiment.py replan_vs_once --workers 4
python scripts/analyze.py results/replan_vs_once --x speed_scale --hue planner --facet n_checkpoints
```

Sweeps resume where they left off (`--fresh` to start over); crashed trials
are logged with their traceback in `error` (`--fail-fast` to stop instead).
The trial loop calls `ReplanningController.reset/step` in `dap/planner.py`, so
nothing runs end to end until that's implemented. Success tolerances in
`ExperimentConfig` are placeholders until we settle them (proposal section 7).

## Working together

- Branch per workstream (`scene/partner-arm`, `meeting-point`, `planner`, ...),
  open a PR into `main`, one other person reviews.
- Keep `main` green: `pytest` passes before merging.
- Owners for each module are tracked in section 6 of the proposal doc.
