# Dap: Angle and Alignment Matching for Robotic Hands

6.4210 final project — Josh Nix, Oliver Wang, Samuel Zhou.

Two simulated iiwa arms face each other in Drake. The partner arm's hand
approaches along a randomized 3D path; our arm only sees noisy pose
measurements and has to meet it palm to palm, on time, palms parallel, after
passing a set of checkpoints, and then clasp.

**Proposal (living draft):** https://claude.ai/artifact/QUBj973spHWSZKigTEf1Cz

## Layout

| Path | What | Status |
| --- | --- | --- |
| `models/dap_hand.sdf` | Flat palm + hinged finger plate (`package://dap/...`) | done |
| `dap/scene.py` | Two iiwa7s 1.4 m apart, hands welded, palms facing | done |
| `dap/partner.py` | Randomized partner palm path + noisy pose sensor | path done; arm not driven yet |
| `dap/estimator.py` | Constant-acceleration Kalman filter + palm-normal filter | done (first pass) |
| `dap/meeting.py` | Meeting time + palm pose via IK sweep | TODO |
| `dap/planner.py` | `KinematicTrajectoryOptimization` + replanning loop | TODO |
| `dap/clasp.py` | Compliant follow + finger close | TODO |
| `dap/metrics.py` | Per-trial result record | done |
| `scripts/show_scene.py` | View the scene + a partner path in Meshcat | done |
| `tests/` | Scene geometry and estimator sanity checks | passing |
| `notebooks/` | Scratch notebooks (clear outputs before committing) | |
| `results/` | Experiment outputs (git-ignored) | |

Hand frame convention: palm `+z` is the palm normal (the face that meets the
partner), `+x` points to the finger hinge.

## Setup

Same environment as the psets (Drake 1.57, `manipulation`, Python 3.13):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the course venv, you can use it directly.

> Drake downloads its robot models on first use and that download breaks if the
> venv path contains a space (e.g. `MIT Stuff/`). Keep the venv in a path
> without spaces if you hit `returncode == 512`.

## Run

```bash
python scripts/show_scene.py --seed 0
python -m pytest -q
```

## Working together

- Branch per workstream (`scene/partner-arm`, `meeting-point`, `planner`, ...),
  open a PR into `main`, one other person reviews.
- Keep `main` green: `pytest` passes before merging.
- Owners for each module are tracked in section 6 of the proposal doc.
