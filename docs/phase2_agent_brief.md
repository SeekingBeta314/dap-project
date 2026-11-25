# Brief for the Phase 2 agent

You are picking up Phase 2 of an MIT 6.4210 (Robotic Manipulation) project in
`/Users/wenfe1/Programs/dap-project` (GitHub: SeekingBeta314/dap-project,
branch `main`). `AGENTS.md` is loaded automatically: follow it. This brief
says what exists, what to do, and the rules that keep the result
trustworthy. The user is strict about correctness, honest reporting and
exact citations, and prefers short, plain explanations.

## The project in one paragraph

In Drake, our KUKA iiwa must meet a partner iiwa's hand palm to palm
(antiparallel, fingers opposite, within 15°) while the partner reaches along
a randomized, human-like path at a randomized speed. Our arm sees the
partner only through a head-mounted, D435-style depth camera (30 fps, one
frame of latency, modelled noise). Everything is scored from the
simulator's ground truth. Phase 1 built the setup, the baselines and the
evaluation tooling, and found that **perception is the bottleneck**: the
same servo succeeds in 10–11 of 12 validation scenarios given the true palm
pose, and in 0–3 of 12 when it must find the palm in the depth images
with a classical pipeline. Phase 2 asks whether a depth policy distilled
from a privileged teacher does better. The trajectory planner is no longer
part of the project.

## Read first, in this order

1. `docs/phase2_plan.md`: **the plan**. Hypothesis H1, questions Q2–Q3,
   method, work items 0–10 with "done when" criteria, rules, decisions,
   risks. Work from it.
2. `README.md`, then `docs/setup.md` sections 1b (baselines), 8 (RL env),
   9 (experiments and **seed splits**), 10 and 11.
3. `docs/perception_pipeline.md`: the classical baseline `servo_depth`.
4. `reports/Dap task research landscape.md`: the literature review behind
   the plan. Verified bibliographic details are in
   `research_notes/Dap task research landscape/` and in the reference list
   at the end of `docs/setup.md`.

Key code: `dap/envs/dap_env.py` (the `pixels`, `privileged` and `state`
modes; `reward_fn` hook), `dap/bank.py`, `dap/policy_eval.py` (scores a
policy exactly like a controller), `dap/experiment.py` (configs and seed
splits), `dap/baselines.py`, `dap/perception.py`, `dap/scoring.py`.

## What to do

Follow the work items in `docs/phase2_plan.md` section 4, in order.

- **Ask the user before you start:**
  - item 0: the success tolerance, whether success needs timing, and the
    loss weights;
  - item 1: delete the planner stubs (`ReplanningController`,
    `dap/meeting.py`) or park them;
  - the default training preset (plan section 6 proposes `nominal` only).
- **Ask before anything costly or outward-facing:**
  - installing torch / stable-baselines3 (item 3);
  - building the full bank (about 1.5 h, item 4);
  - any training run longer than a few minutes;
  - **the evaluation run (item 9), which happens once**;
  - editing the Overleaf report or the shared proposal doc;
  - any commit or push.
- **A good first session:** settle item 0 with the user, then do items 1
  and 2. Item 2 promotes the scratch diagnostic "servo fed the true pose at
  the camera's capture times" to a named reference `servo_gt_cam` in
  `dap/baselines.py`; it scored 8/12 nominal and 9/12 wide on validation
  seeds 0–11. Both items are small, testable and need no new dependencies.
- **Before you write the teacher or student:** propose a concrete design
  to the user (reward terms, algorithm, network, budget), citing the plan
  and the review.

## Rules that are not negotiable

- **Seed splits** (`dap/experiment.py`, `docs/setup.md` section 9):
  - Training uses `train_rngs` (the bank, fresh env resets).
  - Everything decided by looking at results uses **validation** seeds:
    reward terms, hyperparameters, architecture, checkpoints, early
    stopping, sanity checks. Use `ExperimentConfig(split="val")` and
    `env.reset(options={"trial_seed": s, "split": "val"})`.
  - **Evaluation** seeds (1000 and up) are touched once per frozen method,
    in the final run. Seeds below 1000 are spent; the config raises an
    error if you use them.
  - Never look at evaluation results while developing. If you need a
    number, run it on validation seeds.
- **The student sees only `pixels`.** Privileged state is for the teacher,
  the critic and the DAgger labels. The teacher is latency-matched (the
  default `privileged_delay=True`); do not give it the partner's endpoint,
  speed profile or switch time.
- **Score only through the harness:** `run_trial` / `run_policy_trial`.
  Nothing a policy reports is a result.
- **The setup is frozen.** Do not change the camera, the RL env's
  interface, the arm limits (no teleporting) or the scenario sampler.
  Adding a mode or an option is fine; redesigning is not.
- **Every new simulation or training parameter** gets a source labelled
  **V** (verified in the paper, datasheet or code), **E** (measured here)
  or **Ours**. Check citations against the source; never cite from memory.
- **Make surgical changes** (`AGENTS.md`), and don't propose features
  beyond the plan unless asked.
- **Regression lock:** `tests/test_regression.py` fingerprints trial and env
  behavior. Re-record it (`REGRESSION_WRITE=1 .venv/bin/python -m pytest
  tests/test_regression.py`) only for an intended behavior change, and say
  which keys changed.
- **Report honestly:** say what is verified and what is not, name the seed
  split behind every number, and never present a check as a result.

## Practicalities

- **Python:** use `.venv/bin/python` (Python 3.13, Drake 1.58). Scratch
  scripts outside the package need `PYTHONPATH=.`.
- **Tests:** `make test` runs about 100 tests in about 4 minutes; run it
  in the background. pytest-xdist is not installed. Lint with
  `python3 -m pyflakes dap scripts tests`; the unused `import dap.envs`
  warnings are intentional.
- **Speed:** a `pixels` policy step takes about 8 ms and a `privileged`
  step about 1 ms per core. Without a bank, a reset samples a fresh
  scenario in about 1 s.
- **Scratch work** belongs in your scratchpad, not the repo.
- **Running baselines:** `scripts/run_experiment.py <config> --controller
  <name>` runs a baseline. `first_look` and `first_look_wide` are the
  validation-seed configs. The planner configs (`replan_vs_once`,
  `speed_limits`) are due to be retired in item 1.
- **Git:** commit or push only when the user asks. "Merge into main" means
  commit, merge and push `main` without asking at each step. End commit
  messages with the attribution line your environment specifies.
- **External documents.** The user's Overleaf report ("6.4210 Project
  Report", `conference_101719.tex`) and the shared proposal doc (linked in
  the README) were updated to the Phase 2 framing on 2026-11-22
  (`docs/report_fixes.md`, section C). Update them only when the user asks.
