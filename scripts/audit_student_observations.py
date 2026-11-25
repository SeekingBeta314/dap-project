"""Replay frozen students to measure pose error at the camera timestamp.

The wrapper saves the estimate actually passed to the policy's action head.
Delayed ground truth is read only after that action has been computed. These
are descriptive diagnostics, with one record per evaluated scenario.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.experiment import Condition, ExperimentConfig
from dap.policy_eval import make_eval_env, run_policy_trial
from dap.student import load_student
from render_perception_pair import checked_row, frozen_input, read_rows


def rotation_error(predicted, truth):
    normal, fingers = predicted[6:9], predicted[9:12]
    R = np.column_stack((fingers, np.cross(normal, fingers), normal))
    return np.rad2deg(np.arccos(np.clip((np.trace(R.T @ truth) - 1.) / 2., -1., 1.)))


def one_trial(job):
    preset, row, checkpoint = job
    torch.set_num_threads(1)
    cfg = ExperimentConfig(name='student_pose_audit', scenario=preset)
    cond = Condition('student', 1.)
    env = make_eval_env(cfg, cond, 'pixels')
    model = load_student(checkpoint)
    samples = []
    predict = model.predict

    def record_estimate(depth, proprio):
        partner, action = predict(depth, proprio)
        estimate = partner[0].detach().cpu().numpy()
        _, (pos, velocity, frame) = env._partner_seen()
        samples.append((env.t, np.linalg.norm(estimate[:3] - pos),
                        np.linalg.norm(estimate[3:6] - velocity), rotation_error(estimate, frame)))
        return partner, action

    model.predict = record_estimate
    options = frozen_input(cfg, row['seed'], [row])
    result = run_policy_trial(cfg, cond, row['seed'], model, env=env, **options)
    checked_row(result, [row])
    stop = env.motion.t_end
    record = dict(preset=preset, seed=row['seed'], outcome=row['outcome'], success=result.success,
                  end_s=env.t, partner_stop_s=stop, pose_observations=len(samples),
                  success_before_stop=bool(result.success and result.t_event < stop),
                  partner_speed_at_event_mps=(np.linalg.norm(env.motion.v(result.t_event))
                                              if np.isfinite(result.t_event) else None))
    for window in ('all', 'last_200ms'):
        data = [s for s in samples if window == 'all' or s[0] >= env.t - .2 - 1e-8]
        for i, field in enumerate(('position_m', 'velocity_mps', 'orientation_deg'), 1):
            record[f'{window}_median_{field}'] = float(np.median([s[i] for s in data])) if data else None
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', required=True, choices=('nominal', 'wide'))
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--student', type=Path, default=ROOT / 'models/policies/student.pt')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    rows = sorted([r for r in read_rows(args.rows) if r['planner'] == 'student'
                   and r['error_kind'] != 'scenario'], key=lambda r: r['seed'])
    if not rows or len({r['seed'] for r in rows}) != len(rows):
        p.error('input must contain unique student records for one preset')
    jobs = [(args.preset, r, args.student) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(one_trial, jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        w = csv.DictWriter(stream, records[0], lineterminator='\n')
        w.writeheader()
        w.writerows(records)


if __name__ == '__main__':
    main()
