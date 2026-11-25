"""Export a paired trial as silent, real-time MP4 and a terminal still for the paper.

Pass the two archived trial rows (JSONL or gzip) so the replay is checked
against the published evaluation before anything is written.
"""

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.baselines import SERVO_DEPTH_CAMERA, servo_depth_factory
from dap.camera import HeadCameraConfig
from dap.experiment import Condition, ExperimentConfig
from dap.policy_eval import make_eval_env, run_policy_trial
from dap.scene import Scene
from dap.student import load_student
from dap.trial import run_trial

WIDTH, HEIGHT, FPS = 1280, 720, 25
FONT_DIR = Path(get_data_path()) / 'fonts' / 'ttf'
FONTS = {size: ImageFont.truetype(str(FONT_DIR / 'DejaVuSans.ttf'), size)
         for size in (17, 20, 24, 31)}
BG, FG, MUTED = '#152426', '#fafafa', '#bdcecc'


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checked_row(result, rows):
    matches = [row for row in rows if row['seed'] == result.seed and row['planner'] == result.planner]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one archived row for {result.planner}, seed {result.seed}')
    archived, replay = matches[0], result.to_dict()
    for key in ('outcome', 'success', 'rot_err', 'time_err', 't_event', 't_meet_true',
                'layout_distance', 'layout_lateral', 'layout_yaw_deg', 'partner_peak_speed'):
        a, b = archived[key], replay[key]
        equal = np.isclose(a, b, atol=1e-5, rtol=1e-5) if isinstance(a, float) and b is not None else a == b
        if not equal:
            raise ValueError(f'{result.planner} seed {result.seed}: {key} differs ({a} vs {b})')
    if result.error_kind:
        raise ValueError(result.error)


def frozen_input(cfg, seed, rows):
    if cfg.split != 'eval':
        return {}
    from dap.eval_scenarios import load_frozen_scenario
    frozen = load_frozen_scenario(cfg, seed)
    hashes = {r.get('scenario_sha256') for r in rows if r['seed'] == seed}
    if hashes != {frozen.sha256}:
        raise ValueError(f'Seed {seed} archive does not match the published scenario')
    return {'frozen_scenario': frozen}


def run_pair(cfg, seed, student, rows):
    # Attach read-only recorders to the actual controller interfaces. Each side
    # sees its own occlusion pattern after its first different arm command.
    learned, classical = Condition('student', 1.), Condition('servo_depth', 1.)
    env = make_eval_env(cfg, learned, 'pixels')
    policy = load_student(student)
    student_depth, classical_depth = [], []

    class RecordStudent:
        obs_mode = 'pixels'

        def __call__(self, obs):
            student_depth.append((env.t, obs['depth'][-1].copy()))
            return policy(obs)

    def record_classical(*args):
        control = servo_depth_factory(*args)
        observe, step = control.observe, control.step
        frame = None

        def receive(captured, depth, q):
            nonlocal frame
            frame = depth.copy()
            observe(captured, depth, q)

        def move(t, q, estimator):
            nonlocal frame
            if frame is not None:
                classical_depth.append((t, frame))
                frame = None
            return step(t, q, estimator)

        control.observe, control.step = receive, move
        return control

    options = frozen_input(cfg, seed, rows)
    a, la = run_policy_trial(cfg, learned, seed, RecordStudent(), env=env, return_log=True, **options)
    b, lb = run_trial(cfg, classical, seed, camera=SERVO_DEPTH_CAMERA,
                      controller_factory=record_classical, return_log=True, **options)
    for result in (a, b):
        checked_row(result, rows)
    return env, ((a, la, student_depth), (b, lb, classical_depth))


def text(draw, xy, label, size=20, fill=FG, **kw):
    draw.text(xy, label, font=FONTS[size], fill=fill, **kw)


def outcome(r):
    if r.success:
        return f'Aligned contact   |   {r.rot_err:.1f}° error'
    if r.contact:
        return f'Misaligned contact   |   {r.rot_err:.1f}° error'
    return {'collision': 'Collision', 'timeout': 'Timeout'}[r.outcome]


def comparison_frame(t, pair, env, side, preset, seed):
    image = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (24, 10), 'Meeting a moving palm from depth', 31)
    text(draw, (1256, 19), f'{preset.capitalize()}   /   seed {seed}   /   1× playback', 20,
         anchor='ra', fill=MUTED)
    for i, (r, log, depth_frames) in enumerate(pair):
        x = 24 + i * 630
        text(draw, (x, 67), ('Depth student', 'Classical depth servo')[i], 24)
        k = int(np.clip(np.searchsorted(log.t, t + 1e-8, side='right') - 1, 0, len(log.t) - 1))
        side.set_ours(log.q[k])
        side.set_partner(env.motion.q(log.t[k]))
        image.paste(Image.fromarray(side.render_rgb()), (x, 104))
        available = [f for stamp, f in depth_frames if stamp <= log.t[k] + 1e-8]
        if available and np.any(available[-1] > 0):
            depth = available[-1]
            # Darker = nearer. No return is shown in white, on a fixed metric scale.
            gray = np.where(depth > 0, 35 + 175 * np.clip(depth / 3, 0, 1), 255)
            raw = Image.fromarray(gray.astype('uint8')).resize((318, 180), Image.Resampling.NEAREST)
            image.paste(raw.convert('RGB'), (x, 457))
        else:
            draw.rectangle((x, 457, x + 317, 636), fill='#48565a')
            text(draw, (x + 13, 529), 'Awaiting depth frame', 20)
        text(draw, (x + 334, 469), 'Depth input', 20)
        text(draw, (x + 334, 503), 'Darker is nearer.', 17, MUTED)
        text(draw, (x + 334, 531), 'White is no return.', 17, MUTED)
        finished = k == len(log.t) - 1
        text(draw, (x, 653), outcome(r) if finished else 'Approaching', 24)
        text(draw, (x, 686), f'{"Stopped at" if finished else "Simulation time"} {log.t[k]:.2f} s',
             17, MUTED)
    return image


def select_case(cfg, rows, case):
    from dap.eval_scenarios import load_frozen_scenario
    for row in sorted((r for r in rows if r['planner'] == 'student'), key=lambda r: r['seed']):
        if row['error_kind']:
            continue
        if case == 'target-switch':
            scenario = load_frozen_scenario(cfg, row['seed'])
            matches = scenario.spec['reference']['switch'] is not None
        else:
            matches = row['success'] == (case == 'success')
        if matches:
            return row['seed']
    raise ValueError(f'no {case} among the archived {cfg.scenario} student trials')


def render(args):
    cfg = ExperimentConfig(name='replay', scenario=args.preset)
    rows = read_rows(args.rows)
    seed = select_case(cfg, rows, args.case) if args.case else args.seed
    env, pair = run_pair(cfg, seed, args.student, rows)
    cam = HeadCameraConfig(width=602, height=335, rgb=True, position=(0.56, -1.65, 0.98),
                           target=(0.62, 0, 0.49))
    side = Scene(env.scenario.layout, camera=cam)
    end = max(log.t[-1] for _, log, _ in pair)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        path = Path(temp) / 'pair.mp4'
        encoder = imageio_ffmpeg.write_frames(str(path), (WIDTH, HEIGHT), fps=FPS,
            codec='libx264', output_params=['-crf', '19', '-movflags', '+faststart'], macro_block_size=1)
        encoder.send(None)
        try:
            for k in range(int(np.ceil(end * FPS)) + 1):
                frame = comparison_frame(k / FPS, pair, env, side, args.preset, seed)
                encoder.send(frame.tobytes())
            for _ in range(FPS):  # one labeled terminal frame held for reading
                encoder.send(frame.tobytes())
        finally:
            encoder.close()
        frame.save(args.out.with_suffix('.png'))
        path.replace(args.out)
    for r, _, _ in pair:
        print(f'{args.preset} seed {seed}, {r.planner}: {outcome(r)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=['nominal', 'wide'], required=True)
    p.add_argument('--rows', type=Path, required=True)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument('--seed', type=int)
    which.add_argument('--case', choices=('success', 'failure', 'target-switch'),
                       help='use the first matching student trial in seed order')
    p.add_argument('--student', type=Path, default=ROOT / 'models/policies/student.pt')
    p.add_argument('--out', type=Path, required=True)
    render(p.parse_args())
