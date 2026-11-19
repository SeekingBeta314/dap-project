"""Scenario bank: scenarios sampled once, saved, and rebuilt cheaply.

Sampling a scenario is most of an RL reset (hundreds of IK solves for the
partner's motion; about 1 s on average with the camera, several seconds at
high partner speeds). A bank stores what those solves produced, so a reset
only rebuilds the scene and the partner's spline (tens of ms):

    python scripts/build_bank.py --out banks/train_nominal.pkl --n 20000
    env = gym.make("dap/PalmMeet-v0", obs_mode="pixels", bank="banks/train_nominal.pkl")

Banks are for TRAINING. They are sampled from dap.experiment.train_rngs,
whose streams are disjoint from the evaluation seeds (trial_rngs), so a
policy is never trained on a scenario it is evaluated on. Evaluation
(dap.policy_eval, scripts/run_experiment.py) always samples fresh from the
experiment seeds.

A rebuilt scenario is identical to the sampled one: same layout, same IK
knots, so the same spline, partner motion and our start (tests/test_bank.py).
Files are Python pickles of plain numbers and arrays; only load banks you
built.
"""

import pickle
import time
from dataclasses import asdict

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from dap.experiment import train_rngs
from dap.feasibility import MeetingWitness
from dap.partner import PalmPathReference, PartnerMotion, joint_spline
from dap.scenario import SCENARIO_PRESETS, Scenario, ScenarioError, sample_scenario
from dap.scene import Scene, SceneLayout

BANK_VERSION = 1


def _X(X: RigidTransform):
    return X.GetAsMatrix4()


def scenario_spec(sc: Scenario) -> dict:
    """Everything needed to rebuild `sc`, as plain numbers and arrays."""
    ref = sc.partner.reference.args
    sw = ref["switch"]
    times, Q = sc.partner.knots
    w = sc.witness
    return dict(
        layout=asdict(sc.layout),
        reference=dict(
            p_start=ref["p_start"], X_meet=_X(ref["X_meet"]), R_start=ref["R_start"].matrix(),
            peak_speed=ref["peak_speed"], bulge=ref["bulge"], approach_len=ref["approach_len"],
            overshoot=ref["overshoot"],
            switch=None if sw is None else (np.asarray(sw[0], float), float(sw[1]), float(sw[2])),
        ),
        knot_times=np.asarray(times, float), knots=np.asarray(Q, float),
        dense_dt=float(sc.partner.dense_dt), target_peak_speed=float(sc.target_peak_speed),
        witness=dict(q=w.q, q_ours=w.q_ours, q_partner=w.q_partner, X_W_ours=_X(w.X_W_ours),
                     X_W_partner=_X(w.X_W_partner), p_contact=w.p_contact, n_ours=w.n_ours,
                     solve_ms=w.solve_ms, attempts=w.attempts, quality=w.quality),
        q_ours0=sc.q_ours0, q_ours_meet=sc.q_ours_meet, meet_prior=sc.meet_prior,
        attempts=sc.attempts, rejections=dict(sc.rejections), sample_s=sc.sample_s,
        reachable_in_time=bool(sc.reachable_in_time),
        partner_joint_speed_ratio=float(sc.partner_joint_speed_ratio),
    )


def scenario_from_spec(spec: dict, camera=None) -> Scenario:
    """Rebuild a Scenario (a fresh Scene, with the head camera if given)."""
    layout = SceneLayout(**spec["layout"])
    scene = Scene(layout, camera=camera)
    r = spec["reference"]
    ref = PalmPathReference(
        r["p_start"], RigidTransform(r["X_meet"]), RotationMatrix(r["R_start"]),
        r["peak_speed"], r["bulge"], r["approach_len"], r["overshoot"], r["switch"])
    times, Q = spec["knot_times"], spec["knots"]
    partner = PartnerMotion(scene, joint_spline(times, Q), ref, spec["dense_dt"],
                            spec["target_peak_speed"])
    partner.knots = (times, Q)
    w = dict(spec["witness"])
    w["X_W_ours"], w["X_W_partner"] = RigidTransform(w["X_W_ours"]), RigidTransform(w["X_W_partner"])
    sc = Scenario(
        layout=layout, scene=scene, witness=MeetingWitness(**w), partner=partner,
        q_ours0=np.array(spec["q_ours0"]), q_ours_meet=np.array(spec["q_ours_meet"]),
        meet_prior=np.array(spec["meet_prior"]), attempts=spec["attempts"],
        rejections=dict(spec["rejections"]), sample_s=spec["sample_s"],
        target_peak_speed=spec["target_peak_speed"],
        reachable_in_time=spec["reachable_in_time"],
        partner_joint_speed_ratio=spec["partner_joint_speed_ratio"],
    )
    scene.set_ours(sc.q_ours0)
    scene.set_partner(partner.q(0.0))
    return sc


def sample_spec(seed, preset):
    """One training scenario (train_rngs(seed)), as a spec, or None if the
    sampler gave up (recorded by the caller)."""
    try:
        sc = sample_scenario(train_rngs(seed)["scenario"], SCENARIO_PRESETS[preset])
    except ScenarioError:
        return None
    spec = scenario_spec(sc)
    spec["seed"] = int(seed)
    return spec


def build_bank(n, preset="nominal", first_seed=0, workers=1, progress=None):
    """Sample training seeds first_seed, first_seed + 1, ... until n
    scenarios exist. Returns the bank dict (see `save_bank`)."""
    t0 = time.perf_counter()
    specs, failed = [], []
    seed = first_seed
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(workers)
    try:
        while len(specs) < n:
            batch = list(range(seed, seed + (n - len(specs))))
            seed += len(batch)
            if workers > 1:
                got = list(pool.map(sample_spec, batch, [preset] * len(batch), chunksize=4))
            else:
                got = [sample_spec(s, preset) for s in batch]
            for s, g in zip(batch, got):
                (specs.append(g) if g is not None else failed.append(s))
            if progress:
                progress(len(specs), n)
    finally:
        if workers > 1:
            pool.shutdown()
    return dict(version=BANK_VERSION, preset=preset, config=repr(SCENARIO_PRESETS[preset]),
                first_seed=first_seed, failed_seeds=failed, specs=specs,
                build_s=time.perf_counter() - t0)


def save_bank(bank, path):
    with open(path, "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)


class ScenarioBank:
    """A saved bank. `get(i, camera)` rebuilds scenario i."""

    def __init__(self, path_or_bank):
        if isinstance(path_or_bank, dict):
            bank = path_or_bank
        else:
            with open(path_or_bank, "rb") as f:
                bank = pickle.load(f)
        if bank.get("version") != BANK_VERSION:
            raise ValueError(f"bank version {bank.get('version')} != {BANK_VERSION}; rebuild it")
        if bank["config"] != repr(SCENARIO_PRESETS[bank["preset"]]):
            raise ValueError(f"bank was built with a different {bank['preset']!r} preset; "
                             "rebuild it")
        self.preset = bank["preset"]
        self.specs = bank["specs"]
        if not self.specs:
            raise ValueError("empty scenario bank")

    def __len__(self):
        return len(self.specs)

    def get(self, i, camera=None) -> Scenario:
        return scenario_from_spec(self.specs[i], camera)
