"""Sample one randomized dap scenario: scene layout, a certified meeting, the
partner's motion, and our arm's start configuration.

    scenario = sample_scenario(rng, SCENARIO_PRESETS["nominal"])

Order (each step rejects the sample and the loop retries, up to
`max_attempts`; rejection reasons are counted in `Scenario.rejections`):

  1. Layout ~ ScenarioConfig.layout ranges; cheap base-distance gate.
  2. Nominal meeting point = midpoint between the bases at palm height, plus
     clipped Gaussian noise (people hand over at about half their separation,
     Nemlekar et al. 2019).
  3. Feasibility: one 14-joint IK finds both arms' configurations for a
     palm-to-palm dap near that point (dap.feasibility). This is the
     "a dap is possible" certificate.
  4. Partner motion toward the witness's partner palm pose
     (dap.partner.make_partner_motion).
  5. Our start configuration: Q_READY + noise, collision-free against the
     partner's start, and able to reach our side of the dap before t_meet at
     the joint velocity limits. "Our side" is the configuration closest to
     the start that puts our palm at the witness palm pose (single-arm IK,
     partner held at its witness configuration, collision-free).

All ranges and their sources are documented in docs/setup.md.
"""

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from dap.feasibility import MeetingWitness, base_distance_ok, check_feasible, reachable_in_time
from dap.ik import arm_ik
from dap.partner import PartnerMotion, PathConfig, make_partner_motion, sample_start_noise
from dap.scene import OURS, Q_READY, Scene, SceneLayout


@dataclass(frozen=True)
class LayoutRanges:
    distance: tuple = (1.2, 1.75)  # m, base to base along x
    lateral: float = 0.3  # m, +- partner base y offset
    yaw_deg: float = 30.0  # +- partner yaw away from facing us
    height: float = 0.15  # m, +- partner base height relative to ours


@dataclass(frozen=True)
class MeetingRanges:
    palm_height: float = 0.6  # m above the mean base height (current Q_READY palm height)
    sigma: float = 0.05  # m, per-axis Gaussian noise around the midpoint
    clip: float = 0.10  # m, per-axis clip on that noise
    box: float = 0.03  # m, IK may place our palm within +- box of the point


# Start-configuration noise tiers around Q_READY: (distribution, scale in rad).
# "small" is the robosuite / ManiSkill default (N(0, 0.02)); "medium" and
# "large" follow Isaac Lab's +-0.5 rad joint-offset resets.
START_NOISE = {
    "none": ("normal", 0.0),
    "small": ("normal", 0.02),
    "medium": ("uniform", 0.3),
    "large": ("uniform", 0.5),
}


@dataclass(frozen=True)
class ScenarioConfig:
    layout: LayoutRanges = LayoutRanges()
    meeting: MeetingRanges = MeetingRanges()
    path: PathConfig = PathConfig()
    start_noise: str = "medium"  # our arm
    partner_start_noise: str = "medium"  # partner arm (through its start IK)
    max_base_distance: float = 1.85  # m, analytic gate (feasibility limit ~1.88 m)
    feasibility_restarts: int = 5
    start_tries: int = 20  # our-start samples per scenario attempt
    start_clearance: float = 0.01  # m
    reach_speed_margin: float = 0.9
    joint_limit_margin_deg: float = 5.0
    max_attempts: int = 30


SCENARIO_PRESETS = {
    # Nominal geometry, only the partner path and start noise randomized.
    "fixed": ScenarioConfig(
        layout=LayoutRanges(distance=(1.4, 1.4), lateral=0.0, yaw_deg=0.0, height=0.0),
        start_noise="small", partner_start_noise="small",
    ),
    # Ranges that pass the feasibility check (almost) always.
    "nominal": ScenarioConfig(),
    # Harder: wider layouts that rely on rejection sampling, larger start
    # noise, and mid-course target switches.
    "wide": ScenarioConfig(
        layout=LayoutRanges(distance=(1.0, 1.9), lateral=0.45, yaw_deg=45.0, height=0.25),
        meeting=MeetingRanges(sigma=0.08, clip=0.15),
        path=PathConfig(switch_prob=0.3),
        start_noise="large", partner_start_noise="large",
    ),
}


class ScenarioError(RuntimeError):
    pass


@dataclass
class Scenario:
    layout: SceneLayout
    scene: Scene
    witness: MeetingWitness  # certifies that a dap is possible
    partner: PartnerMotion  # ground truth
    q_ours0: np.ndarray  # our arm's start configuration (7,)
    q_ours_meet: np.ndarray  # our configuration closest to q_ours0 that achieves the dap
    meet_prior: np.ndarray  # nominal meeting point; fine for the controller to know
    attempts: int
    rejections: dict = field(default_factory=dict)
    sample_s: float = float("nan")

    @property
    def t_meet(self):
        return self.partner.t_meet


def sample_layout(rng, r: LayoutRanges):
    return SceneLayout(
        distance=float(rng.uniform(*r.distance)),
        lateral=float(rng.uniform(-r.lateral, r.lateral)),
        yaw_deg=float(rng.uniform(-r.yaw_deg, r.yaw_deg)),
        height=float(rng.uniform(-r.height, r.height)),
    )


def nominal_meeting_point(layout: SceneLayout, palm_height):
    return np.array([layout.distance / 2, layout.lateral / 2, palm_height + layout.height / 2])


def sample_scenario(rng, cfg: ScenarioConfig = ScenarioConfig(), speed_scale=1.0,
                    camera=None) -> Scenario:
    """Draw scenarios until one passes every check. `camera` (a
    HeadCameraConfig) is added to the scene if given."""
    t0 = time.perf_counter()
    rejections = Counter()
    margin = np.deg2rad(cfg.joint_limit_margin_deg)
    for attempt in range(1, cfg.max_attempts + 1):
        layout = sample_layout(rng, cfg.layout)
        if not base_distance_ok(layout, cfg.max_base_distance):
            rejections["base_distance"] += 1
            continue
        scene = Scene(layout, camera=camera)
        m = cfg.meeting
        prior = nominal_meeting_point(layout, m.palm_height)
        target = prior + np.clip(rng.normal(0, m.sigma, 3), -m.clip, m.clip)
        witness = check_feasible(scene, rng, target, m.box, cfg.feasibility_restarts)
        if witness is None:
            rejections["no_meeting"] += 1
            continue

        partner, why = make_partner_motion(
            scene, witness.X_W_partner, rng, cfg.path, speed_scale,
            START_NOISE[cfg.partner_start_noise],
        )
        if partner is None:
            rejections[why] += 1
            continue

        lo = scene.q_lower[scene.idx[OURS]] + margin
        hi = scene.q_upper[scene.idx[OURS]] - margin
        v_ours = scene.v_max[scene.idx[OURS]]
        q_ours0 = None
        for _ in range(cfg.start_tries):
            q = np.clip(Q_READY + sample_start_noise(rng, *START_NOISE[cfg.start_noise]), lo, hi)
            scene.set_partner(partner.q(0.0))
            scene.set_ours(q)
            if scene.clearance() < cfg.start_clearance:
                rejections["start_collision"] += 1
                continue
            q_meet = arm_ik(scene, OURS, witness.X_W_ours, q, q, 0.001, 1.0,
                            bounds=(lo, hi), min_distance=0.005, q_full=witness.q)
            if q_meet is None:
                q_meet = witness.q_ours
            if not reachable_in_time(q, q_meet, v_ours, partner.t_meet,
                                     cfg.reach_speed_margin):
                rejections["start_too_far"] += 1
                continue
            q_ours0 = q
            break
        if q_ours0 is None:
            continue

        scene.set_ours(q_ours0)
        scene.set_partner(partner.q(0.0))
        return Scenario(
            layout=layout, scene=scene, witness=witness, partner=partner,
            q_ours0=q_ours0, q_ours_meet=q_meet, meet_prior=prior, attempts=attempt,
            rejections=dict(rejections), sample_s=time.perf_counter() - t0,
        )
    raise ScenarioError(
        f"no feasible scenario in {cfg.max_attempts} attempts; rejections: {dict(rejections)}"
    )

