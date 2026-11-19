"""Per-trial result record. Every number here is measured by the harness
from ground truth (dap.scoring.evaluate on the logged trial, the scenario,
and wall-clock timing around controller calls); nothing is reported by the
controller itself."""

from dataclasses import asdict, dataclass

import numpy as np

NAN = float("nan")


@dataclass
class TrialResult:
    seed: int
    planner: str
    noise_scale: float

    # --- outcome (dap.scoring.evaluate) -----------------------------------
    outcome: str = "timeout"  # "contact" | "collision" | "timeout" | "error"
    success: bool = False  # contact with palms antiparallel (rot_err <= tolerance)
    contact: bool = False
    collision: bool = False
    t_event: float = NAN  # s, time of the contact or collision that ended the trial
    time_err: float = NAN  # s, contact time minus the partner's TRUE meeting time
    gap: float = NAN  # m, our face center along the partner normal at contact
    pos_err: float = NAN  # m, palm-center offset in the partner palm plane at contact
    rot_err: float = NAN  # deg, full relative-rotation error vs. antiparallel at contact
    angle_err: float = NAN  # deg, palms not parallel (between normals)
    finger_err: float = NAN  # deg, fingers not opposite (about the normal)
    rel_speed: float = NAN  # m/s, palm-palm relative speed at contact
    min_palm_dist: float = NAN  # m, closest face-center distance before the event
    closest_rot_err: float = NAN  # deg, rot_err at that closest approach
    min_arm_dist: float = NAN  # m, closest our-arm/hand vs. partner-arm/hand (palm pair excluded)
    vmax_frac: float = NAN  # fraction of steps with some joint at >= 95% of its speed limit

    # --- scenario covariates --------------------------------------------
    target_peak_speed: float = NAN  # m/s, partner speed drawn for this scenario
    partner_peak_speed: float = NAN  # m/s, realized partner palm peak speed
    partner_speed_at_meet: float = NAN  # m/s, partner palm speed at its meeting time
    t_meet_true: float = NAN  # s, when the partner palm reaches its meeting pose
    reachable_in_time: bool = True  # our arm could reach the dap by t_meet (speed limits)
    partner_joint_speed_ratio: float = NAN  # max partner |qdot| / v_max (<= 1)
    layout_distance: float = NAN
    layout_lateral: float = NAN
    layout_yaw_deg: float = NAN
    layout_height: float = NAN
    scenario_attempts: int = 0  # scenario samples drawn before one passed every check

    # --- harness measurements --------------------------------------------
    n_steps: int = 0
    step_ms_mean: float = NAN  # wall time of controller.step, measured by the harness
    step_ms_p95: float = NAN
    step_ms_max: float = NAN
    n_bad_commands: int = 0  # non-finite / wrongly shaped commands (arm held instead)
    wall_time_s: float = NAN
    error: str = ""  # exception text if the trial crashed
    # Who crashed: "scenario" (no scenario could be sampled for this seed:
    # the same for every condition), "controller" (an exception inside the
    # controller: a failure of the method), or "harness" (a bug in our code).
    error_kind: str = ""

    def to_dict(self):
        d = asdict(self)
        # NaN and inf are not valid JSON.
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in d.items()}
