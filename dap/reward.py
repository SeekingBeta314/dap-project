"""Training reward for the dap teacher (docs/teacher_design.md, section 1).
For learning only: results are always the harness scores (dap.scoring).

    env = DapReward(gym.make("dap/PalmMeet-v0", obs_mode="privileged", bank=...))

Per 20 Hz step, from the env's ground-truth `info`:

    r_t = R_end + [gamma Phi(s_t+1) - Phi(s_t)] - w_a |a_t - a_t-1|^2

  R_end   at a contact or collision: `offset` - the trial's dap_loss (the
          evaluation's tiered loss, one definition). Timeouts are truncations
          with no terminal reward; the learner bootstraps their value.
  Phi     -k (rot_err / rot_scale + rho / dist_scale), where rho is our face
          center's distance to a point s(rot_err) = standoff min(1, rot_err /
          rot_tol) in front of the partner face along its normal: the servo's
          standoff-then-close approach as a state function. Potential-based
          (Ng, Harada & Russell 1999), with Phi = 0 at contact or collision so
          the shaping cannot change the ranking of terminated episodes
          (Grzes 2017).
  w_a     action-rate penalty on the action the env applied.
All weights and scales are Ours.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from dap.scoring import ROT_TOL_DEFAULT, LossWeights, dap_loss


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99  # must equal the learner's discount
    offset: float = 12.5  # puts a timeout (loss 10-15) at about 0
    k: float = 1.0  # shaping strength
    rot_scale: float = 45.0  # deg
    dist_scale: float = 0.25  # m
    standoff: float = 0.05  # m, the servo's standoff (dap.baselines.DiffIKServo)
    rot_tol: float = ROT_TOL_DEFAULT  # deg, where the approach point reaches the face
    # Sized against exploration noise: i.i.d. Gaussian actions (std 0.5) alone
    # give E|a_t - a_t-1|^2 = 2 * 7 * 0.25 = 3.5 per step, so 0.005 costs about
    # 0.9 over a 50-step episode; 0.05 cost about 8 (E, smoke run), a whole
    # success reward, and mostly penalized exploration. Ours.
    w_action_rate: float = 0.005
    loss: LossWeights = LossWeights()


def potential(info, cfg: RewardConfig):
    """Phi(s) <= 0 from the env's info (0 once the episode has ended)."""
    if info["event"] is not None:
        return 0.0
    rot = info["rot_err_deg"]
    s = cfg.standoff * min(1.0, rot / cfg.rot_tol)
    rho = np.hypot(info["in_plane_offset"], info["normal_gap"] - s)
    return -cfg.k * (rot / cfg.rot_scale + rho / cfg.dist_scale)


def outcome_reward(info, cfg: RewardConfig):
    """`offset` - dap_loss at a contact or collision, else 0. The env's info
    at the event is the event sample's geometry, as in dap.scoring.evaluate."""
    if info["event"] is None:
        return 0.0
    contact = info["event"] == "contact"
    nan = float("nan")
    row = {"outcome": np.array([info["event"]]), "success": np.array([info["success"]]),
           "pos_err": [info["in_plane_offset"] if contact else nan],
           "rot_err": [info["rot_err_deg"] if contact else nan],
           "time_err": [info["t"] - info["t_meet"] if contact else nan],
           "rel_speed": [nan], "min_palm_dist": [nan]}
    return cfg.offset - float(dap_loss(row, cfg.loss)[0])


class DapReward(gym.Wrapper):
    """Adds the training reward to a DapEnv (any obs mode). `info["reward_terms"]`
    holds the three parts for logging."""

    def __init__(self, env, cfg: RewardConfig = RewardConfig()):
        super().__init__(env)
        self.cfg = cfg

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._phi = potential(info, self.cfg)
        self._prev_a = np.zeros(7)  # the env's prev_action at reset
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        a = self.env.unwrapped.prev_action  # what the env applied (clipped, NaN -> 0)
        phi = potential(info, self.cfg)
        terms = {"outcome": outcome_reward(info, self.cfg),
                 "shaping": self.cfg.gamma * phi - self._phi,
                 "action_rate": -self.cfg.w_action_rate * float(np.sum((a - self._prev_a) ** 2))}
        self._phi, self._prev_a = phi, a.copy()
        info["reward_terms"] = terms
        return obs, sum(terms.values()), terminated, truncated, info
