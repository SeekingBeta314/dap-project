"""Gymnasium registration. `import dap.envs` makes "dap/PalmMeet-v0"
available to `gymnasium.make`."""

import gymnasium as gym

from dap.envs.dap_env import (
    FRAME_STACK,
    OBS_MODES,
    PRIVILEGED_DIM,
    PRIVILEGED_LAYOUT,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
    STATE_DIM,
    STATE_LAYOUT,
    DapEnv,
)

ENV_ID = "dap/PalmMeet-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="dap.envs.dap_env:DapEnv")

__all__ = [
    "DapEnv", "ENV_ID", "FRAME_STACK", "OBS_MODES", "PRIVILEGED_DIM", "PRIVILEGED_LAYOUT",
    "PROPRIO_DIM", "PROPRIO_LAYOUT", "STATE_DIM", "STATE_LAYOUT",
]
