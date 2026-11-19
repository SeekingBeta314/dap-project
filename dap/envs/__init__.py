"""Gymnasium registration. `import dap.envs` makes "dap/PalmMeet-v0"
available to `gymnasium.make`."""

import gymnasium as gym

from dap.envs.dap_env import STATE_DIM, STATE_LAYOUT, DapEnv

ENV_ID = "dap/PalmMeet-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="dap.envs.dap_env:DapEnv")

__all__ = ["DapEnv", "ENV_ID", "STATE_DIM", "STATE_LAYOUT"]
