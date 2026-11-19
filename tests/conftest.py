import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def scenario():
    """One nominal scenario, shared across tests (sampling takes ~0.3 s)."""
    from dap.scenario import SCENARIO_PRESETS, sample_scenario

    return sample_scenario(np.random.default_rng(0), SCENARIO_PRESETS["nominal"])
