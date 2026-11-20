import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motion_audit import success_counts


def test_stop_sample_and_intended_meeting_are_distinct():
    times = pd.DataFrame({"t_end": [1.1, 1.205, 2., 3., 4.]})
    rows = {seed: dict(seed=seed, success=succeeded, t_event=t, time_err=err)
            for seed, succeeded, t, err in [
                (0, True, 1.10, .07),       # first-contact sample at stop
                (1, True, 1.22, .11),       # both samples after stop
                (2, True, 1.50, -.15),      # during reach, outside 100 ms
                (3, False, 1.20, -.02),     # touch failed the alignment test
                (4, True, 1.00, 0.),        # absent from another method
            ]}
    assert success_counts(rows, times, {0, 1, 2, 3}, .01) == {
        "paired_trials": 4, "aligned_touches": 3, "before_stop": 1,
        "at_or_after_stop": 2, "stop_in_final_sample_interval": 1,
        "before_stop_and_within_100ms": 0, "before_stop_and_within_200ms": 1,
        "all_within_100ms": 1, "all_within_200ms": 3,
    }
