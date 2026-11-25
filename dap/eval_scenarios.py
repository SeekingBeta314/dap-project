"""Portable per-seed IK solutions for paired evaluation across CPU architectures."""

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "perception_results" / "scenarios"


def encode_array(value):
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def decode_array(value):
    return np.asarray(value["array"]) if value.keys() == {"array"} else value


def freeze_record(preset, split, seed, spec):
    payload = dict(preset=preset, split=split, seed=seed, spec=spec)
    encoded = json.dumps(payload, default=encode_array, allow_nan=False,
                         sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "scenario": encoded}


@dataclass
class FrozenScenario:
    preset: str
    split: str
    seed: int
    sha256: str
    spec: dict | None

    def build(self, preset, split, seed, camera=None):
        from dap.bank import scenario_from_spec
        from dap.scenario import ScenarioError

        if (self.preset, self.split, self.seed) != (preset, split, seed):
            raise ValueError("frozen scenario identity differs from requested trial")
        if self.spec is None:
            raise ScenarioError(f"scenario sampler exhausted attempts at {preset} {split} seed {seed}")
        return scenario_from_spec(self.spec, camera)


@lru_cache(maxsize=4)
def _read_archive(path):
    records = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            checksum = hashlib.sha256(r["scenario"].encode()).hexdigest()
            if checksum != r["sha256"]:
                raise ValueError(f"{path}: scenario checksum mismatch")
            data = json.loads(r["scenario"], object_hook=decode_array)
            frozen = FrozenScenario(sha256=checksum, **data)
            key = (frozen.preset, frozen.split, frozen.seed)
            if key in records:
                raise ValueError(f"{path}: duplicate {key}")
            records[key] = frozen
    return records


def load_frozen_scenario(cfg, seed, directory=DEFAULT_ARCHIVE):
    path = Path(directory) / f"{cfg.scenario}.jsonl.gz"
    return _read_archive(path)[cfg.scenario, cfg.split, seed]
