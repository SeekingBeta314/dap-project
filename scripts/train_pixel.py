"""Distill a goal-joint policy from the examples in scripts/collect_pixel.py.

Uses PyTorch only for training. The output is a plain NumPy checkpoint.
Each fifth training scenario is reserved for model selection by joint error.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def train(paths, output, vision, epochs, seed, split_wrist=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features, labels, validation = [], [], []
    for path in paths:
        with np.load(path) as row:
            state = row["state"]
            if vision:
                state = np.c_[state, row["image"].reshape(len(state), -1) * np.float32(.001)]
            features.append(state)
            labels.append(np.tile(row["goal"], (len(state), 1)))
            validation.extend([int(path.stem) % 5 == 0] * len(state))
            v_max = row["v_max"]
    valid = np.array(validation)
    if valid.all() or not valid.any():
        raise ValueError("need both training and validation scenarios")
    x = np.concatenate(features).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.float32)
    del features, labels
    params = {"vision": np.array(vision), "v_max": v_max.astype(np.float32),
              "x_mean": x[~valid].mean(0), "x_scale": np.maximum(.08, x[~valid].std(0)),
              "y_mean": y[~valid].mean(0), "y_scale": np.maximum(.08, y[~valid].std(0))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor((x - params["x_mean"]) / params["x_scale"], device=device)
    # The expert uses two very different wrist postures. Use the sign of
    # q5 + q7 to define the training branch from the goal configuration.
    branch = torch.tensor((y[:, 4] + y[:, 6]) > 0, device=device)
    y = torch.tensor((y - params["y_mean"]) / params["y_scale"], device=device)
    net = nn.Sequential(nn.Linear(x.shape[1], 192), nn.ReLU(), nn.Linear(192, 128),
                        nn.ReLU(), nn.Linear(128, 15 if split_wrist else 7)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.003)
    best = float("inf")
    train_rows = np.flatnonzero(~valid)
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(train_rows), max(1, len(train_rows) // 128)):
            pred = net(x[idx])
            if split_wrist:
                gate = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred[:, 14], branch[idx].float())
                pred = torch.where(branch[idx, None], pred[:, 7:14], pred[:, :7])
            else:
                gate = 0.
            loss = torch.nn.functional.smooth_l1_loss(pred, y[idx]) + gate
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(x[valid])
            if split_wrist:
                pred = torch.where(pred[:, 14:15] > 0, pred[:, 7:14], pred[:, :7])
            mae = float((pred - y[valid]).abs().mean().item())
        if mae < best:
            best = mae
            for i, layer in enumerate(net[::2]):
                params[f"weight{i}"] = layer.weight.detach().cpu().numpy().T.copy()
                params[f"bias{i}"] = layer.bias.detach().cpu().numpy().copy()
            params["epoch"] = epoch + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **params)
    seeds = [int(p.stem) for p in paths]
    metadata = dict(
        scenarios=len(seeds), validation_scenarios=sum(s % 5 == 0 for s in seeds),
        examples=len(x), validation_examples=int(valid.sum()), seed=seed, epochs=epochs,
        split_wrist=split_wrist, selected_epoch=int(params["epoch"]),
        normalized_validation_mae=best, training_device=str(device),
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        successful_collection_seeds=seeds,
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{output.name} examples={len(x)} validation_examples={valid.sum()} "
          f"best_epoch={params['epoch']} normalized_validation_mae={best:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[ROOT / "banks/pixel_data"])
    parser.add_argument("--split-wrist", action="store_true",
                        help="learn a wrist-branch classifier and two conditional joint targets")
    parser.add_argument("--out", type=Path, default=ROOT / "banks/direct")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=761)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    paths = sorted((p for directory in args.data for p in directory.glob("[0-9]*.npz")),
                   key=lambda p: int(p.stem))
    if len({int(p.stem) for p in paths}) != len(paths):
        raise ValueError("duplicate scenario seeds across input directories")
    print(f"Using {len(paths)} scenarios", flush=True)
    for vision, name in [(False, "proprio"), (True, "depth")]:
        train(paths, args.out / f"{name}.npz", vision, args.epochs, args.seed, args.split_wrist)


if __name__ == "__main__":
    main()
