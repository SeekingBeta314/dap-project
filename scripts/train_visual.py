"""Fit the depth encoder with paired wrist branches and a local palm-pose loss."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class VisualStudent(nn.Module):
    def __init__(self, vision=True):
        super().__init__()
        self.encoder = None
        if vision:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(24, 24, 3, 2, 1), nn.ReLU(),
            )
        width = 24 * 7 * 9 + 15 if vision else 15
        self.head = nn.Sequential(nn.Linear(width, 192), nn.ReLU(),
                                  nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 15))

    def forward(self, state, image):
        if self.encoder is not None:
            encoded = self.encoder(image[:, None]).flatten(1)
            state = torch.cat([state, encoded], dim=1)
        return self.head(state)


def fit(args):
    paths = sorted(args.data.glob("[0-9]*.npz"))
    if not paths:
        raise ValueError("training directory contains no data")
    seeds = np.array([int(p.stem) for p in paths])
    with np.load(args.jacobians) as data:
        jac = dict(zip(data["seeds"], data["jacobian"]))
    states, images, goals, owner = [], [], [], []
    for i, path in enumerate(paths):
        with np.load(path) as row:
            if row["image"].shape[1:] != (56, 68):
                raise ValueError("recollect with --fine-view")
            states.append(row["state"])
            images.append(row["image"])
            goals.append(row["goal"])
            v_max = row["v_max"].astype("float32")
            owner.extend([i] * len(row["state"]))
    owner = np.array(owner)
    train_mask = seeds[owner] % 5 != 0
    train_rows, valid_rows = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
    x = np.concatenate(states).astype("float32")
    y = np.array(goals, dtype="float32")
    params = dict(x_mean=x[train_mask].mean(0), x_scale=x[train_mask].std(0).clip(.08),
                  y_mean=y[seeds % 5 != 0].mean(0), y_scale=y[seeds % 5 != 0].std(0).clip(.08),
                  vision=np.array(not args.blind), v_max=v_max)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(x):
        return torch.as_tensor(x, device=device)

    x = tensor((x - params["x_mean"]) / params["x_scale"])
    # Integer millimetres keep all examples on-device with a small footprint.
    image = tensor(np.concatenate(images).astype("int16"))
    del images, states
    branch = tensor(y[:, 4] + y[:, 6] > 0)
    y = tensor((y - params["y_mean"]) / params["y_scale"])
    jac = tensor(np.stack([jac[s] for s in seeds]))
    y_scale = tensor(params["y_scale"])
    owner = tensor(owner)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = VisualStudent(vision=not args.blind).to(device)
    if device.type == "cpu" and net.encoder is not None:
        net.encoder.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.003)

    def objectives(idx, training):
        pred = net(x[idx], (image[idx].float() * .001 - 1.5) / .75)
        goal_idx = owner[idx]
        gate = pred[:, 14]
        which = branch[goal_idx] if training else gate > 0
        goal = torch.where(which[:, None], pred[:, 7:14], pred[:, :7])
        residual = goal - y[goal_idx]
        # Ground-truth FK Jacobian only supplies weights during training.
        pose = (jac[goal_idx] @ (residual * y_scale).unsqueeze(-1)).squeeze(-1)
        pose_error = torch.cat((pose[:, :3] / .26, pose[:, 3:] / .05), dim=1)
        return (F.smooth_l1_loss(residual, torch.zeros_like(residual)) +
                F.smooth_l1_loss(pose_error, torch.zeros_like(pose_error)) +
                F.binary_cross_entropy_with_logits(gate, branch[goal_idx].float()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    for epoch in range(args.epochs):
        for idx in np.array_split(rng.permutation(train_rows), max(1, len(train_rows) // 256)):
            loss = objectives(idx, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            sums = [len(idx) * objectives(idx, False).item()
                    for idx in np.array_split(valid_rows, max(1, len(valid_rows) // 256))]
        validation = sum(sums) / len(valid_rows)
        if validation < best:
            best = validation
            for i in range(3):
                layer = net.head[2 * i]
                if not args.blind:
                    conv = net.encoder[2 * i]
                    params[f"kernel{i}"] = conv.weight.detach().cpu().numpy().copy()
                    params[f"offset{i}"] = conv.bias.detach().cpu().numpy().copy()
                params[f"weight{i}"] = layer.weight.detach().cpu().numpy().T.copy()
                params[f"bias{i}"] = layer.bias.detach().cpu().numpy().copy()
            params["epoch"] = epoch + 1
            tmp = args.out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **params)
            tmp.replace(args.out)
        if validation == best or epoch + 1 == args.epochs:
            meta = dict(
                scenarios=len(seeds), validation_scenarios=int(sum(seeds % 5 == 0)),
                examples=len(train_rows) + len(valid_rows), validation_examples=len(valid_rows),
                seed=args.seed, epochs=args.epochs, completed_epochs=epoch + 1,
                selected_epoch=int(params["epoch"]), validation_objective=best,
                training_device=str(device), torch_version=str(torch.__version__),
                numpy_version=np.__version__,
                checkpoint_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                successful_collection_seeds=seeds.tolist(),
            )
            tmp = args.out.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(meta, indent=2) + "\n")
            tmp.replace(args.out.with_suffix(".json"))
        if epoch % 5 == 0:
            print(f"epoch {epoch + 1}: validation objective {validation:.4f}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--jacobians", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blind", action="store_true", help="fit an arm-state-only model with identical labels and loss")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=761)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    fit(args)
