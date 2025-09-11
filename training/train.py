# GitHub/ternaryedge-sdk/training/train.py
"""
Minimal training scaffold to exercise TWN ternarization.

Default: MNIST + small CNN using TernaryConv2d/TernaryLinear.
This is a starting point—cleanly organized to grow into KWS/VWW later.

Usage:
  python -m training.train --epochs 1 --batch-size 128 --lr 1e-3 --dataset mnist --quant --t 0.7

Notes:
- Quantization uses TWN-style mapping with STE. Biases are kept in FP.
- Saving: checkpoints go to ./models/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
from rich.console import Console

# Local imports
from quant.ternarize import TernaryConv2d, TernaryLinear, disable_quant, enable_quant

console = Console()


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    dataset: str = "mnist"
    data_root: str = "./data"
    num_workers: int = 2
    t: float = 0.7
    per_channel: bool = True
    quant: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./models"
    model_name: str = "mnist_ternary_cnn.pt"
    seed: int = 1337


class SmallTernaryCNN(nn.Module):
    """
    Small CNN for MNIST; uses ternary conv/linear when quant=True.
    """

    def __init__(self, quant: bool = True, t: float = 0.7, per_channel: bool = True):
        super().__init__()

        Conv = (
            (
                lambda *args, **kw: TernaryConv2d(
                    *args, t=t, per_channel=per_channel, **kw
                )
            )
            if quant
            else nn.Conv2d
        )
        Linear = (
            (lambda *args, **kw: TernaryLinear(*args, t=t, per_channel=False, **kw))
            if quant
            else nn.Linear
        )

        self.net = nn.Sequential(
            Conv(1, 16, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            Conv(16, 32, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            Linear(32 * 7 * 7, 128, bias=True),
            nn.ReLU(inplace=True),
            Linear(128, 10, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset.lower() == "mnist":
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST(
            cfg.data_root, train=True, download=True, transform=tfm
        )
        test_ds = datasets.MNIST(
            cfg.data_root, train=False, download=True, transform=tfm
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        logits = model(images)
        loss = loss_fn(logits, targets)
        running_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="TernaryEdge SDK — training scaffold (TWN)"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--t", type=float, default=0.7, help="Threshold factor for Δ = t * mean(|W|)"
    )
    parser.add_argument(
        "--per-channel", action="store_true", help="Per-out-channel α/Δ (Conv layers)"
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable ternary quantization (use FP weights)",
    )
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--model-name", type=str, default="mnist_ternary_cnn.pt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset=args.dataset,
        data_root=args.data_root,
        num_workers=args.num_workers,
        t=args.t,
        per_channel=args.per_channel,
        quant=not args.no_quant,
        save_dir=args.save_dir,
        model_name=args.model_name,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(args.seed)
    console.rule("[bold]TernaryEdge Training")
    console.print(cfg)

    model = SmallTernaryCNN(quant=cfg.quant, t=cfg.t, per_channel=cfg.per_channel).to(
        cfg.device
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, test_loader = get_dataloaders(cfg)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        console.print(f"[cyan]Epoch {epoch+1}/{cfg.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, cfg.device
        )
        val_loss, val_acc = evaluate(model, test_loader, cfg.device)

        console.print(
            f"[green]Train[/] loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"[yellow]Val[/] loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(cfg.save_dir, cfg.model_name)
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_acc": best_acc,
                },
                ckpt_path,
            )
            console.print(f"[bold green]Saved[/] checkpoint → {ckpt_path}")

    # Example: run with FP weights for eval sanity check if quant was enabled
    if cfg.quant:
        console.print("[magenta]Evaluating with quant disabled (sanity check)...")
        disable_quant(model)
        fp_loss, fp_acc = evaluate(model, test_loader, cfg.device)
        console.print(f"[magenta]FP Eval[/] loss={fp_loss:.4f} acc={fp_acc:.4f}")
        enable_quant(model)


if __name__ == "__main__":
    main()
