from __future__ import annotations

# --- добавить корень репозитория в PYTHONPATH ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ------------------------------------------------

import argparse
import numpy as np
from torchvision import datasets, transforms
from src.datasets.partition import split_iid, split_dirichlet, save_split

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="splits")
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--mode", type=str, choices=["iid", "dirichlet"], default="dirichlet")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    trainset = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=tf)
    labels = np.array(trainset.targets, dtype=int)
    n, num_classes = len(trainset), 10

    if args.mode == "iid":
        split = split_iid(n, args.num_clients, seed=args.seed)
        name = f"cifar10_iid_K{args.num_clients}_seed{args.seed}.json"
    else:
        split = split_dirichlet(labels, args.num_clients, args.alpha, num_classes, seed=args.seed)
        name = f"cifar10_dirichlet_a{args.alpha}_K{args.num_clients}_seed{args.seed}.json"

    out = Path(args.out_dir) / name
    meta = {
        "dataset": "cifar10",
        "mode": args.mode,
        "alpha": args.alpha if args.mode == "dirichlet" else None,
        "num_clients": args.num_clients,
        "seed": args.seed,
        "num_samples": n,
        "num_classes": num_classes,
    }
    save_split(out, split, meta)
    sizes = [len(s) for s in split]
    print(f"✅ Saved: {out}")
    print(f"Clients: {len(split)} | min/max per-client: {min(sizes)}/{max(sizes)} | total: {sum(sizes)}")

if __name__ == "__main__":
    main()