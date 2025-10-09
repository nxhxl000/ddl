from __future__ import annotations

# --- добавить корень репозитория в PYTHONPATH ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ------------------------------------------------

import argparse
import numpy as np
from collections import Counter
from torchvision import datasets
from src.datasets.partition import load_split

CLASSES = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split_path", type=str, required=True)
    args = ap.parse_args()

    trainset = datasets.CIFAR10(args.data_dir, train=True, download=False)
    labels = np.array(trainset.targets, dtype=int)

    obj = load_split(Path(args.split_path))
    clients = obj["clients"]
    print("Loaded split:", args.split_path)
    print("Clients:", len(clients))

    for cid in range(min(5, len(clients))):
        idxs = clients[cid]
        y = labels[idxs]
        cnt = Counter(y.tolist())
        top = sorted([(CLASSES[k], v) for k, v in cnt.items()], key=lambda x: -x[1])[:3]
        print(f"Client {cid:02d}: size={len(idxs)}, top-classes={top}")

    all_idx = [i for cl in clients for i in cl]
    print("Total indices:", len(all_idx), "| unique:", len(set(all_idx)))

if __name__ == "__main__":
    main()