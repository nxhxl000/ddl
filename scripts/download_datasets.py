"""
Скачивает датасеты (CIFAR-10 и MNIST) и сохраняет локально в папку data/
Запуск:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --data-dir data
  python scripts/download_datasets.py --only cifar10
  python scripts/download_datasets.py --only mnist
  python scripts/download_datasets.py --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datasets import load_dataset


def download_and_save(dataset_name: str, out_dir: Path, force: bool = False) -> None:
    out_dir = out_dir.resolve()

    if out_dir.exists():
        if force:
            print(f"[{dataset_name}] Removing existing directory: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[{dataset_name}] Already exists, skipping: {out_dir}")
            return

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] Downloading via Hugging Face Datasets...")
    ds = load_dataset(dataset_name)  # DatasetDict обычно со split'ами train/test

    # Мини-статистика
    splits = list(ds.keys())
    print(f"[{dataset_name}] Splits: {splits}")
    for s in splits:
        print(f"  - {s}: {len(ds[s])} samples")

    print(f"[{dataset_name}] Saving to disk: {out_dir}")
    ds.save_to_disk(str(out_dir))

    print(f"[{dataset_name}] Done ✅\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Куда сохранять датасеты (по умолчанию: data/)")
    parser.add_argument(
        "--only",
        choices=["cifar10", "mnist", "all"],
        default="all",
        help="Какой датасет скачать: cifar10 | mnist | all (по умолчанию: all)",
    )
    parser.add_argument("--force", action="store_true", help="Перескачать/пересохранить, удалив старую папку")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.only in ("all", "cifar10"):
        download_and_save("cifar10", data_dir / "cifar10", force=args.force)

    if args.only in ("all", "mnist"):
        download_and_save("mnist", data_dir / "mnist", force=args.force)

    print("All requested datasets are ready.")
    print(f"Local paths:\n  - { (data_dir / 'cifar10').resolve() }\n  - { (data_dir / 'mnist').resolve() }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())