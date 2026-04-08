"""
Скачивает датасеты (CIFAR-100, PlantVillage) и сохраняет локально в папку data/
Запуск:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --data-dir data
  python scripts/download_datasets.py --only cifar100
  python scripts/download_datasets.py --only plantvillage
  python scripts/download_datasets.py --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
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


def download_plantvillage_kaggle(out_dir: Path, force: bool = False) -> None:
    """
    Скачивает PlantVillage с Kaggle (abdallahalidev/plantvillage-dataset),
    берёт вариант color/, конвертирует в HuggingFace Dataset и сохраняет на диск.

    Требования:
      pip install kaggle
      ~/.kaggle/kaggle.json  (API-ключ с kaggle.com/settings → API → Create New Token)
    """
    out_dir = out_dir.resolve()

    if out_dir.exists():
        if force:
            print(f"[plantvillage] Removing existing directory: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[plantvillage] Already exists, skipping: {out_dir}")
            return

    # Проверяем наличие kaggle CLI
    if shutil.which("kaggle") is None:
        print("[plantvillage] ERROR: kaggle CLI не найден. Установите: pip install kaggle", file=sys.stderr)
        print("  Затем создайте ~/.kaggle/kaggle.json: https://www.kaggle.com/settings → API → Create New Token",
              file=sys.stderr)
        sys.exit(1)

    tmp_dir = out_dir.parent / "_plantvillage_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("[plantvillage] Downloading from Kaggle: abdallahalidev/plantvillage-dataset ...")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "abdallahalidev/plantvillage-dataset",
                "--path", str(tmp_dir),
                "--unzip",
            ],
            check=True,
        )

        # Ищем папку color/ (оригинальные цветные изображения)
        color_candidates = sorted(tmp_dir.rglob("color"))
        if not color_candidates:
            raise FileNotFoundError(
                f"Папка color/ не найдена внутри {tmp_dir}. "
                f"Содержимое: {list(tmp_dir.iterdir())}"
            )
        color_dir = color_candidates[0]
        print(f"[plantvillage] Найдена color/: {color_dir}")
        print(f"[plantvillage] Классов: {len(list(color_dir.iterdir()))}")

        print("[plantvillage] Конвертирую в HuggingFace Dataset (imagefolder)...")
        ds = load_dataset("imagefolder", data_dir=str(color_dir))

        splits = list(ds.keys())
        print(f"[plantvillage] Splits: {splits}")
        for s in splits:
            print(f"  - {s}: {len(ds[s]):,} samples")

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[plantvillage] Saving to disk: {out_dir}")
        ds.save_to_disk(str(out_dir))

        print("[plantvillage] Done ✅\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Куда сохранять датасеты (по умолчанию: data/)")
    parser.add_argument(
        "--only",
        choices=["cifar100", "plantvillage", "all"],
        default="all",
        help="Какой датасет скачать: cifar100 | plantvillage | all (по умолчанию: all)",
    )
    parser.add_argument("--force", action="store_true", help="Перескачать/пересохранить, удалив старую папку")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.only in ("all", "cifar100"):
        download_and_save("cifar100", data_dir / "cifar100", force=args.force)

    if args.only in ("all", "plantvillage"):
        download_plantvillage_kaggle(data_dir / "plantvillage", force=args.force)

    requested = args.only if args.only != "all" else "cifar100, plantvillage"
    print("All requested datasets are ready.")
    print(f"Requested: {requested}")
    paths = {
        "cifar100":     data_dir / "cifar100",
        "plantvillage": data_dir / "plantvillage",
    }
    for name, path in paths.items():
        if args.only in ("all", name):
            print(f"  - {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
