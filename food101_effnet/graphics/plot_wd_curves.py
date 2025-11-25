from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Папка с логами (runs лежит на уровень выше, чем graphics)
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

# Путь, куда сохранить картинку (в ту же папку, где этот скрипт)
OUT_PATH = Path(__file__).with_name("wd_curves.png")

# фильтруем только эксперименты с нужным сетапом
# (exp_name вида erasing_bs64_ls0.15_wd0, wd3e-5, wd1e-4, wd3e-4)
run_dirs = sorted(
    d for d in RUNS_DIR.iterdir()
    if d.is_dir() and "erasing_bs64_ls0.15_wd" in d.name
)

if not run_dirs:
    raise SystemExit(f"Не нашёл ни одной папки с 'erasing_bs64_ls0.15_wd' в {RUNS_DIR}")

print("Найдено экспериментов:")
for d in run_dirs:
    print(" -", d.name)

# одна фигура с 4 сабплотами
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
ax_train_loss, ax_val_loss, ax_train_acc, ax_val_acc = axes.ravel()

for run_dir in run_dirs:
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        print(f"WARNING: в {run_dir} нет log.csv, пропускаю")
        continue

    df = pd.read_csv(log_path)
    epochs = df["epoch"]

    # подпись: обрезаем таймстемп
    # пример: "20251126_120000_erasing_bs64_ls0.15_wd3e-5" -> "erasing_bs64_ls0.15_wd3e-5"
    parts = run_dir.name.split("_", 2)
    label_full = parts[2] if len(parts) >= 3 else run_dir.name

    # можно сократить подпись до только значения wd:
    # "erasing_bs64_ls0.15_wd3e-5" -> "wd3e-5"
    wd_part = label_full.split("_wd")[-1]
    label = "wd" + wd_part

    # потери
    ax_train_loss.plot(epochs, df["train_loss"], label=label)
    ax_val_loss.plot(epochs, df["val_loss"], label=label)

    # точности
    ax_train_acc.plot(epochs, df["train_acc"], label=label)
    ax_val_acc.plot(epochs, df["val_acc"], label=label)

ax_train_loss.set_title("Train loss")
ax_val_loss.set_title("Val loss")
ax_train_acc.set_title("Train accuracy")
ax_val_acc.set_title("Val accuracy")

for ax in (ax_train_loss, ax_val_loss, ax_train_acc, ax_val_acc):
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200)
print(f"Фигура сохранена в {OUT_PATH}")
