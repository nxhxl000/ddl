# train_food101.py
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import amp as torch_amp
import torch.nn as nn

from .config import parse_args
from .data import make_loaders
from .models import build_model
from .optim import make_optimizer
from .train_loop import train_one_epoch, validate
from .utils import set_seed, pick_device, log_env_and_device, save_checkpoint

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(cfg.run_dir) / f"{timestamp}_{cfg.exp_name}"
    run_path.mkdir(parents=True, exist_ok=True)

    (run_path / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = pick_device(cfg.device)
    if cfg.device == "cuda" and device.type != "cuda" and cfg.force_device:
        raise RuntimeError("Запрошен --device cuda, но CUDA недоступна.")

    use_amp_cuda = cfg.amp and (device.type == "cuda")
    log_env_and_device(run_path, device, amp_enabled=use_amp_cuda)

    train_loader, val_loader, classes = make_loaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = make_optimizer(cfg, model.parameters())
    scaler = torch_amp.GradScaler(enabled=use_amp_cuda)

    csv_path = run_path / "log.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,train_time_s,val_time_s,device\n")

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, t_train = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler, use_amp_cuda
        )
        val_loss, val_acc, t_val = validate(
            model, val_loader, device, criterion, use_amp_cuda
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"(time {t_train:.1f}s) | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% "
            f"(time {t_val:.1f}s) | device={device.type}"
        )

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{t_train:.3f},{t_val:.3f},{device.type}\n"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(run_path / "best.ckpt", model, optimizer, epoch, best_val_acc)

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(run_path / f"epoch_{epoch}.ckpt", model, optimizer, epoch, best_val_acc)

    save_checkpoint(run_path / "last.ckpt", model, optimizer, cfg.epochs, best_val_acc)

    summary = {
        "best_val_acc": best_val_acc,
        "epochs": cfg.epochs,
        "csv_log": str(csv_path),
        "best_ckpt": str(run_path / "best.ckpt"),
        "last_ckpt": str(run_path / "last.ckpt"),
        "device": device.type,
        "classes": classes[:10] if isinstance(classes, list) else None,
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✔ Done. Logs: {csv_path}\nBest val acc: {best_val_acc*100:.2f}%\nRun dir: {run_path}")

if __name__ == "__main__":
    main()