from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr
import torch


def _model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    sd = model.state_dict()
    total_bytes = sum(
        int(t.numel() * t.element_size()) for t in sd.values() if hasattr(t, "numel")
    )
    return {
        "total_params":     sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "size_mb":          total_bytes / (1024 ** 2),
        "repr":             repr(model),
        "named_shapes":     {k: tuple(v.shape) for k, v in sd.items()},
    }


def make_exp_dir(
    partition_name: str,
    model_name: str,
    agg_name: str,
    runs_dir: str = "runs",
) -> Tuple[Path, str]:
    """Создать директорию эксперимента. Возвращает (path, exp_name).

    К имени добавляется уникальный 4-символьный hex-суффикс чтобы
    несколько запусков одного эксперимента не перезаписывали друг друга.
    """
    import secrets
    run_id = secrets.token_hex(2)          # 2 байта → 4 hex-символа, напр. "a3f2"
    exp_name = f"{partition_name}__{model_name}__{agg_name}__{run_id}"
    exp_dir = Path(runs_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, exp_name


def write_log_header(
    log_path: Path,
    *,
    config: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    ms = _model_summary(model)
    try:
        import torchvision
        tv_ver = torchvision.__version__
    except ImportError:
        tv_ver = "not-installed"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Flower training log\n")
        f.write(f"Start time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("== Environment ==\n")
        f.write(f"python      : {sys.version.replace(chr(10), ' ')}\n")
        f.write(f"platform    : {platform.platform()}\n")
        f.write(f"flwr        : {flwr.__version__}\n")
        f.write(f"torch       : {torch.__version__}\n")
        f.write(f"torchvision : {tv_ver}\n")
        f.write(f"device      : {device}\n\n")

        f.write("== Experiment config ==\n")
        for k, v in config.items():
            if k != "class_names":
                f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("== Model ==\n")
        f.write(f"params      : {ms['total_params']:,}\n")
        f.write(f"size        : {ms['size_mb']:.3f} MB\n")
        f.write(f"repr        : {ms['repr']}\n\n")
        f.write("State dict shapes:\n")
        for name, shape in ms["named_shapes"].items():
            f.write(f"  {name}: {shape}\n")
        f.write("\n")


def init_csvs(rounds_path: Path, clients_path: Path, classes_path: Path) -> None:
    with rounds_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "round", "test_acc", "test_f1", "delta_acc", "test_loss", "n_clients",
            "comm_down_mb", "comm_up_mb", "cum_comm_mb",
            "train_time_sec", "agg_time_sec", "eval_time_sec", "round_total_time_sec",
            "max_client_time_sec", "mean_client_time_sec",
            "min_client_time_sec", "std_client_time_sec",
            "mean_train_loss", "std_train_loss",
            "mean_drift", "max_drift",
            "effective_js",
        ])
    with clients_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "round", "client_id", "num_examples", "local_epochs",
            "train_loss_last", "train_loss_first", "local_improvement",
            "round_time_sec", "sec_per_1k", "drift",
        ])
    with classes_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "round", "class_id", "class_name", "correct", "total", "accuracy",
        ])


def log_round(
    log_path: Path,
    *,
    server_round: int,
    client_logs: Dict[int, Dict[str, Any]],
    test_acc: float,
    test_f1: float = 0.0,
    test_loss: float,
    train_time: float = 0.0,
    agg_time: float = 0.0,
    eval_time: float = 0.0,
    effective_js: float = 0.0,
) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Round {server_round}:\n")
        for cid in sorted(client_logs):
            p = client_logs[cid]
            improvement = p["first_epoch_loss"] - p["last_epoch_loss"]
            n = p["num_examples"]
            sec_per_1k = p["round_time_sec"] / n * 1000 if n > 0 else 0.0
            f.write(f"  Client {cid + 1}:\n")
            f.write(f"    first epoch loss : {p['first_epoch_loss']:.6f}\n")
            f.write(f"    last  epoch loss : {p['last_epoch_loss']:.6f}  (Δ {improvement:+.6f})\n")
            f.write(f"    drift ||Δw||     : {p['drift']:.6f}\n")
            f.write(f"    train time       : {p['round_time_sec']:.2f}s  ({sec_per_1k:.2f}s/1k examples)\n")
        f.write(f"  Timing: train={train_time:.1f}s  agg={agg_time:.2f}s  eval={eval_time:.2f}s"
                f"  total={train_time + agg_time + eval_time:.1f}s\n")
        js_str = f"  effective_js={effective_js:.4f}" if effective_js > 0.0 else ""
        f.write(f"  Server eval: acc={test_acc:.4f}  f1={test_f1:.4f}  loss={test_loss:.6f}{js_str}\n\n")


def append_rounds_row(
    rounds_path: Path,
    *,
    server_round: int,
    acc: float,
    f1: float = 0.0,
    delta_acc: float,
    loss: float,
    client_logs: Dict[int, Dict[str, Any]],
    model_bytes: int,
    train_time_sec: float,
    agg_time_sec: float,
    eval_time_sec: float,
    cum_comm_mb: float,
    effective_js: float = 0.0,
) -> float:
    """Добавить строку в rounds.csv. Возвращает обновлённый cum_comm_mb."""
    n = len(client_logs)
    client_times = [v["round_time_sec"] for v in client_logs.values()]
    losses       = [v["last_epoch_loss"] for v in client_logs.values()]
    drifts       = [v["drift"] for v in client_logs.values()]

    comm_each    = (model_bytes * n) / 1e6
    cum_comm_mb += comm_each * 2  # down + up

    def _stats(vals: List[float]) -> Tuple[float, float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
        return max(vals), mean, min(vals), std

    max_t, mean_t, min_t, std_t = _stats(client_times)
    mean_loss, std_loss         = sum(losses) / n if n else 0.0, 0.0
    if n > 1:
        std_loss = math.sqrt(sum((x - mean_loss) ** 2 for x in losses) / n)
    mean_drift = sum(drifts) / n if n else 0.0
    max_drift  = max(drifts) if drifts else 0.0
    round_total = train_time_sec + agg_time_sec + eval_time_sec

    with rounds_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            server_round,
            f"{acc:.6f}", f"{f1:.6f}", f"{delta_acc:+.6f}", f"{loss:.6f}", n,
            f"{comm_each:.4f}", f"{comm_each:.4f}", f"{cum_comm_mb:.4f}",
            f"{train_time_sec:.2f}", f"{agg_time_sec:.2f}",
            f"{eval_time_sec:.2f}", f"{round_total:.2f}",
            f"{max_t:.2f}", f"{mean_t:.2f}", f"{min_t:.2f}", f"{std_t:.2f}",
            f"{mean_loss:.6f}", f"{std_loss:.6f}",
            f"{mean_drift:.6f}", f"{max_drift:.6f}",
            f"{effective_js:.6f}",
        ])
    return cum_comm_mb


def append_client_rows(
    clients_path: Path,
    *,
    server_round: int,
    client_logs: Dict[int, Dict[str, Any]],
) -> None:
    with clients_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for cid, v in sorted(client_logs.items()):
            improvement = v["first_epoch_loss"] - v["last_epoch_loss"]
            n = v["num_examples"]
            sec_per_1k = v["round_time_sec"] / n * 1000 if n > 0 else 0.0
            w.writerow([
                server_round, cid,
                n, v["local_epochs"],
                f"{v['last_epoch_loss']:.6f}",
                f"{v['first_epoch_loss']:.6f}",
                f"{improvement:.6f}",
                f"{v['round_time_sec']:.2f}",
                f"{sec_per_1k:.3f}",
                f"{v['drift']:.6f}",
            ])


def append_classes_rows(
    classes_path: Path,
    *,
    server_round: int,
    per_class: Dict[int, Tuple[int, int]],
    class_names: List[str],
) -> None:
    """Добавить per-class accuracy строки в classes.csv."""
    with classes_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for cls_id in sorted(per_class):
            correct, total = per_class[cls_id]
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            acc  = correct / total if total > 0 else 0.0
            w.writerow([server_round, cls_id, name, correct, total, f"{acc:.6f}"])


def generate_plots(
    rounds_csv: Path,
    clients_csv: Path,
    classes_csv: Path,
    out_prefix: Path,
    class_names: List[str],
    exp_name: str,
) -> List[Path]:
    """Генерирует три графика и сохраняет их рядом с CSV-файлами.

    Returns:
        Список путей сохранённых PNG-файлов.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linestyle":   "--",
        "font.size":        11,
    })

    df_r  = pd.read_csv(rounds_csv)
    df_c  = pd.read_csv(clients_csv)
    df_cl = pd.read_csv(classes_csv)
    saved: List[Path] = []

    has_f1 = "test_f1" in df_r.columns

    def _line_plot(metric_col: str, ylabel: str, color: str, title: str, out_path: Path) -> None:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_r["round"], df_r[metric_col] * 100,
                 color=color, linewidth=2, marker="o", markersize=5, label=ylabel)
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel(f"{ylabel} (%)", color=color, fontsize=12)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(bottom=0)

        best_idx = df_r[metric_col].idxmax()
        best_r   = int(df_r.loc[best_idx, "round"])
        best_v   = float(df_r.loc[best_idx, metric_col]) * 100
        ax1.scatter(best_r, best_v, color="gold", s=200, zorder=6,
                    marker="*", label=f"Best: {best_v:.1f}% (r{best_r})")

        ax2 = ax1.twinx()
        ax2.plot(df_r["round"], df_r["test_loss"],
                 color="tomato", linewidth=1.5, linestyle="--", alpha=0.8, label="Test Loss")
        ax2.set_ylabel("Loss", color="tomato", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="tomato")

        lines  = ax1.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower right", fontsize=10)
        ax1.set_title(f"{title} — {exp_name}", fontsize=13, pad=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 1. Accuracy + Loss ────────────────────────────────────────────────────
    acc_path = Path(f"{out_prefix}__accuracy.png")
    _line_plot("test_acc", "Test Accuracy", "steelblue", "Server Accuracy & Loss", acc_path)
    saved.append(acc_path)

    # ── 2. F1 + Loss ──────────────────────────────────────────────────────────
    if has_f1:
        f1_path = Path(f"{out_prefix}__f1.png")
        _line_plot("test_f1", "Macro F1", "seagreen", "Server F1 & Loss", f1_path)
        saved.append(f1_path)

    # ── 3. Train Loss Boxplot (per round) ─────────────────────────────────────
    box_path = Path(f"{out_prefix}__train_loss_boxplot.png")
    rounds   = sorted(df_c["round"].unique())
    data     = [df_c[df_c["round"] == r]["train_loss_last"].dropna().values for r in rounds]
    means    = [d.mean() if len(d) else 0.0 for d in data]

    fig_w = max(8, len(rounds) * 0.75 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bp = ax.boxplot(
        data, positions=rounds, widths=0.55, patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.45, linewidth=1.2),
        medianprops=dict(color="navy", linewidth=2),
        whiskerprops=dict(color="steelblue", linewidth=1.2),
        capprops=dict(color="steelblue", linewidth=1.5),
        flierprops=dict(marker="o", color="gray", markersize=4, alpha=0.5),
    )
    ax.plot(rounds, means, color="tomato", zorder=5, linewidth=1.5,
            linestyle="--", alpha=0.8)
    ax.scatter(rounds, means, color="tomato", zorder=6,
               marker="D", s=50, label="Mean", linewidths=0.5, edgecolors="darkred")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_xticks(rounds)
    if len(rounds) > 20:
        ax.set_xticklabels(
            [str(r) if r % 5 == 0 else "" for r in rounds], fontsize=9
        )
    ax.legend(fontsize=10)
    ax.set_title(f"Client Train Loss Distribution per Round — {exp_name}", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(box_path)

    # ── 4. Per-Class Accuracy Heatmap ─────────────────────────────────────────
    if not df_cl.empty and len(class_names) > 0:
        heat_path = Path(f"{out_prefix}__class_accuracy.png")

        pivot = df_cl.pivot(index="class_name", columns="round", values="accuracy")
        ordered = [cn for cn in class_names if cn in pivot.index]
        if ordered:
            pivot = pivot.loc[ordered]

        n_cls = len(pivot)
        n_rnd = len(pivot.columns)
        fig_w = max(6, n_rnd * 0.75 + 2)
        fig_h = max(4, n_cls * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.85, pad=0.02)

        ax.set_xticks(range(n_rnd))
        ax.set_xticklabels(pivot.columns.tolist(), fontsize=9,
                           rotation=45 if n_rnd > 15 else 0)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Class", fontsize=12)
        ax.set_title(f"Per-Class Accuracy — {exp_name}", fontsize=13, pad=10)

        # Значения в ячейках
        thresh = 0.45
        for i in range(n_cls):
            for j in range(n_rnd):
                val = pivot.values[i, j]
                color = "white" if val < thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        fig.tight_layout()
        fig.savefig(heat_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(heat_path)

    return saved


def print_summary_table(
    rounds_csv: Path,
    *,
    exp_name: str,
    total_wall_time: float,
    cum_comm_mb: float,
) -> None:
    """Печатает читаемую таблицу результатов по раундам в терминал."""
    import pandas as pd

    df = pd.read_csv(rounds_csv)
    if df.empty:
        return

    has_f1 = "test_f1" in df.columns

    sep   = "─" * (74 + (10 if has_f1 else 0))
    hsep  = "═" * (74 + (10 if has_f1 else 0))
    hdr   = f"{'Round':>5}  {'Acc':>8}  " + (f"{'F1':>8}  " if has_f1 else "") + \
            f"{'Δ Acc':>8}  {'Loss':>8}  {'MeanDrift':>10}  {'RoundTime':>10}"

    print(f"\n{hsep}")
    print(f"  {exp_name}")
    print(hsep)
    print(hdr)
    print(sep)

    for _, row in df.iterrows():
        r      = int(row["round"])
        acc    = float(row["test_acc"]) * 100
        f1     = float(row["test_f1"]) * 100 if has_f1 else None
        delta  = float(row["delta_acc"]) * 100
        loss   = float(row["test_loss"])
        drift  = float(row.get("mean_drift", 0.0))
        rtime  = float(row.get("round_total_time_sec", 0.0))
        f1_str = f"  {f1:>7.2f}%" if f1 is not None else ""
        print(
            f"  {r:>3}    {acc:>7.2f}%{f1_str}  {delta:>+8.2f}%  {loss:>8.4f}"
            f"  {drift:>10.4f}  {rtime:>8.1f}s"
        )

    best_row = df.loc[df["test_acc"].idxmax()]
    best_acc = float(best_row["test_acc"]) * 100
    best_r   = int(best_row["round"])
    best_f1_str = ""
    if has_f1:
        best_f1_row = df.loc[df["test_f1"].idxmax()]
        best_f1_str = f"  │  Best F1: {float(best_f1_row['test_f1'])*100:.2f}% @ r{int(best_f1_row['round'])}"

    print(sep)
    print(
        f"  Best Acc: {best_acc:.2f}% @ round {best_r}"
        f"{best_f1_str}"
        f"  │  Comm: {cum_comm_mb:.1f} MB"
        f"  │  Time: {total_wall_time:.0f}s"
    )
    print(f"{hsep}\n")


def write_summary(
    summary_path: Path,
    *,
    exp_name: str,
    all_round_accs: List[Tuple[int, float]],
    all_round_f1s: List[Tuple[int, float]],
    total_wall_time: float,
    cum_comm_mb: float,
    config: Dict[str, Any],
) -> None:
    best_round, best_acc = max(all_round_accs, key=lambda x: x[1])
    best_f1_round, best_f1 = max(all_round_f1s, key=lambda x: x[1]) if all_round_f1s else (0, 0.0)

    def rounds_to(target: float) -> Optional[int]:
        return next((r for r, a in all_round_accs if a >= target), None)

    summary = {
        "exp_name":            exp_name,
        "best_test_acc":       float(best_acc),
        "best_round":          int(best_round),
        "best_test_f1":        float(best_f1),
        "best_f1_round":       int(best_f1_round),
        "rounds_to_80pct":     rounds_to(0.80),
        "rounds_to_90pct":     rounds_to(0.90),
        "rounds_to_95pct":     rounds_to(0.95),
        "total_comm_mb":       float(cum_comm_mb),
        "total_wall_time_sec": float(total_wall_time),
        "config":              config,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
