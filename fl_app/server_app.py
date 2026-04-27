"""Flower ServerApp — центральная evaluate + сохранение артефактов."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_app.data import build_loader
from fl_app.models import build_model, get_hparams
from fl_app.profiling import (
    print_profiling_summary,
    run_profiling_round,
    save_cluster_profile,
)
from fl_app.strategies import build_strategy, with_cosine_lr_decay
from fl_app.training import evaluate, get_device

app = ServerApp()

# CIFAR-100: 20 суперклассов × 5 fine classes (стандартный маппинг)
CIFAR100_SUPERCLASSES = [
    ("aquatic mammals",   [4, 30, 55, 72, 95]),
    ("fish",              [1, 32, 67, 73, 91]),
    ("flowers",           [54, 62, 70, 82, 92]),
    ("food containers",   [9, 10, 16, 28, 61]),
    ("fruit/vegetables",  [0, 51, 53, 57, 83]),
    ("electrical devices",[22, 39, 40, 86, 87]),
    ("furniture",         [5, 20, 25, 84, 94]),
    ("insects",           [6, 7, 14, 18, 24]),
    ("large carnivores",  [3, 42, 43, 88, 97]),
    ("man-made outdoor",  [12, 17, 37, 68, 76]),
    ("natural outdoor",   [23, 33, 49, 60, 71]),
    ("large omni/herb",   [15, 19, 21, 31, 38]),
    ("medium mammals",    [34, 63, 64, 66, 75]),
    ("non-insect inv.",   [26, 45, 77, 79, 99]),
    ("people",            [2, 11, 35, 46, 98]),
    ("reptiles",          [27, 29, 44, 78, 93]),
    ("small mammals",     [36, 50, 65, 74, 80]),
    ("trees",             [47, 52, 56, 59, 96]),
    ("vehicles 1",        [8, 13, 48, 58, 90]),
    ("vehicles 2",        [41, 69, 81, 85, 89]),
]
CIFAR100_FINE_NAMES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm",
]

_PLOT_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "font.size": 11,
}


def _line_plot(df, metric, ylabel, color, title, out_path):
    with plt.rc_context(_PLOT_RC):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["round"], df[metric] * 100, color=color, linewidth=2,
                 marker="o", markersize=5, label=ylabel)
        ax1.set_xlabel("Round"); ax1.set_ylabel(f"{ylabel} (%)", color=color)
        ax1.tick_params(axis="y", labelcolor=color); ax1.set_ylim(bottom=0)
        best_i = df[metric].idxmax()
        br, bv = int(df.loc[best_i, "round"]), float(df.loc[best_i, metric]) * 100
        ax1.scatter(br, bv, color="gold", s=200, zorder=6, marker="*",
                    label=f"Best: {bv:.1f}% (r{br})", edgecolors="goldenrod")
        ax2 = ax1.twinx()
        ax2.plot(df["round"], df["test_loss"], color="tomato", linewidth=1.5,
                 linestyle="--", alpha=0.8, label="Test Loss")
        ax2.set_ylabel("Loss", color="tomato"); ax2.tick_params(axis="y", labelcolor="tomato")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower right")
        ax1.set_title(title, pad=10)
        fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _boxplot_train_loss(df_cli, out_path, title):
    rounds = sorted(df_cli["round"].unique())
    data = [df_cli[df_cli["round"] == r]["train_loss_last"].dropna().values for r in rounds]
    means = [d.mean() if len(d) else 0.0 for d in data]
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.3 + 2), 5))
        ax.boxplot(data, positions=rounds, widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.45, linewidth=1.2),
                   medianprops=dict(color="navy", linewidth=2),
                   whiskerprops=dict(color="steelblue", linewidth=1.2),
                   capprops=dict(color="steelblue", linewidth=1.5),
                   flierprops=dict(marker="o", color="gray", markersize=4, alpha=0.5))
        ax.plot(rounds, means, color="tomato", linestyle="--", linewidth=1.5, alpha=0.8, zorder=5)
        ax.scatter(rounds, means, color="tomato", marker="D", s=50, zorder=6,
                   edgecolors="darkred", linewidths=0.5, label="Mean")
        ax.set_xlabel("Round"); ax.set_ylabel("Train Loss")
        step = max(1, len(rounds) // 20)
        ax.set_xticks(rounds[::step])
        ax.legend(); ax.set_title(title, pad=10)
        fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _cifar100_superclass_heatmap(per_class, out_path, title):
    grid = np.array([[per_class[c] for c in fine] for _, fine in CIFAR100_SUPERCLASSES])
    labels = np.array([[CIFAR100_FINE_NAMES[c] for c in fine] for _, fine in CIFAR100_SUPERCLASSES])
    superclass_names = [s for s, _ in CIFAR100_SUPERCLASSES]
    with plt.rc_context(_PLOT_RC):
        fig, ax = plt.subplots(figsize=(11, 11))
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.7, pad=0.02)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"#{i+1}" for i in range(5)])
        ax.set_yticks(range(20)); ax.set_yticklabels(superclass_names)
        ax.set_xlabel("Fine class index within superclass"); ax.set_ylabel("Superclass")
        ax.set_title(title, pad=10); ax.grid(False)
        for i in range(20):
            for j in range(5):
                v = grid[i, j]
                color = "white" if v < 0.45 else "black"
                ax.text(j, i, f"{labels[i, j]}\n{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
        fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    model_name = rc["model"]
    agg_name = str(rc["aggregation"]).lower()
    num_rounds = int(rc.get("num-server-rounds", 10))
    model_hp = get_hparams(model_name, agg_name)
    local_epochs = int(rc.get("local-epochs", model_hp["local-epochs"]))
    partition = rc["partition-name"]
    data_dir = rc.get("data-dir", "data/")
    exp_root = rc.get("experiments-dir", "simulation")

    # Модель и начальные веса
    model = build_model(model_name)
    initial_arrays = ArrayRecord(model.state_dict())
    comm_mb = sum(a.numpy().nbytes for a in initial_arrays.values()) / (1024 ** 2)

    # Test loader — централизованная evaluate на стороне сервера
    test_loader = build_loader(
        Path(data_dir) / "partitions" / partition / "test",
        batch_size=256, train=False,
    )
    device = get_device()

    # Дир эксперимента: {root}/{dataset}/{model}/{agg}/{partition_tail}__{timestamp}/
    dataset, _, partition_tail = partition.partition("__")
    partition_tail = partition_tail.replace("__", "_") or "default"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(exp_root) / dataset / model_name / agg_name / f"{partition_tail}__r{num_rounds}__{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(dict(rc), indent=2, default=str))

    # ── Профилировочный раунд (опционально) ───────────────────────────────────
    if str(rc.get("enable-profiling", "false")).lower() == "true":
        min_nodes = int(rc.get("min-train-nodes", 1))
        min_avail = int(rc.get("min-available-nodes", min_nodes))
        profiles = run_profiling_round(
            grid, initial_arrays,
            fraction_train=float(rc.get("fraction-train", 1.0)),
            min_train_nodes=min_nodes,
            min_available_nodes=min_avail,
            benchmark_samples=int(rc.get("benchmark-samples", 1000)),
            benchmark_epochs=int(rc.get("benchmark-epochs", 2)),
        )
        num_classes = {"cifar100": 100, "plantvillage": 38}.get(dataset, 10)
        save_cluster_profile(profiles, exp_dir, partition_name=partition, num_classes=num_classes)
        print_profiling_summary(profiles, num_classes=num_classes)

    # Перехват per-client метрик + таймингов (см. experiments_system_heterogeneity.md).
    # T_up per client через delivered_at недоступен: в proto.Metadata нет такого поля,
    # а Python-Metadata.delivered_at пустой при доставке через GrpcGrid. Используем
    # NTP-sync absolute timestamps: created_at (клиент) + t_aggr_start (сервер).
    per_client_rows: list[dict] = []
    round_counter = [0]

    def with_per_client_timing_capture(strategy):
        """Wrap aggregate_train: считывает metadata.created_at + меряет server-side
        t_aggr_start. Логирует t_compute, t_serialize, t_local, t_lifecycle (на NTP)."""
        original = strategy.aggregate_train

        def wrapped(server_round, replies):
            t_aggr_start = time.time()
            replies = list(replies)
            round_counter[0] = server_round
            drifts: list[float] = []
            for reply in replies:
                if not reply.has_content():
                    continue
                m = reply.content["metrics"]
                t_compute = float(m.get("t-compute", 0))
                t_serialize = float(m.get("t-serialize", 0))
                created_at = float(reply.metadata.created_at)
                # t_lifecycle = T_up + idle_wait (по NTP, сервер ↔ клиент).
                # На NTP-кластере YC точность ≈ 10-50 ms.
                t_lifecycle = t_aggr_start - created_at
                drift = float(m.get("w-drift", 0))
                drifts.append(drift)
                per_client_rows.append({
                    "round":            server_round,
                    "partition_id":     int(m.get("partition-id", -1)),
                    "num_examples":     float(m.get("num-examples", 0)),
                    "train_loss_first": float(m.get("train-loss-first", 0)),
                    "train_loss_last":  float(m.get("train-loss-last", 0)),
                    "t_compute":        t_compute,
                    "t_serialize":      t_serialize,
                    "t_local":          t_compute + t_serialize,
                    "created_at":       created_at,
                    "t_aggr_start":     t_aggr_start,
                    "t_lifecycle":      t_lifecycle,
                    "w_drift":          drift,
                    "update_norm_rel":  float(m.get("update-norm-rel", 0)),
                    "grad_norm_last":   float(m.get("grad-norm-last", 0)),
                })
            if drifts:
                print(f"  [r{server_round}] drift mean={sum(drifts)/len(drifts):.4f}  max={max(drifts):.4f}  min={min(drifts):.4f}")
            return original(server_round, replies)

        strategy.aggregate_train = wrapped
        return strategy

    def train_aggr(reply_contents, weighted_by_key):
        # Просто переиспользуем стандартную агрегацию метрик; per-client логирование
        # делается в обёртке aggregate_train выше.
        return aggregate_metricrecords(reply_contents, weighted_by_key)

    strategy = build_strategy(agg_name, cfg=rc)
    strategy = with_cosine_lr_decay(strategy, num_rounds)
    strategy = with_per_client_timing_capture(strategy)
    strategy.train_metrics_aggr_fn = train_aggr

    # Callback центральной evaluate + трекинг лучшей модели
    best = {"acc": -1.0, "round": 0, "arrays": None}
    def eval_fn(server_round: int, arrays: ArrayRecord):
        m = build_model(model_name)
        m.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        r = evaluate(m, test_loader, device)
        if r["acc"] > best["acc"]:
            best["acc"] = r["acc"]
            best["round"] = server_round
            best["arrays"] = arrays
        return MetricRecord({
            "test-loss": r["loss"],
            "test-acc":  r["acc"],
            "test-f1":   r["f1_macro"],
        })

    # FL loop
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": local_epochs}),
        evaluate_fn=eval_fn,
    )

    # ── Артефакты ─────────────────────────────────────────────────────────────
    diagnostics = {d["round"]: d for d in getattr(strategy, "diagnostics", [])}
    rows = []
    for r in range(1, num_rounds + 1):
        tm = result.train_metrics_clientapp.get(r, {})
        em = result.evaluate_metrics_serverapp.get(r, {})
        dg = diagnostics.get(r, {})
        rows.append({
            "round": r,
            "test_loss": float(em.get("test-loss", 0)),
            "test_acc":  float(em.get("test-acc", 0)),
            "test_f1":   float(em.get("test-f1", 0)),
            "train_loss_first_mean": float(tm.get("train-loss-first", 0)),
            "train_loss_last_mean":  float(tm.get("train-loss-last", 0)),
            "t_compute_mean":        float(tm.get("t-compute", 0)),
            "drift_mean":            float(tm.get("w-drift", 0)),
            "update_norm_rel_mean":  float(tm.get("update-norm-rel", 0)),
            "grad_norm_last_mean":   float(tm.get("grad-norm-last", 0)),
            "delta_norm":            float(dg.get("delta_norm", 0)),
            "momentum_norm":         float(dg.get("momentum_norm", 0)),
            "c_server_norm":         float(dg.get("c_server_norm", 0)),
            "comm_mb": comm_mb,
        })
    pd.DataFrame(rows).to_csv(exp_dir / "rounds.csv", index=False)
    pd.DataFrame(per_client_rows).to_csv(exp_dir / "clients.csv", index=False)

    # Финальная модель
    final = build_model(model_name)
    final.load_state_dict(result.arrays.to_torch_state_dict(), strict=True)
    torch.save(final.state_dict(), exp_dir / "model_final.pt")

    exp_tag = f"{dataset}/{model_name}/{agg_name}"

    # Best model + per-class heatmap
    if best["arrays"] is not None:
        bm = build_model(model_name)
        bm.load_state_dict(best["arrays"].to_torch_state_dict(), strict=True)
        torch.save(bm.state_dict(), exp_dir / "model_best.pt")
        br = evaluate(bm, test_loader, device)
        pc = br["per_class"]
        pd.DataFrame({"class_id": range(len(pc)),
                      "class_name": CIFAR100_FINE_NAMES[:len(pc)] if len(pc) == 100 else list(range(len(pc))),
                      "accuracy": pc}).to_csv(exp_dir / "class_accuracy.csv", index=False)
        if len(pc) == 100:
            _cifar100_superclass_heatmap(
                pc, exp_dir / "class_accuracy.png",
                f"Per-class accuracy — best (r{best['round']}, acc={br['acc']:.4f}) — {exp_tag}",
            )

    # Графики
    df = pd.DataFrame(rows)
    _line_plot(df, "test_acc", "Test Accuracy", "steelblue",
               f"Server Accuracy & Loss — {exp_tag}", exp_dir / "accuracy.png")
    _line_plot(df, "test_f1", "Macro F1", "seagreen",
               f"Server F1 & Loss — {exp_tag}", exp_dir / "f1.png")

    df_cli = pd.DataFrame(per_client_rows)
    if not df_cli.empty:
        _boxplot_train_loss(df_cli, exp_dir / "train_loss_boxplot.png",
                            f"Client Train Loss per Round — {exp_tag}")

    print(f"[server] Done. Artifacts: {exp_dir}")
