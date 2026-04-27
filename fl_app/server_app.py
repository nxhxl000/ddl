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
from fl_app.profiling import _gini_sizes, _mean_pairwise_js
from fl_app.scheduler import Schedule, compute_schedule
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

    # Перехват per-client метрик + таймингов (см. experiments_system_heterogeneity.md).
    # T_up per client через delivered_at недоступен: в proto.Metadata нет такого поля,
    # а Python-Metadata.delivered_at пустой при доставке через GrpcGrid. Используем
    # NTP-sync absolute timestamps: created_at (клиент) + t_aggr_start (сервер).
    per_client_rows: list[dict] = []
    round_counter = [0]
    # Хранилища для агрегации в rounds.csv / summary.json
    system_het_per_round: dict[int, dict] = {}
    data_het_overall: dict = {}
    # Schedule (chunks/epochs per pid), вычисляется после round 1 → используется в round 2+
    current_schedule: list[Schedule | None] = [None]

    def with_per_client_timing_capture(strategy):
        """Wrap aggregate_train: считывает metadata.created_at + меряет server-side
        t_aggr_start. Логирует t_compute, t_serialize, t_local, t_lifecycle (на NTP)."""
        original = strategy.aggregate_train

        def wrapped(server_round, replies):
            t_aggr_start = time.time()
            replies = list(replies)
            round_counter[0] = server_round
            drifts: list[float] = []
            class_counts_by_pid: dict[int, dict[int, int]] = {}
            n_examples_by_pid: dict[int, float] = {}
            t_compute_by_pid: dict[int, float] = {}
            for reply in replies:
                if not reply.has_content():
                    continue
                m = reply.content["metrics"]
                t_compute = float(m.get("t-compute", 0))
                t_serialize = float(m.get("t-serialize", 0))
                created_at = float(reply.metadata.created_at)
                t_lifecycle = t_aggr_start - created_at
                drift = float(m.get("w-drift", 0))
                drifts.append(drift)
                pid = int(m.get("partition-id", -1))
                num_ex = float(m.get("num-examples", 0))
                t_compute_by_pid[pid] = t_compute
                n_examples_by_pid[pid] = num_ex
                chunk_frac = float(m.get("chunk-fraction", 1.0))
                local_eps = float(m.get("local-epochs", local_epochs))
                # Объём локальной работы клиента: эффективное число просмотренных сэмплов
                w_client = num_ex * chunk_frac * local_eps
                # Round 1: вытащить data_cls_{N} → распределение классов клиента pid.
                # Удалить data_cls_* и data_* ключи из metrics, чтобы не сломать
                # weighted-aggregate в train_metrics_aggr_fn (он ожидает только числа,
                # а data_cls_* не имеют осмысленного среднего).
                if server_round == 1:
                    counts: dict[int, int] = {}
                    keys_to_drop = [k for k in m.keys() if k.startswith("data_")]
                    for k in keys_to_drop:
                        if k.startswith("data_cls_"):
                            cls_id = int(k.removeprefix("data_cls_"))
                            counts[cls_id] = int(m[k])
                        del m[k]
                    if counts:
                        class_counts_by_pid[pid] = counts
                per_client_rows.append({
                    "round":            server_round,
                    "partition_id":     pid,
                    "num_examples":     num_ex,
                    "chunk_fraction":   chunk_frac,
                    "local_epochs":     local_eps,
                    "w_client":         w_client,
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
                print(f"  [r{server_round}] drift mean={sum(drifts)/len(drifts):.4f}  max={max(drifts):.4f}  min={min(drifts):.4f}", flush=True)

            # ── Метрики гетерогенности ─────────────────────────────────────────
            # Объём локальной работы (round-level)
            ws = [r["w_client"] for r in per_client_rows if r["round"] == server_round]
            n_dropped = sum(1 for w in ws if w == 0)
            w_total = sum(ws)
            w_mean = w_total / len(ws) if ws else 0.0
            w_std = (sum((w - w_mean) ** 2 for w in ws) / len(ws)) ** 0.5 if ws else 0.0
            w_imbal = w_std / w_mean if w_mean > 0 else 0.0

            # System (каждый раунд): SR, IF, I_s — формулы 26-31.
            # Считаем по АКТИВНЫМ клиентам (t_compute > 0); excluded в drop-mode с t=0 пропускаем.
            sr = idle_frac = i_s_val = T_min = T_max = 0.0
            active_pids = [p for p, t in t_compute_by_pid.items() if t > 0]
            if active_pids:
                times = [t_compute_by_pid[p] for p in active_pids]
                T_max, T_min = max(times), min(times)
                sr = T_max / T_min if T_min > 0 else 0.0
                idle_frac = sum((T_max - t) / T_max for t in times) / len(times) if T_max > 0 else 0.0
                ss = [n_examples_by_pid[p] * local_epochs / t_compute_by_pid[p] for p in active_pids]
                N = len(ss)
                sum_abs = sum(abs(ss[i] - ss[j]) for i in range(N) for j in range(N))
                i_s_val = sum_abs / (2 * N * sum(ss)) if sum(ss) > 0 else 0.0
                print(f"  [r{server_round}] SYS HET: SR={sr:.3f}  IF={idle_frac:.3f}  I_s={i_s_val:.4f}  T_min={T_min:.1f}s  T_max={T_max:.1f}s  (active={len(active_pids)}/{len(t_compute_by_pid)})", flush=True)
                print(f"  [r{server_round}] WORK:    W_total={w_total:.0f}  W_imbal={w_imbal:.3f}  n_dropped={n_dropped}", flush=True)

            system_het_per_round[server_round] = {
                "SR": sr, "IF": idle_frac, "I_s": i_s_val,
                "T_min": T_min, "T_max": T_max,
                "W_total": w_total, "W_imbalance": w_imbal,
                "n_dropped": n_dropped,
            }

            # Data (только round 1): MPJS, Gini — формулы 23, 24
            if server_round == 1 and class_counts_by_pid:
                dataset_for_nc = partition.split("__", 1)[0]
                num_classes_d = {"cifar100": 100, "plantvillage": 38}.get(dataset_for_nc, 10)
                dists = [class_counts_by_pid[p] for p in sorted(class_counts_by_pid.keys())]
                mpjs = _mean_pairwise_js(dists, num_classes_d)
                gini_q = _gini_sizes(dists)
                print(f"  [r1] DATA HET: MPJS={mpjs:.4f}  Gini_quantity={gini_q:.4f}  num_classes={num_classes_d}", flush=True)
                data_het_overall.update({
                    "MPJS": mpjs, "Gini_quantity": gini_q, "num_classes": num_classes_d,
                })

            # Schedule (только round 1): из t_compute_by_pid → chunks/epochs/excluded
            if server_round == 1 and t_compute_by_pid:
                straggler_mode     = str(rc.get("straggler-mode", "none")).lower()
                straggler_target   = str(rc.get("straggler-target", "min")).lower()
                straggler_tol      = float(rc.get("straggler-tolerance", 0.05))
                straggler_drop_tol = float(rc.get("straggler-drop-tolerance", 0.5))
                straggler_max_drop = int(rc.get("straggler-max-dropped", 3))
                straggler_min_chk  = float(rc.get("straggler-min-chunk", 0.1))
                straggler_min_ep   = int(rc.get("straggler-min-epochs", 1))
                sched = compute_schedule(
                    t_compute_by_pid,
                    mode=straggler_mode,
                    base_epochs=local_epochs,
                    target=straggler_target,
                    tolerance=straggler_tol,
                    drop_tolerance=straggler_drop_tol,
                    max_dropped=straggler_max_drop,
                    min_chunk=straggler_min_chk,
                    min_epochs=straggler_min_ep,
                )
                current_schedule[0] = sched
                (exp_dir / "schedule.json").write_text(json.dumps(sched.to_dict(), indent=2))
                if sched.mode == "drop":
                    print(f"  [r1] SCHEDULE (drop, target={sched.target}→T_target={sched.T_target:.1f}s, T_drop={sched.T_drop:.1f}s, drop_tol={sched.drop_tolerance:.0%}, max_dropped={straggler_max_drop}):", flush=True)
                    for p in sorted(t_compute_by_pid.keys()):
                        marker = "✗ excluded" if p in sched.excluded else "✓ kept"
                        print(f"    pid {p:2d}  t_comp={t_compute_by_pid[p]:6.1f}s  {marker}", flush=True)
                    print(f"    → {len(sched.excluded)}/{len(t_compute_by_pid)} dropped: {sched.excluded}", flush=True)
                elif sched.mode in ("chunk", "epochs"):
                    print(f"  [r1] SCHEDULE ({sched.mode}, target={sched.target}→T_target={sched.T_target:.1f}s, T_upper={sched.T_upper:.1f}s, tol={sched.tolerance:.0%}):", flush=True)
                    for p in sorted(sched.chunks.keys()):
                        in_band = "↻ in-band" if t_compute_by_pid[p] <= sched.T_upper else ""
                        print(f"    pid {p:2d}  t_comp={t_compute_by_pid[p]:6.1f}s  chunk={sched.chunks[p]:.3f}  epochs={sched.epochs[p]}  {in_band}", flush=True)
                else:
                    print(f"  [r1] SCHEDULE: mode=none (no straggler mitigation)", flush=True)

            result = original(server_round, replies)

            # Инкрементальный append: per-client строки этого раунда → clients.csv
            rows_this = [r for r in per_client_rows if r["round"] == server_round]
            if rows_this:
                clients_path = exp_dir / "clients.csv"
                pd.DataFrame(rows_this).to_csv(
                    clients_path, mode="a",
                    header=not clients_path.exists(),
                    index=False,
                )

            return result

        strategy.aggregate_train = wrapped
        return strategy

    def with_dynamic_schedule(strategy):
        """Инжектит per-client-chunks / per-client-epochs / excluded-clients из
        current_schedule в outbound config. До round 1 ничего не добавляет.
        """
        original = strategy.configure_train

        def wrapped(server_round, arrays, config, grid):
            sched = current_schedule[0]
            if sched is not None and sched.mode != "none":
                if sched.mode == "drop":
                    config["excluded-clients"] = sched.excluded_str()
                else:  # chunk/epochs
                    config["per-client-chunks"] = sched.chunks_str()
                    config["per-client-epochs"] = sched.epochs_str()
            return original(server_round, arrays, config, grid)

        strategy.configure_train = wrapped
        return strategy

    # Захват агрегированных train-метрик (для инкрементальной записи в rounds.csv).
    train_metrics_per_round: dict[int, dict] = {}

    def train_aggr(reply_contents, weighted_by_key):
        agg = aggregate_metricrecords(reply_contents, weighted_by_key)
        if round_counter[0] > 0:
            train_metrics_per_round[round_counter[0]] = {
                k: float(v) for k, v in dict(agg).items() if isinstance(v, (int, float))
            }
        return agg

    strategy = build_strategy(agg_name, cfg=rc)
    strategy = with_cosine_lr_decay(strategy, num_rounds)
    strategy = with_per_client_timing_capture(strategy)
    strategy = with_dynamic_schedule(strategy)
    strategy.train_metrics_aggr_fn = train_aggr

    # Callback центральной evaluate + трекинг лучшей модели + инкрементальная запись
    best = {"acc": -1.0, "round": 0, "arrays": None}
    rows: list[dict] = []  # rounds-level строки, накапливаем для plot'ов в конце

    def _write_summary() -> None:
        sr_vals = [r["SR"]  for r in rows if r["SR"]  > 0]
        if_vals = [r["IF"]  for r in rows if r["T_max"] > 0]
        is_vals = [r["I_s"] for r in rows if r["I_s"] > 0]
        summary = {
            "config":            dict(rc),
            "best_acc":          best["acc"],
            "best_round":        best["round"],
            "rounds_completed":  len(rows),
            "num_rounds":        num_rounds,
            "data_heterogeneity": data_het_overall,
            "system_heterogeneity_mean": {
                "SR":          sum(sr_vals) / len(sr_vals) if sr_vals else 0.0,
                "IF":          sum(if_vals) / len(if_vals) if if_vals else 0.0,
                "I_s":         sum(is_vals) / len(is_vals) if is_vals else 0.0,
                "W_total_sum": sum(r["W_total"] for r in rows),
            },
            "excluded_clients":  rc.get("excluded-clients", ""),
            "per_client_chunks": rc.get("per-client-chunks", ""),
        }
        (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    def eval_fn(server_round: int, arrays: ArrayRecord):
        m = build_model(model_name)
        m.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        r = evaluate(m, test_loader, device)
        if r["acc"] > best["acc"]:
            best["acc"] = r["acc"]
            best["round"] = server_round
            best["arrays"] = arrays

        # server_round=0 — initial global eval (до обучения), не пишем в rounds.csv
        if server_round == 0 or server_round > num_rounds:
            return MetricRecord({
                "test-loss": r["loss"],
                "test-acc":  r["acc"],
                "test-f1":   r["f1_macro"],
            })

        # ── Инкрементальная запись round-level строки ─────────────────────────
        tm = train_metrics_per_round.get(server_round, {})
        sh = system_het_per_round.get(server_round, {})
        diag_dict = {d["round"]: d for d in getattr(strategy, "diagnostics", [])}
        dg = diag_dict.get(server_round, {})
        row = {
            "round":                 server_round,
            "test_loss":             float(r["loss"]),
            "test_acc":              float(r["acc"]),
            "test_f1":               float(r["f1_macro"]),
            "train_loss_first_mean": float(tm.get("train-loss-first", 0)),
            "train_loss_last_mean":  float(tm.get("train-loss-last", 0)),
            "t_compute_mean":        float(tm.get("t-compute", 0)),
            "drift_mean":            float(tm.get("w-drift", 0)),
            "update_norm_rel_mean":  float(tm.get("update-norm-rel", 0)),
            "grad_norm_last_mean":   float(tm.get("grad-norm-last", 0)),
            "delta_norm":            float(dg.get("delta_norm", 0)),
            "momentum_norm":         float(dg.get("momentum_norm", 0)),
            "c_server_norm":         float(dg.get("c_server_norm", 0)),
            "comm_mb":               comm_mb,
            "SR":                    float(sh.get("SR", 0)),
            "IF":                    float(sh.get("IF", 0)),
            "I_s":                   float(sh.get("I_s", 0)),
            "T_min":                 float(sh.get("T_min", 0)),
            "T_max":                 float(sh.get("T_max", 0)),
            "W_total":               float(sh.get("W_total", 0)),
            "W_imbalance":           float(sh.get("W_imbalance", 0)),
            "n_dropped":             int(sh.get("n_dropped", 0)),
        }
        rows.append(row)
        rounds_path = exp_dir / "rounds.csv"
        pd.DataFrame([row]).to_csv(
            rounds_path, mode="a",
            header=not rounds_path.exists(),
            index=False,
        )
        _write_summary()

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
    # rounds.csv, clients.csv, summary.json уже записаны инкрементально
    # (eval_fn после каждого раунда + wrapper для clients.csv).
    # Здесь только финальный summary update + модели + графики.
    _write_summary()

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
