"""Стратегии агрегации: FedAvg, FedAvgM, FedProx, SCAFFOLD.

Три первых — встроены в flwr 1.28, просто реэкспорт с параметрами.
SCAFFOLD реализован здесь (наследник FedAvg).

Публичный API: build_strategy(name, cfg) -> Strategy
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import logging
import math

import numpy as np
from flwr.common import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx

log = logging.getLogger(__name__)


def _nd_norm(nds) -> float:
    return float(np.sqrt(sum(float((a.astype(np.float64) ** 2).sum()) for a in nds)))


# ─────────────────────────────────────────────────────────────────────────────
# FedAvgM с корректной обработкой BN buffers
# ─────────────────────────────────────────────────────────────────────────────
#
# Стандартный FedAvgM применяет моментум (экстраполяцию в пространстве весов)
# ко ВСЕМ ключам state_dict, включая BN running_mean/running_var.
# running_var — неотрицательная статистика; экстраполяция может загнать его в <0
# → sqrt(var) = NaN на следующем forward.
#
# Правильное поведение: моментум — только для trainable параметров.
# BN buffers усредняются как в FedAvg.


def _is_bn_buffer(key: str) -> bool:
    return key.endswith(("running_mean", "running_var", "num_batches_tracked"))


class FedAvgMBn(FedAvgM):
    """FedAvgM с исключением BN buffers из server-side momentum."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostics: list[dict] = []

    def aggregate_train(self, server_round, replies):
        # Обходим родительский FedAvgM.aggregate_train и идём сразу в FedAvg
        # чтобы получить чистое взвешенное среднее.
        agg_arrays, agg_metrics = FedAvg.aggregate_train(self, server_round, replies)

        if not self.server_opt or agg_arrays is None:
            return agg_arrays, agg_metrics

        if self.current_arrays is None:
            self.current_arrays = agg_arrays
            return agg_arrays, agg_metrics

        old_nd = self.current_arrays.to_numpy_ndarrays()
        new_nd = agg_arrays.to_numpy_ndarrays()
        keys = list(agg_arrays.keys())

        raw_delta = [o - n for o, n in zip(old_nd, new_nd)]
        pseudo_grad = raw_delta

        if self.server_momentum > 0.0:
            if self.momentum_vector is None:
                self.momentum_vector = pseudo_grad
            else:
                self.momentum_vector = [
                    self.server_momentum * mv + g
                    for mv, g in zip(self.momentum_vector, pseudo_grad)
                ]
            pseudo_grad = self.momentum_vector

        updated: list[Array] = []
        for k, o, n, g in zip(keys, old_nd, new_nd, pseudo_grad):
            if _is_bn_buffer(k):
                updated.append(Array(np.asarray(n)))              # чистое среднее
            else:
                updated.append(Array(np.asarray(o - self.server_learning_rate * g)))

        agg_arrays = ArrayRecord(dict(zip(keys, updated)))
        self.current_arrays = agg_arrays

        # Метрики считаем только по trainable (исключая BN buffers),
        # иначе num_batches_tracked даёт ложно огромные нормы.
        train_idx = [i for i, k in enumerate(keys) if not _is_bn_buffer(k)]
        delta_norm = _nd_norm([raw_delta[i] for i in train_idx])
        mom_norm = (
            _nd_norm([self.momentum_vector[i] for i in train_idx])
            if self.momentum_vector is not None else 0.0
        )
        self.diagnostics.append({
            "round": server_round, "delta_norm": delta_norm, "momentum_norm": mom_norm,
        })
        print(f"  [FedAvgMBn r{server_round}] delta-norm={delta_norm:.4f}  momentum-norm={mom_norm:.4f}", flush=True)
        return agg_arrays, agg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD
# ─────────────────────────────────────────────────────────────────────────────
#
# Сервер хранит:     веса x (ArrayRecord, управляется flwr) + c_server (ArrayRecord)
# Клиент хранит:     c_i (persistent в context.state)
# S→C payload:       arrays=x, config[...] + arrays-доп.ключ "c_server"
# C→S reply:         arrays=y_i, metrics{num-examples,...} + arrays-доп.ключ "c_delta"
#
# Ключевой трюк: Flower ждёт ровно 1 ArrayRecord в reply, поэтому c_delta
# кладётся в тот же ArrayRecord под зарезервированным префиксом, и сервер
# извлекает/удаляет его до вызова super().aggregate_train().

C_SERVER_KEY = "c_server"          # отдельный ArrayRecord в payload S→C
C_DELTA_PREFIX = "__c_delta__/"    # префикс для ключей Δc_i внутри reply-ArrayRecord


class Scaffold(FedAvg):
    """SCAFFOLD (Karimireddy et al., 2020) поверх FedAvg."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._c_server: ArrayRecord | None = None   # инициализируется в первом раунде
        self.diagnostics: list[dict] = []

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Первый раунд: c_server = нули той же формы, что веса
        if self._c_server is None:
            self._c_server = ArrayRecord(
                {k: Array(np.zeros_like(a.numpy())) for k, a in arrays.items()}
            )

        # Кладём c_server в payload отдельным ArrayRecord
        # Переопределяем _construct_messages через временный RecordDict
        node_ids, _ = _sample(self, grid)
        config["server-round"] = server_round
        record = RecordDict({
            self.arrayrecord_key: arrays,
            self.configrecord_key: config,
            C_SERVER_KEY: self._c_server,
        })
        return [
            Message(content=record, message_type="train", dst_node_id=nid)
            for nid in node_ids
        ]

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)

        # Извлекаем Δc_i из каждого reply и удаляем до валидации flwr
        deltas: list[dict[str, np.ndarray]] = []
        for msg in replies:
            if msg.has_error():
                continue
            ar = msg.content[self.arrayrecord_key]
            d = {}
            for k in list(ar.keys()):
                if k.startswith(C_DELTA_PREFIX):
                    d[k[len(C_DELTA_PREFIX):]] = ar[k].numpy()
                    del ar[k]
            deltas.append(d)

        # Обычная агрегация весов + метрик
        arrays, metrics = super().aggregate_train(server_round, replies)

        # c_server += (1/M) * Σ Δc_i    где M = общее число участников (= min_train_nodes)
        if deltas and self._c_server is not None:
            m = float(self.min_train_nodes)
            keys = list(self._c_server.keys())
            c_np = {k: self._c_server[k].numpy() for k in keys}
            for d in deltas:
                for k in keys:
                    if k in d:
                        c_np[k] = c_np[k] + d[k] / m
            self._c_server = ArrayRecord({k: Array(v) for k, v in c_np.items()})

        c_norm = _nd_norm([a.numpy() for a in self._c_server.values()]) if self._c_server else 0.0
        self.diagnostics.append({"round": server_round, "c_server_norm": c_norm})
        print(f"  [SCAFFOLD r{server_round}] c-server-norm={c_norm:.4f}", flush=True)
        return arrays, metrics


# ─────────────────────────────────────────────────────────────────────────────
# FedNova (Wang et al., NeurIPS 2020)
# ─────────────────────────────────────────────────────────────────────────────
#
# Решает objective inconsistency при разном числе локальных шагов τ_i:
#   x_{t+1} = x_t - τ_eff * Σ p_i * (x_t - y_i) / τ_i
#   p_i = n_i / Σ n_j      (вес по объёму данных)
#   τ_eff = Σ p_i * τ_i    (эффективное число шагов)
#
# BN buffers агрегируются как обычное взвешенное среднее (это статистики, не градиенты).


class FedNova(FedAvg):
    """FedNova — нормализованная агрегация для гетерогенных τ_i.

    При server_momentum > 0 применяется серверный моментум поверх
    нормализованного шага: m_{t+1} = β·m_t + τ_eff·H; x_{t+1} = x_t - m_{t+1}.
    BN buffers исключаются из моментума (как в FedAvgMBn).
    """

    def __init__(
        self,
        *,
        server_momentum: float = 0.0,
        server_learning_rate: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._x: ArrayRecord | None = None
        self._momentum: dict[str, np.ndarray] | None = None
        self._server_momentum = float(server_momentum)
        self._server_lr = float(server_learning_rate)
        self.diagnostics: list[dict] = []

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._x = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        valid = [m for m in replies if not m.has_error() and m.has_content()]

        if not valid or self._x is None:
            return super().aggregate_train(server_round, replies)

        ns, taus, ys = [], [], []
        for msg in valid:
            metr = msg.content["metrics"]
            ns.append(float(metr.get("num-examples", 1.0)))
            taus.append(max(float(metr.get("num-steps", 1.0)), 1.0))
            ys.append(msg.content["arrays"])

        N = sum(ns)
        ps = [n / N for n in ns]
        tau_eff = sum(p * t for p, t in zip(ps, taus))

        x_np = {k: a.numpy() for k, a in self._x.items()}
        keys = list(x_np.keys())
        ys_np = [{k: a.numpy() for k, a in y.items()} for y in ys]

        out: dict[str, Array] = {}
        steps: dict[str, np.ndarray] = {}
        for k in keys:
            if _is_bn_buffer(k):
                avg = sum(p * y[k] for p, y in zip(ps, ys_np))
                out[k] = Array(np.asarray(avg))
            else:
                h = sum(p * (x_np[k] - y[k]) / t for p, t, y in zip(ps, taus, ys_np))
                steps[k] = tau_eff * h  # "псевдоградиент"

        if self._server_momentum > 0.0:
            if self._momentum is None:
                self._momentum = {k: np.zeros_like(v) for k, v in steps.items()}
            for k, g in steps.items():
                self._momentum[k] = self._server_momentum * self._momentum[k] + g
            step_applied = self._momentum
        else:
            step_applied = steps

        for k, g in step_applied.items():
            out[k] = Array(np.asarray(x_np[k] - self._server_lr * g))

        x_new = ArrayRecord(out)

        # Метрики — стандартное взвешенное среднее по num-examples
        _, agg_metrics = super().aggregate_train(server_round, replies)

        self.diagnostics.append({
            "round": server_round, "tau_eff": tau_eff,
            "tau_min": min(taus), "tau_max": max(taus),
        })
        print(
            f"  [FedNova r{server_round}] tau_eff={tau_eff:.1f}  "
            f"tau_min={min(taus):.0f}  tau_max={max(taus):.0f}",
            flush=True,
        )
        return x_new, agg_metrics


# Небольшой helper вместо дублирования sample_nodes
def _sample(strat: FedAvg, grid: Grid) -> tuple[list[int], list[int]]:
    from flwr.serverapp.strategy.strategy_utils import sample_nodes
    total = list(grid.get_node_ids())
    sample_size = max(int(len(total) * strat.fraction_train), strat.min_train_nodes)
    return sample_nodes(grid, strat.min_available_nodes, sample_size)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def with_cosine_lr_decay(strategy: FedAvg, num_rounds: int) -> FedAvg:
    """Оборачивает strategy.configure_train: в config добавляется `lr-scale`
    по cosine-кривой от 1.0 (r=1) до 0.0 (r=num_rounds).
    Клиент умножает свой base_lr на lr-scale.
    """
    original = strategy.configure_train

    def wrapped(server_round, arrays, config, grid):
        t = (server_round - 1) / max(num_rounds, 1)
        config["lr-scale"] = float(0.5 * (1.0 + math.cos(math.pi * t)))
        config["server-round"] = server_round
        return original(server_round, arrays, config, grid)

    strategy.configure_train = wrapped
    return strategy


def build_strategy(name: str, *, cfg: dict[str, Any]) -> FedAvg:
    """Собрать стратегию по имени. cfg — полный run_config из pyproject.toml.

    Используемые ключи cfg (остальные игнорируются):
      общие: min-train-nodes, min-available-nodes, fraction-train
      fedavgm:  server-momentum, server-lr
      fedprox:  proximal-mu
    """
    common = dict(
        fraction_train=float(cfg.get("fraction-train", 1.0)),
        fraction_evaluate=0.0,                  # central evaluate на сервере, federated выключен
        min_train_nodes=int(cfg.get("min-train-nodes", 2)),
        min_evaluate_nodes=0,
        min_available_nodes=int(cfg.get("min-available-nodes", 2)),
    )
    name = name.lower()
    if name == "fedavg":
        return FedAvg(**common)
    if name == "fedavgm":
        return FedAvgMBn(
            **common,
            server_learning_rate=float(cfg.get("server-lr", 1.0)),
            server_momentum=float(cfg.get("server-momentum", 0.9)),
        )
    if name == "fedprox":
        return FedProx(**common, proximal_mu=float(cfg.get("proximal-mu", 0.01)))
    if name == "scaffold":
        return Scaffold(**common)
    if name == "fednova":
        return FedNova(**common)
    if name == "fednovam":
        return FedNova(
            **common,
            server_momentum=float(cfg.get("server-momentum", 0.5)),
            server_learning_rate=float(cfg.get("server-lr", 1.0)),
        )
    raise ValueError(f"unknown aggregation: {name!r}")
