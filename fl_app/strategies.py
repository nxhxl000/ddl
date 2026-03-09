from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import torch
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import (
    Bulyan,
    FedAdam,
    FedAdagrad,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    MultiKrum,
)


# ── Strategy configs (дефолтные параметры) ───────────────────────────────────

@dataclass
class FedAvgCfg:
    pass

@dataclass
class FedProxCfg:
    proximal_mu: float = 0.01

@dataclass
class FedAvgMCfg:
    server_learning_rate: float = 1.0
    server_momentum:      float = 0.9

@dataclass
class FedAdamCfg:
    eta:    float = 0.01
    eta_l:  float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau:    float = 1e-9

@dataclass
class FedYogiCfg:
    eta:    float = 0.01
    eta_l:  float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau:    float = 1e-9

@dataclass
class FedAdagradCfg:
    eta:   float = 0.01
    eta_l: float = 0.001
    tau:   float = 1e-9

@dataclass
class FedMedianCfg:
    pass

@dataclass
class FedTrimmedAvgCfg:
    beta: float = 0.2

@dataclass
class KrumCfg:
    num_malicious_nodes: int = 0

@dataclass
class MultiKrumCfg:
    num_malicious_nodes:  int = 0
    num_nodes_to_select:  int = 1

@dataclass
class BulyanCfg:
    num_malicious_nodes: int = 0

@dataclass
class ScaffoldCfg:
    pass  # параметры совпадают с FedAvg; server_lr=1.0 зафиксирован в алгоритме


# ── SCAFFOLD Strategy ─────────────────────────────────────────────────────────

class ScaffoldStrategy(FedAvg):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

    Корректирует client drift через control variates:
      - c_server (глобальный) хранится на сервере, передаётся клиентам каждый раунд
      - c_i (локальный) хранится в context.state клиента, персистентно между раундами
      - Поправка к градиенту: grad += c_server - c_i
      - Обновление c_i (Option I): c_i_new = c_i - c_server + (x - y_i) / (K * lr)
      - Обновление c_server: c = c + (1/n) * sum(delta_c_i)

    Ref: Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging
         for Federated Learning", ICML 2020.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._c_server: Optional[ArrayRecord] = None  # инициализируется в первом раунде
        # Поля совместимости с LoggingStrategy (нужны server_app.py)
        self.round_client_logs: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._agg_start_times: Dict[int, float] = {}
        self._agg_end_times:   Dict[int, float] = {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: Any, grid: Any
    ) -> Iterable[Message]:
        # Инициализируем c_server нулями по структуре модели
        if self._c_server is None:
            state = arrays.to_torch_state_dict()
            self._c_server = ArrayRecord({k: torch.zeros_like(v) for k, v in state.items()})

        # Базовая логика FedAvg (node selection, server-round injection)
        messages = list(super().configure_train(server_round, arrays, config, grid))

        # Добавляем c_server в каждое сообщение
        for msg in messages:
            msg.content["c_server"] = self._c_server

        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies_list = list(replies)
        per_round: Dict[int, Dict[str, Any]] = {}

        # Извлекаем c_delta и логируем метрики
        c_delta_sum: Optional[Dict[str, torch.Tensor]] = None
        n_scaffold = 0

        for rep in replies_list:
            if not rep.has_content():
                continue
            src = rep.metadata.src_node_id

            if "metrics" in rep.content:
                m: MetricRecord = rep.content["metrics"]  # type: ignore[assignment]
                cid = int(m.get("partition-id", float(src)))
                per_round[cid] = {
                    "first_epoch_loss": float(m.get("first-epoch-loss", 0.0)),
                    "last_epoch_loss":  float(m.get("last-epoch-loss", 0.0)),
                    "round_time_sec":   float(m.get("round-time-sec", 0.0)),
                    "local_epochs":     int(m.get("local-epochs", 0)),
                    "num_examples":     int(m.get("num-examples", 0)),
                    "src_node_id":      int(src),
                    "drift":            float(m.get("drift", 0.0)),
                }

            if "c_delta" in rep.content:
                delta_state = rep.content["c_delta"].to_torch_state_dict()
                # Удаляем до вызова super() — FedAvg требует ровно один ArrayRecord
                del rep.content["c_delta"]
                if c_delta_sum is None:
                    c_delta_sum = {k: torch.zeros_like(v) for k, v in delta_state.items()}
                for k, v in delta_state.items():
                    c_delta_sum[k] += v
                n_scaffold += 1

        # Обновляем c_server: c = c + (1/n) * sum(delta_c_i)
        if c_delta_sum is not None and n_scaffold > 0 and self._c_server is not None:
            c_state = self._c_server.to_torch_state_dict()
            self._c_server = ArrayRecord({
                k: c_state[k] + c_delta_sum[k] / n_scaffold
                for k in c_state
            })

        self.round_client_logs[server_round] = per_round
        self._agg_start_times[server_round] = time.time()
        result = super().aggregate_train(server_round, replies_list)
        self._agg_end_times[server_round] = time.time()
        return result

    def get_round_logs(self, server_round: int) -> Dict[int, Dict[str, Any]]:
        return self.round_client_logs.get(server_round, {})


# ── Registry ──────────────────────────────────────────────────────────────────

# Стратегии, требующие минимум 2 клиентов
_MIN_2 = {"fedmedian", "fedtrimmedavg", "krum", "multikrum", "bulyan"}

STRATEGY_REGISTRY: Dict[str, Tuple[Type, Any]] = {
    "fedavg":        (FedAvg,        FedAvgCfg()),
    "fedprox":       (FedProx,       FedProxCfg()),
    "fedavgm":       (FedAvgM,       FedAvgMCfg()),
    "fedadam":       (FedAdam,       FedAdamCfg()),
    "fedyogi":       (FedYogi,       FedYogiCfg()),
    "fedadagrad":    (FedAdagrad,    FedAdagradCfg()),
    "fedmedian":     (FedMedian,     FedMedianCfg()),
    "fedtrimmedavg": (FedTrimmedAvg, FedTrimmedAvgCfg()),
    "krum":          (Krum,          KrumCfg()),
    "multikrum":     (MultiKrum,     MultiKrumCfg()),
    "bulyan":        (Bulyan,        BulyanCfg()),
    "scaffold":      (ScaffoldStrategy, ScaffoldCfg()),
}


# ── LoggingStrategy wrapper ───────────────────────────────────────────────────

def _make_logging_cls(base_cls: Type) -> Type:
    """Оборачивает стратегию: перехватывает aggregate_train для сбора метрик клиентов."""

    class LoggingStrategy(base_cls):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.round_client_logs: Dict[int, Dict[int, Dict[str, Any]]] = {}
            self._agg_start_times: Dict[int, float] = {}
            self._agg_end_times:   Dict[int, float] = {}

        def aggregate_train(
            self, server_round: int, replies: Iterable[Message]
        ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            replies_list = list(replies)
            per_round: Dict[int, Dict[str, Any]] = {}

            for rep in replies_list:
                src = rep.metadata.src_node_id
                if not rep.has_content() or "metrics" not in rep.content:
                    continue
                m: MetricRecord = rep.content["metrics"]  # type: ignore[assignment]
                cid = int(m.get("partition-id", float(src)))
                per_round[cid] = {
                    "first_epoch_loss": float(m.get("first-epoch-loss", 0.0)),
                    "last_epoch_loss":  float(m.get("last-epoch-loss", 0.0)),
                    "round_time_sec":   float(m.get("round-time-sec", 0.0)),
                    "local_epochs":     int(m.get("local-epochs", 0)),
                    "num_examples":     int(m.get("num-examples", 0)),
                    "src_node_id":      int(src),
                    "drift":            float(m.get("drift", 0.0)),
                }

            self.round_client_logs[server_round] = per_round
            self._agg_start_times[server_round] = time.time()
            result = super().aggregate_train(server_round, replies_list)
            self._agg_end_times[server_round] = time.time()
            return result

        def get_round_logs(self, server_round: int) -> Dict[int, Dict[str, Any]]:
            return self.round_client_logs.get(server_round, {})

    return LoggingStrategy


# ── Public API ────────────────────────────────────────────────────────────────

def build_strategy(
    name: str,
    *,
    fraction_train: float,
    min_train_nodes: int,
    min_available_nodes: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Создать стратегию по имени с параметрами из реестра.

    Returns:
        (strategy_instance, params_dict) — экземпляр стратегии и dict её параметров
        для сохранения в артефакты эксперимента.
    """
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown aggregation '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )

    base_cls, cfg = STRATEGY_REGISTRY[key]
    params: Dict[str, Any] = dataclasses.asdict(cfg)

    if key in _MIN_2:
        min_train_nodes    = max(min_train_nodes, 2)
        min_available_nodes = max(min_available_nodes, 2)

    common = dict(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=0,
        min_available_nodes=min_available_nodes,
        weighted_by_key="num-examples",
        arrayrecord_key="arrays",
        configrecord_key="config",
    )

    # ScaffoldStrategy уже включает logging — не оборачиваем через _make_logging_cls
    if key == "scaffold":
        strategy = ScaffoldStrategy(**common, **params)
    else:
        LoggingCls = _make_logging_cls(base_cls)
        strategy = LoggingCls(**common, **params)
    return strategy, params
