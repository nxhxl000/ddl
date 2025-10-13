from __future__ import annotations

from pathlib import Path
import numpy as np
import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents

from src.fl_client import load_split
from fl_app.task import build_server_eval_loader, server_evaluate_fn


def server_fn(context: Context) -> ServerAppComponents:
    run_cfg = context.run_config
    rounds   = int(run_cfg.get("rounds", 5))
    cpr      = int(run_cfg.get("clients_per_round", 5))
    data_dir = str(run_cfg.get("data_dir", "data"))
    split    = Path(run_cfg["split_path"])

    # Централизованная валидация
    eval_fn = server_evaluate_fn(build_server_eval_loader(data_dir))

    # Не привязываемся жёстко к общему числу клиентов в рантайме
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cpr,
        min_evaluate_clients=cpr,
        min_available_clients=cpr,
        evaluate_fn=eval_fn,
        fit_metrics_aggregation_fn=lambda results: {},       
        evaluate_metrics_aggregation_fn=lambda results: {},
    )

    config = fl.server.ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)