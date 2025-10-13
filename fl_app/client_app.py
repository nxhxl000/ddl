from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from flwr.common import Context
from flwr.client import ClientApp

from src.fl_client import CifarClient, load_split


def client_fn(context: Context):
    run_cfg = context.run_config
    split_path = Path(run_cfg["split_path"])
    lr         = float(run_cfg.get("lr", 0.01))
    local_ep   = int(run_cfg.get("local_epochs", 1))
    batch_size = int(run_cfg.get("batch_size", 64))

    clients = load_split(split_path)  # List[List[int]]
    n = len(clients)

    # Маппим произвольный node_id в [0..n-1] устойчиво
    try:
        cid = int(context.node_id)
        if cid < 0 or cid >= n:
            raise ValueError
    except Exception:
        h = hashlib.sha1(str(context.node_id).encode()).hexdigest()
        cid = int(h, 16) % n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np_client = CifarClient(
        cid=str(cid),
        indices=clients[cid],
        device=device,
        lr=lr,
        local_epochs=local_ep,
        batch_size=batch_size,
    )
    # Новый рантайм требует вернуть Client, а не NumpyClient
    return np_client.to_client()


client_app = ClientApp(client_fn=client_fn)