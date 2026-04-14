"""Локальное обучение + серверная evaluate (чистый PyTorch, без I/O)."""

from __future__ import annotations

import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _flat_norm(tensors) -> float:
    sq = 0.0
    for t in tensors:
        sq += float(t.detach().pow(2).sum().cpu())
    return sq ** 0.5


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    device: torch.device,
    proximal_mu: float = 0.0,                           # > 0 → FedProx
    c_server: dict[str, torch.Tensor] | None = None,    # SCAFFOLD
    c_client: dict[str, torch.Tensor] | None = None,    # SCAFFOLD
) -> dict:
    """Один раунд локального обучения.

    Возвращает: loss_first, loss_last, num_examples, t_compute,
                w_drift, update_norm_rel, grad_norm_last
    Плюс при SCAFFOLD: c_new, c_delta
    """
    scaffold = c_server is not None and c_client is not None
    if scaffold:
        momentum = 0.0  # Option I требует plain SGD

    model.to(device).train()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Стартовые веса — для proximal / SCAFFOLD / диагностики drift
    init_w = {n: p.detach().clone() for n, p in model.named_parameters()}
    init_norm = _flat_norm(init_w.values())

    per_epoch_loss: list[float] = []
    num_examples = 0
    grad_norm_last = 0.0
    t0 = time.perf_counter()

    use_amp = device.type == "cuda"
    for epoch in range(epochs):
        loss_sum, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), y)
                if proximal_mu > 0:
                    prox = sum(((p - init_w[nm]) ** 2).sum()
                               for nm, p in model.named_parameters())
                    loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            if scaffold:
                for nm, p in model.named_parameters():
                    if p.grad is None or nm not in c_server:
                        continue
                    p.grad.add_(c_server[nm].to(device) - c_client[nm].to(device))
            # Последняя норма градиента (до шага optimizer)
            grad_norm_last = _flat_norm(
                p.grad for p in model.parameters() if p.grad is not None
            )
            opt.step()
            bs = y.size(0)
            loss_sum += loss.item() * bs
            n += bs
        per_epoch_loss.append(loss_sum / max(n, 1))
        if epoch == 0:
            num_examples = n

    t_compute = time.perf_counter() - t0

    # Drift / update norm
    diffs = [p.detach() - init_w[nm].to(p.device) for nm, p in model.named_parameters()]
    w_drift = _flat_norm(diffs)
    update_norm_rel = w_drift / max(init_norm, 1e-12)

    result: dict = {
        "loss_first": per_epoch_loss[0],
        "loss_last": per_epoch_loss[-1],
        "num_examples": num_examples,
        "t_compute": t_compute,
        "w_drift": w_drift,
        "update_norm_rel": update_norm_rel,
        "grad_norm_last": grad_norm_last,
    }

    if scaffold:
        K = max(len(loader) * epochs, 1)
        c_new = {
            nm: c_client[nm] - c_server[nm]
                + (init_w[nm].cpu() - p.detach().cpu()) / (K * lr)
            for nm, p in model.named_parameters()
            if nm in c_server
        }
        result["c_new"] = c_new
        result["c_delta"] = {nm: c_new[nm] - c_client[nm] for nm in c_new}

    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate: loss, acc, macro-f1."""
    model.to(device).eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    ys, ps = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ys.append(y.cpu()); ps.append(pred.cpu())
    y_all = torch.cat(ys).numpy(); p_all = torch.cat(ps).numpy()
    num_classes = int(y_all.max()) + 1
    per_class = []
    for c in range(num_classes):
        mask = y_all == c
        per_class.append(float((p_all[mask] == c).sum() / max(mask.sum(), 1)))
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "f1_macro": float(f1_score(y_all, p_all, average="macro", zero_division=0)),
        "per_class": per_class,
    }
