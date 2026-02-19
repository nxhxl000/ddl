from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision import models as tv_models
except Exception as e:
    raise RuntimeError(
        "torchvision is required for this script. Install torchvision and try again."
    ) from e


# -------------------------
# Model builders
# -------------------------

def build_simple_cnn(in_channels: int, num_classes: int = 10) -> nn.Module:
    """Простой CNN похожий на твой текущий (подходит для MNIST/CIFAR)."""
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.gap(x).flatten(1)
            return self.fc(x)

    return Net()


def build_lenet5(num_classes: int = 10) -> nn.Module:
    """Классический LeNet-5 (для MNIST)."""
    class LeNet5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28->24
            self.pool = nn.AvgPool2d(2, 2)                 # 24->12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12->8
            # 8->4 after pool
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.tanh(self.conv1(x))
            x = self.pool(x)
            x = torch.tanh(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)

    return LeNet5()


def build_resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet18, адаптированный под CIFAR (32x32):
    - conv1: 3x3 stride 1
    - убираем maxpool
    - fc на num_classes
    """
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)

    # conv1 for CIFAR (3x3, stride=1, padding=1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    """
    Заменяет первый conv слой у MobileNetV3 (и похожих),
    сохраняя out_channels/stride/padding.
    """
    # Для mobilenet_v3_small первый conv обычно: model.features[0][0]
    first = model.features[0][0]
    if not isinstance(first, nn.Conv2d):
        raise RuntimeError("Unexpected MobileNetV3 structure: first layer is not Conv2d")

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=(first.bias is not None),
        padding_mode=first.padding_mode,
    )
    model.features[0][0] = new_conv


def build_mobilenet_v3_small(num_classes: int = 10, pretrained: bool = False, in_channels: int = 3) -> nn.Module:
    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.mobilenet_v3_small(weights=weights)
    _replace_first_conv(model, in_channels=in_channels)
    # classifier last layer -> num_classes
    if isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear):
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        else:
            raise RuntimeError("Unexpected classifier structure in MobileNetV3")
    else:
        raise RuntimeError("Unexpected classifier type in MobileNetV3")
    return model


# -------------------------
# Saving utilities
# -------------------------

def _meta(dataset: str, model_id: str, pretrained: bool, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "dataset": str(dataset),
        "model_id": str(model_id),
        "pretrained_requested": bool(pretrained),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),  # <-- важно
        "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),  # <-- важно
    }
    if extra:
        # extra у тебя и так строки/булевы — но на всякий оставим как есть
        m.update(extra)
    return m


def _save_bundle(path: Path, model: nn.Module, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(bundle, path)
    # ещё и json-метаданные рядом (удобно смотреть без torch.load)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_build(build_fn, *args, **kwargs) -> Tuple[nn.Module, bool, str]:
    """
    Возвращает (model, used_pretrained, note).
    Если pretrained=True и не получилось скачать — создаём random init.
    """
    pretrained = bool(kwargs.get("pretrained", False))
    try:
        model = build_fn(*args, **kwargs)
        return model, pretrained, "ok"
    except Exception as e:
        if pretrained:
            # fallback to random init
            kwargs["pretrained"] = False
            model = build_fn(*args, **kwargs)
            return model, False, f"pretrained failed -> fallback to random init: {type(e).__name__}: {e}"
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Save base CNN models for CIFAR and MNIST into ./models")
    p.add_argument("--out", type=str, default="models", help="Output folder (default: models)")
    p.add_argument("--pretrained", action="store_true", help="Try to use pretrained ImageNet weights where available")
    args = p.parse_args()

    out_dir = Path(args.out)
    pretrained = bool(args.pretrained)

    # CIFAR-10 models
    cifar_specs = [
        ("simple_cnn", lambda: build_simple_cnn(in_channels=3, num_classes=10)),
        ("resnet18_cifar", lambda: build_resnet18_cifar(num_classes=10, pretrained=pretrained)),
        ("mobilenet_v3_small", lambda: build_mobilenet_v3_small(num_classes=10, pretrained=pretrained, in_channels=3)),
    ]

    # MNIST models
    mnist_specs = [
        ("simple_cnn", lambda: build_simple_cnn(in_channels=1, num_classes=10)),
        ("lenet5", lambda: build_lenet5(num_classes=10)),
        ("mobilenet_v3_small_1ch", lambda: build_mobilenet_v3_small(num_classes=10, pretrained=pretrained, in_channels=1)),
    ]

    saved = []

    # Save CIFAR
    for model_id, ctor in cifar_specs:
        model, used_pretrained, note = _try_build(lambda: ctor())
        path = out_dir / "cifar" / f"{model_id}.pt"
        meta = _meta("cifar", model_id, pretrained, extra={"used_pretrained": used_pretrained, "note": note})
        _save_bundle(path, model, meta)
        saved.append(str(path))

    # Save MNIST
    for model_id, ctor in mnist_specs:
        model, used_pretrained, note = _try_build(lambda: ctor())
        path = out_dir / "mnist" / f"{model_id}.pt"
        meta = _meta("mnist", model_id, pretrained, extra={"used_pretrained": used_pretrained, "note": note})
        _save_bundle(path, model, meta)
        saved.append(str(path))

    print("Saved models:")
    for s in saved:
        print("  -", s)
    print(f"\nDone. Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
