# food101_effnet/config.py
import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str = "data_food"
    run_dir: str = "runs"
    exp_name: str = "baseline_lite0"

    # model
    model_name: str = "tf_efficientnet_lite0"
    pretrained: bool = True
    num_classes: int = 101

    # optimization
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    nesterov: bool = True
    amp: bool = True

    # regularization
    label_smoothing: float = 0.0

    # data
    img_size: int = 224
    resize_short: int = 256
    num_workers: int = 4
    aug_train: bool = False
    aug_preset: str = "basic"

    # device & misc
    device: str = "auto"
    force_device: bool = False
    seed: int = 42
    save_every: int = 0
    cudnn_benchmark: bool = True


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("EfficientNet-Lite0 on Food-101 — training")

    p.add_argument("--data_dir", type=str, default="data_food")
    p.add_argument("--run_dir", type=str, default="runs")
    p.add_argument("--exp_name", type=str, default="baseline_lite0")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")

    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    # regularization
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropyLoss",
    )

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--resize_short", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--aug_train", action="store_true")
    p.add_argument(
        "--aug_preset",
        type=str,
        default="basic",
        choices=["basic", "geo", "color", "erasing", "auto", "rand"],
    )

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--force_device", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=0)

    args = p.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov,
        amp=args.amp,
        label_smoothing=args.label_smoothing,
        img_size=args.img_size,
        resize_short=args.resize_short,
        num_workers=args.num_workers,
        aug_train=args.aug_train,
        aug_preset=args.aug_preset,
        device=args.device,
        force_device=args.force_device,
        seed=args.seed,
        save_every=args.save_every,
    )
