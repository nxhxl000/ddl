from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Нормализация (без аугментаций для чистой проверки)
    tf_train = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tf_test  = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    # Скачиваем/загружаем
    trainset = datasets.CIFAR10(str(data_dir), train=True,  download=True, transform=tf_train)
    testset  = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tf_test)

    print(f"Num classes: {len(CIFAR10_CLASSES)}")
    print(f"Train size: {len(trainset)}")
    print(f"Test size:  {len(testset)}")
    print("Classes:", list(CIFAR10_CLASSES))

    # Проверим распределение классов в train
    labels = trainset.targets if hasattr(trainset, "targets") else trainset.labels
    counts = [0]*10
    for y in labels:
        counts[int(y)] += 1
    for i, (name, cnt) in enumerate(zip(CIFAR10_CLASSES, counts)):
        print(f"[{i:02d}] {name:12s} -> {cnt}")

    # Пробный батч — форма и диапазон значений
    loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    x, y = next(iter(loader))
    assert x.ndim == 4 and x.shape[1:] == (3, 32, 32), f"Unexpected shape: {x.shape}"
    print("Sample batch shapes:", x.shape, y.shape)
    print("dtype:", x.dtype, "min:", float(x.min()), "max:", float(x.max()))
    print("\n✅ CIFAR-10 downloaded and basic checks passed.")

if __name__ == "__main__":
    main()