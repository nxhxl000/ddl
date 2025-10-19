import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path

# Константы CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def main():
    # Путь к данным
    data_dir = Path("data")

    # Трансформация для тестовых данных
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Загружаем тестовый датасет из уже скачанных данных
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=False, transform=tf_test)

    # Извлекаем первые 10 изображений
    images, labels = zip(*[test_set[i] for i in range(10)])

    # Отображаем изображения
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))  # Переупорядочиваем каналы (C, H, W) -> (H, W, C)
        ax.set_title(f"Label: {test_set.classes[labels[i]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
