import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from pathlib import Path
import matplotlib.pyplot as plt

# Параметры CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функция для создания модели
def create_model():
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Адаптация для CIFAR-10 (32x32): Изменяем conv1 и удаляем maxpool
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Убираем maxpool для маленьких изображений
    
    # Упрощение dropout в backbone
    model.layer3[1].add_module('dropout', torch.nn.Dropout(0.7))
    model.layer4[1].add_module('dropout', torch.nn.Dropout(0.8))

    # Упрощённый классификатор
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.7),
        torch.nn.Linear(model.fc.in_features, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(128, 10)
    )

    # Инициализация весов
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    model = model.to(device)
    
    return model

# Загрузка тестового набора данных CIFAR-10
def load_test_data():
    data_dir = Path("data")

    # Трансформация для тестовых данных
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Загружаем тестовый датасет
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=False, transform=tf_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    return test_loader, test_set

# Функция для тестирования модели
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    wrong_examples = []  # Список для хранения примеров ошибок

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Сохраняем ошибки
            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                wrong_examples.append((inputs[idx], labels[idx], predicted[idx]))

    accuracy = 100 * correct / total
    print(f'Accuracy on the test dataset: {accuracy:.2f}%')
    
    return accuracy, wrong_examples[:5]  # Возвращаем точность и первые 5 ошибок

# Загрузка модели и веса
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

# Главная функция
def main():
    # Загрузить тестовые данные
    test_loader, test_set = load_test_data()

    # Создание модели и загрузка сохранённых весов
    model = create_model()
    model_path = "/home/nxhxl/work/ddl/experements/cifar10_dirichlet_a0.07_K10_seed42/final_model_cifar10_dirichlet_a0.07_K10_seed42.pth"
    model = load_model(model, model_path)

    # Тестирование модели
    accuracy, wrong_examples = test_model(model, test_loader)

    # Отображение 5 примеров, где модель ошиблась
    if wrong_examples:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, (image, label, pred) in enumerate(wrong_examples):
            ax = axes[i]
            image = image.cpu().detach().numpy().transpose(1, 2, 0)  # Переносим на CPU и преобразуем в NumPy
            ax.imshow(image)  # Теперь можно безопасно отобразить
            ax.set_title(f"True: {test_set.classes[label]}, Pred: {test_set.classes[pred]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
