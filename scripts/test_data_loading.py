import json
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import logging

# Настроим логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Загрузка кастомного сплита
def load_split(split_path: str) -> dict:
    """Загружает сплит, сохраненный в JSON."""
    logger.info(f"Загружаем сплит данных из: {split_path}")
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    logger.info("Сплит данных успешно загружен.")
    return split_data

# Загрузка данных по индексу для конкретного клиента с разделением на train и val
def get_client_data(client_id: int, split_data: dict, data_dir: str, batch_size: int = 64, val_ratio: float = 0.1):
    """Загружает данные для конкретного клиента и разделяет на train и val."""
    logger.info(f"Загружаем данные для клиента {client_id}...")
    client_indices = split_data["indices"][client_id]  # Получаем индексы для клиента
    
    # Трансформации данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Убедимся, что датасет уже загружен в папку
    try:
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        logger.info(f"Датасет CIFAR-10 успешно загружен из папки {data_dir}.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета CIFAR-10: {e}")
        return

    # Субсет данных для клиента
    client_data = Subset(trainset, client_indices)
    
    # Разделяем данные на train и val (90/10)
    train_size = int((1 - val_ratio) * len(client_data))
    val_size = len(client_data) - train_size
    train_data, val_data = random_split(client_data, [train_size, val_size])

    # DataLoaders для train и val
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    logger.info(f"Для клиента {client_id} загружены данные: {len(train_data)} для тренировки и {len(val_data)} для валидации.")
    
    return train_loader, val_loader

# Тестирование загрузки сплита и данных
def test_data_loading(split_path, data_dir, client_id=0):
    logger.info(f"Запуск теста загрузки данных для клиента {client_id}...")
    split_data = load_split(split_path)
    train_loader, val_loader = get_client_data(client_id, split_data, data_dir)

    # Проверим размер батчей для train и val
    logger.info(f"Размер тренировочного набора: {len(train_loader.dataset)}")
    logger.info(f"Размер валидационного набора: {len(val_loader.dataset)}")

    # Пройдемся по первым батчам данных для проверки
    for i, (inputs, labels) in enumerate(train_loader):
        logger.info(f"Пример {i+1} в тренировочном наборе: Размер батча: {inputs.size()} | Метки: {labels[:5]}")
        if i == 1:  # Показываем первые 2 батча
            break
    
    for i, (inputs, labels) in enumerate(val_loader):
        logger.info(f"Пример {i+1} в валидационном наборе: Размер батча: {inputs.size()} | Метки: {labels[:5]}")
        if i == 1:  # Показываем первые 2 батча
            break

if __name__ == "__main__":
    # Путь к сплиту данных и папке с данными
    split_path = "splits/cifar10_iid_K10_seed42.json"  # Путь к вашему сплиту
    data_dir = "data"  # Папка с данными CIFAR-10

    test_data_loading(split_path, data_dir)
