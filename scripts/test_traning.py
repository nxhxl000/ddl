import logging
import sys
import os

# Добавляем абсолютный путь к папке fl_app в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fl_app.task

# Теперь можно импортировать нужные функции
from fl_app.task import load_split, get_client_data, create_model, fit_fn, evaluate_fn, setup_tensorboard

# Настроим логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_local_training_and_evaluation():
    # Путь к сплиту данных и папке с данными
    split_path = "splits/cifar10_iid_K10_seed42.json"  # Путь к вашему сплиту
    data_dir = "data"  # Папка с данными CIFAR-10

    # Инициализируем TensorBoard для тестирования
    writer = setup_tensorboard("test_training")

    # Загружаем данные
    logger.info("Загружаем данные для теста...")
    split_data = load_split(split_path)

    # Выбираем клиента для теста (например, клиент 0)
    client_id = 0
    train_loader, val_loader = get_client_data(client_id, split_data, data_dir)

    # Создаем модель
    logger.info("Создаем модель...")
    model = create_model()

    # Тестируем тренировку (fit_fn) с TensorBoard
    logger.info("Тестируем тренировку модели...")
    model = fit_fn(rnd=1, model=model, train_data=train_loader, val_data=val_loader, writer=writer)

    # Тестируем оценку модели (evaluate_fn)
    logger.info("Тестируем оценку модели...")
    accuracy = evaluate_fn(model, val_loader)
    logger.info(f"Точность на валидационных данных: {accuracy[1] * 100:.2f}%")

    # Закрываем writer
    writer.close()
    logger.info("Тестирование завершено. Для просмотра графиков запустите: tensorboard --logdir=runs")

if __name__ == "__main__":
    test_local_training_and_evaluation()