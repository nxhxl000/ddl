"""CIFAR-10 Federated Learning: Client App"""

import torch
from flwr.clientapp import ClientApp
from flwr.common import Context, Message, RecordDict, ArrayRecord, MetricRecord
import logging
import fl_app.task

# ✅ ИМПОРТИРУЕМ ВАШУ ФУНКЦИЮ load_split ИЗ task.py
from fl_app.task import create_model, get_client_data, train_model, evaluate_model, load_split

# Настройка для предотвращения дубликатов
root_logger = logging.getLogger()  # Root logger
flwr_logger = logging.getLogger('flwr')  # Logger Flower

# Очистить handlers root-логгера, чтобы избежать дубликатов (как в рекомендациях Flower для Colab)
root_logger.handlers = []

# Отключить пропагацию от flwr-логгера к root (предотвращает дубли)
flwr_logger.propagate = False

# Продолжайте с basicConfig (это добавит новый handler только для вашего app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = ClientApp()

# Глобальные переменные для кэширования данных
_split_data = None
_data_dir = None
_split_path = None

def _load_client_data(context: Context):
    """Вспомогательная функция для загрузки данных клиента один раз."""
    global _split_data, _data_dir, _split_path
    
    if _split_data is None:
        # ✅ ПОЛУЧАЕМ ПУТИ ИЗ PYPROJECT.TOML БЕЗ ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ
        _split_path = context.run_config.get("split_path")
        _data_dir = context.run_config.get("data_dir")
        
        # ✅ ПРОВЕРЯЕМ ЧТО ПАРАМЕТРЫ НАЙДЕНЫ, ИНАЧЕ ОШИБКА
        if _split_path is None:
            raise ValueError(
                "Параметр 'split_path' не найден в pyproject.toml. "
                "Добавьте в раздел [tool.flwr.app.config]: "
                "split_path = 'splits/cifar10_iid_K10_seed42.json'"
            )
        
        if _data_dir is None:
            raise ValueError(
                "Параметр 'data_dir' не найден в pyproject.toml. "
                "Добавьте в раздел [tool.flwr.app.config]: "
                "data_dir = 'data'"
            )
        
        logger.info(f"Загрузка split_data из: {_split_path}")
        logger.info(f"Директория с данными: {_data_dir}")
        
        # ✅ ИСПОЛЬЗУЕМ ВАШУ ФУНКЦИЮ load_split ДЛЯ ЗАГРУЗКИ SPLIT_DATA
        _split_data = load_split(_split_path)
    
    return _split_data, _data_dir, _split_path

@app.train()
def train(message: Message, context: Context):
    """Функция обучения на локальных данных клиента."""
    
    # 1. Получаем ID клиента из контекста
    client_id = context.node_config["partition-id"]
    logger.info(f"Клиент {client_id}: начало обучения")
    
    # 2. Загружаем данные клиента
    split_data, data_dir, split_path = _load_client_data(context)
    
    # 3. Загружаем модель и веса от сервера
    model = create_model()
    server_weights = message.content["arrays"].to_torch_state_dict()
    model.load_state_dict(server_weights)
    model.to(device)
    
    # 4. Логируем конфигурацию от сервера
    logger.info(f"Получена конфигурация от сервера: {message.content['config']}")
    
    # 5. Получаем параметры обучения
    batch_size = message.content["config"].get("batch_size", 64)
    local_epochs = message.content["config"].get("local_epochs", 5)
    learning_rate = message.content["config"].get("lr", 0.0004)
    current_round = message.content["config"].get("current_round", 1)
    
    logger.info(f"Клиент {client_id}: Параметры обучения - batch_size={batch_size}, local_epochs={local_epochs}, lr={learning_rate}, round={current_round}")
    
    # 6. Загружаем данные клиента для тренировки и валидации
    train_loader, val_loader = get_client_data(
        client_id=client_id,
        split_data=split_data,
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    logger.info(f"Клиент {client_id}: загружено {len(train_loader.dataset)} тренировочных примеров")
    
    # 7. Обучаем модель
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=local_epochs,
        lr=learning_rate,
        batch_size=batch_size,
        round_num=current_round
    )
    
    # 8. Рассчитываем метрики для возвращаемого результата
    train_loss, train_accuracy = evaluate_model(trained_model, train_loader)
    
    # 9. Формируем метрики
    metrics = {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "num-examples": len(train_loader.dataset),
    }
    
    updated_weights = ArrayRecord(trained_model.state_dict())
    
    content = RecordDict({
        "arrays": updated_weights,
        "metrics": MetricRecord(metrics)  # Верно возвращаем метрики
    })
    
    logger.info(f"Клиент {client_id}: обучение завершено")
    return Message(content=content, reply_to=message)

@app.evaluate()
def evaluate(message: Message, context: Context):
    """Функция оценки на локальных данных клиента."""
    
    client_id = context.node_config["partition-id"]
    logger.info(f"Клиент {client_id}: начало оценки")
    
    # 1. ✅ ЗАГРУЖАЕМ ДАННЫЕ ИСПОЛЬЗУЯ ВАШУ ФУНКЦИЮ load_split
    split_data, data_dir, split_path = _load_client_data(context)
    
    # 2. Загружаем модель и веса от сервера
    model = create_model()
    server_weights = message.content["arrays"].to_torch_state_dict()
    model.load_state_dict(server_weights)
    model.to(device)
    
    # 3. Логируем параметры из конфига (только batch_size)
    batch_size = message.content["config"].get("batch_size", 64)  # Значение по умолчанию — 64
    logger.info(f"Клиент {client_id}: Параметры оценки - batch_size={batch_size}")
    
    # 4. ✅ ЗАГРУЖАЕМ ДАННЫЕ КЛИЕНТА ИСПОЛЬЗУЯ ВАШУ ФУНКЦИЮ get_client_data
    _, val_loader = get_client_data(
        client_id=client_id,
        split_data=split_data,      # загружено через load_split из pyproject.toml
        data_dir=data_dir,          # из pyproject.toml
        batch_size=batch_size
    )
    
    # 5. Выполняем оценку используя вашу функцию evaluate_model
    val_loss, val_accuracy = evaluate_model(model, val_loader)
    
    metrics = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "num-examples": len(val_loader.dataset),
    }
    
    content = RecordDict({
        "metrics": MetricRecord(metrics)
    })
    
    logger.info(f"Клиент {client_id}: оценка завершена - Accuracy: {val_accuracy*100:.2f}%")
    
    return Message(content=content, reply_to=message)

if __name__ == "__main__":
    logger.info("ClientApp инициализирован и готов к работе")
    logger.info(f"Используется устройство: {device}")