import os
import json
import torch
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.common import Context, ArrayRecord, ConfigRecord
import logging
import fl_app.task

from fl_app.task import create_model

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

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    logger.info("=== ЗАПУСК СЕРВЕРА FEDERATED LEARNING ===")
    
    # 1. Читаем конфигурацию из pyproject.toml
    batch_size = context.run_config.get("batch_size", 64)
    local_epochs = context.run_config.get("local_epochs", 5)
    lr = context.run_config.get("lr", 0.0004)
    rounds = context.run_config.get("rounds", 20)
    
    # Путь к сплиту данных
    split_path = context.run_config.get("split_path", "splits/cifar10_iid_K10_seed42.json")
    split_name = split_path.split("/")[-1].split(".")[0]  # Извлекаем название сплита
    
    # Логируем конфигурацию на сервере
    logger.info(f"Конфигурация сервера:")
    logger.info(f"  - Раундов: {rounds}")
    logger.info(f"  - Локальные эпохи: {local_epochs}")
    logger.info(f"  - Learning rate: {lr}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Split Path: {split_path}")
    
    # 2. Загружаем глобальную модель
    logger.info("Инициализация глобальной модели...")
    global_model = create_model()
    arrays = ArrayRecord(global_model.state_dict())
    logger.info(f"Глобальная модель создана: {type(global_model).__name__}")
    
    # 3. Инициализируем стратегию FedAvg с правильными параметрами
    fraction_train = context.run_config.get("fraction_train", 0.5)  # Переименуйте переменную для ясности, если нужно
    logger.info(f"Используем fraction_train = {fraction_train} для стратегии FedAvg")

    # Для новой версии Flower параметры передаются напрямую
    strategy = FedAvg(
        fraction_train=fraction_train,  # 50% клиентов для обучения (было fraction_fit)
        fraction_evaluate=0.5,          # 50% клиентов для оценки
        min_train_nodes=5,              # Минимум 5 клиентов для обучения (было min_fit_clients)
        min_evaluate_nodes=5,           # Минимум 5 клиентов для оценки (было min_evaluate_clients)
        min_available_nodes=8,          # Ждем минимум 8 доступных клиентов (было min_available_clients)
    )
    logger.info(f"Стратегия FedAvg инициализирована с fraction_train={fraction_train}")
    
    # 4. Определяем конфигурацию для оценки
    evaluate_config = ConfigRecord({
        "batch_size": batch_size,
        "local_epochs": local_epochs, 
        "lr": lr,
        "current_round": 1,
    })
    
    # Логируем конфигурацию для оценки
    logger.info(f"Конфигурация для оценки: {evaluate_config}")
    
    # 5. Запускаем федеративное обучение
    logger.info(f"Запуск федеративного обучения на {rounds} раундов...")
    logger.info("Передача конфигурации для обучения...")
    
    # Создаем конфигурацию для тренировки
    train_config = ConfigRecord({
        "batch_size": batch_size,
        "local_epochs": local_epochs, 
        "lr": lr
    })
    logger.info(f"Конфигурация тренировки: {train_config}")
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        evaluate_config=evaluate_config,
        num_rounds=rounds,
    )
    logger.info("Федеративное обучение завершено!")
    
    # 6. Сохраняем финальную модель
    logger.info("Сохранение финальной модели...")
    try:
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pth")
        logger.info("✅ Финальная модель сохранена в: final_model.pth")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении модели: {e}")
    
    # 7. Логируем и сохраняем результаты
    logger.info("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    
    aggregated_metrics = {}  # dict[round: dict[metric: value]]
    
    # Извлекаем train-метрики для всех раундов
    if result.train_metrics_clientapp:
        for round_num, train_m in result.train_metrics_clientapp.items():
            aggregated_metrics[round_num] = aggregated_metrics.get(round_num, {})
            # Парсим str в float
            aggregated_metrics[round_num]["train_loss"] = float(train_m.get("train_loss", "0"))
            aggregated_metrics[round_num]["train_accuracy"] = float(train_m.get("train_accuracy", "0"))
            # Логируем только train-метрики
            logger.info(f"Раунд {round_num}: Train metrics - {{'train_loss': {aggregated_metrics[round_num]['train_loss']}, 'train_accuracy': {aggregated_metrics[round_num]['train_accuracy']}}}")

    # Извлекаем eval-метрики для всех раундов
    if result.evaluate_metrics_clientapp:
        for round_num, eval_m in result.evaluate_metrics_clientapp.items():
            aggregated_metrics[round_num] = aggregated_metrics.get(round_num, {})
            aggregated_metrics[round_num]["val_loss"] = float(eval_m.get("val_loss", "0"))
            aggregated_metrics[round_num]["val_accuracy"] = float(eval_m.get("val_accuracy", "0"))
            # Логируем только val-метрики (без дублирования train)
            logger.info(f"Раунд {round_num}: Eval metrics - {{'val_loss': {aggregated_metrics[round_num]['val_loss']}, 'val_accuracy': {aggregated_metrics[round_num]['val_accuracy']}}}")
    
    if aggregated_metrics:
        # Создаем папку для результатов
        results_dir = f"runs/{split_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        aggregated_metrics_path = os.path.join(results_dir, "aggregated_metrics.json")
        
        try:
            with open(aggregated_metrics_path, "w") as f:
                json.dump(aggregated_metrics, f, indent=4)
            logger.info(f"✅ Метрики сохранены в: {aggregated_metrics_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении метрик в JSON: {e}")
    else:
        logger.warning("Метрики не были найдены в результате.")

    logger.info("Сервер завершил работу")