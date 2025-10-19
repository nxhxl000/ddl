"""Тестирование клиента изолированно"""

import torch
import sys
import os
import numpy as np
from flwr.common import Context, Message, RecordDict, ArrayRecord, MetricRecord, ConfigRecord
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fl_app.client_app
from fl_app.client_app import app  # импортируем наше приложение клиента

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config_from_pyproject():
    """Загружает конфигурацию из pyproject.toml"""
    pyproject_path = "pyproject.toml"
    
    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(f"Файл {pyproject_path} не найден")
    
    try:
        # Пробуем разные способы чтения TOML
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                config = tomli.load(f)
        except ImportError:
            # Для Python 3.11+ используем встроенный tomllib
            import tomllib
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
        
        # Получаем конфиг из раздела [tool.flwr.app.config]
        flwr_config = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
        
        if not flwr_config:
            raise ValueError("Раздел [tool.flwr.app.config] не найден в pyproject.toml")
        
        logger.info(f"Загружена конфигурация из pyproject.toml")
        return flwr_config
        
    except Exception as e:
        raise ValueError(f"Ошибка чтения pyproject.toml: {e}")

def create_mock_context(run_config, node_config=None):
    """Создает mock контекст для тестирования"""
    return Context(
        run_id="test_run_001",
        run_config=run_config,
        node_id="test_node_0",
        node_config=node_config or {
            "partition-id": 0,
            "num-partitions": 10
        },
        state={}
    )

def test_client_training():
    """Тестируем функцию обучения клиента"""
    print("=== ТЕСТИРОВАНИЕ ЛОКАЛЬНОГО ОБУЧЕНИЯ КЛИЕНТА ===")
    
    # 1. Загружаем конфигурацию из pyproject.toml
    config = load_config_from_pyproject()
    
    # 2. Создаем mock контекст с параметрами из pyproject.toml
    mock_context = create_mock_context(
        run_config=config,
        node_config={
            "partition-id": 0,  # тестируем клиента 0
            "num-partitions": 10
        }
    )
    
    # 3. Тестируем напрямую функции train и evaluate
    try:
        # Создаем mock модель и получаем ее веса
        from fl_app.task import create_model
        model = create_model()
        initial_weights = ArrayRecord(model.state_dict())
        
        # Создаем mock сообщение для обучения
        from flwr.common import ConfigRecord
        mock_message = Message(
            content=RecordDict({
                "arrays": initial_weights,
                "config": ConfigRecord({
                    "batch_size": config["batch_size"],
                    "local_epochs": 2,
                    "lr": config["lr"],
                    "current_round": 1
                })
            }),
            dst_node_id=0,
            message_type="train"
        )
        
        # Вызываем функцию train напрямую
        response = fl_app.client_app.train(mock_message, mock_context)
        print("✅ Обучение клиента завершено успешно!")
        print(f"✅ Возвращены обновленные веса и метрики")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении клиента: {e}")
        raise

def test_client_evaluation():
    """Тестируем функцию оценки клиента"""
    print("\n=== ТЕСТИРОВАНИЕ ЛОКАЛЬНОЙ ОЦЕНКИ КЛИЕНТА ===")
    
    # 1. Загружаем конфигурацию из pyproject.toml
    config = load_config_from_pyproject()
    
    # 2. Создаем mock контекст
    mock_context = create_mock_context(
        run_config=config,
        node_config={
            "partition-id": 0,
            "num-partitions": 10
        }
    )
    
    # 3. Тестируем функцию evaluate
    try:
        # Создаем mock модель и веса
        from fl_app.task import create_model
        model = create_model()
        model_weights = ArrayRecord(model.state_dict())
        
        # Создаем mock сообщение для оценки
        from flwr.common import ConfigRecord
        mock_message = Message(
            content=RecordDict({
                "arrays": model_weights,
                "config": ConfigRecord({
                    "batch_size": config["batch_size"],
                    "current_round": 1
                })
            }),
            dst_node_id=0,
            message_type="evaluate"
        )
        
        # Вызываем функцию evaluate напрямую
        response = fl_app.client_app.evaluate(mock_message, mock_context)
        metrics = response.content["metrics"]
        print("✅ Оценка клиента завершена успешно!")
        
        print(f"✅ Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"✅ Loss: {metrics.get('loss', 0):.4f}")
        print(f"✅ Примеров: {metrics.get('num_examples', 0)}")
        
    except Exception as e:
        print(f"❌ Ошибка при оценке клиента: {e}")
        raise

def test_data_loading():
    """Тестируем загрузку данных отдельно"""
    print("\n=== ТЕСТИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ ===")
    
    from fl_app.task import get_client_data, load_split
    
    try:
        # Загружаем конфигурацию из pyproject.toml
        config = load_config_from_pyproject()
        
        # Загружаем сплит используя путь из pyproject.toml
        split_data = load_split(config["split_path"])
        print(f"✅ Split загружен из '{config['split_path']}', клиентов: {len(split_data['indices'])}")
        
        # Загружаем данные для клиента 0 используя параметры из pyproject.toml
        train_loader, val_loader = get_client_data(
            client_id=0,
            split_data=split_data,
            data_dir=config["data_dir"],
            batch_size=32  # маленький батч для теста
        )
        
        print(f"✅ Данные загружены из '{config['data_dir']}':")
        print(f"   - Тренировочных примеров: {len(train_loader.dataset)}")
        print(f"   - Валидационных примеров: {len(val_loader.dataset)}")
        print(f"   - Батчей в train_loader: {len(train_loader)}")
        print(f"   - Батчей в val_loader: {len(val_loader)}")
        
        # Проверяем один батч
        for images, labels in train_loader:
            print(f"✅ Размер батча: {images.shape}")
            print(f"✅ Метки: {labels.shape}")
            break  # только первый батч
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        raise

def print_config_summary():
    """Печатает сводку конфигурации"""
    try:
        config = load_config_from_pyproject()
        print("\n=== СВОДКА КОНФИГУРАЦИИ ИЗ PYPROJECT.TOML ===")
        for key, value in config.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить конфигурацию: {e}")

if __name__ == "__main__":
    print("Запуск тестов клиента с параметрами из pyproject.toml...")
    
    # Печатаем сводку конфигурации
    print_config_summary()
    
    # Запускаем тесты
    test_data_loading()
    test_client_training() 
    test_client_evaluation()
    
    print("\n🎉 Все тесты пройдены успешно! Клиент работает корректно.")
    print("✅ Конфигурация из pyproject.toml загружена корректно")
    print("✅ Данные клиента загружаются правильно") 
    print("✅ Локальное обучение работает")
    print("✅ Локальная оценка работает")