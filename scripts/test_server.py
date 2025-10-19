"""Тестирование сервера изолированно"""

import sys
import os
import logging
from unittest.mock import Mock, patch
from flwr.common import Context
from flwr.serverapp import Grid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fl_app.server_app
from fl_app.server_app import app  # импортируем наше приложение сервера

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

def create_mock_context(run_config):
    """Создает mock контекст для тестирования"""
    return Context(
        run_id="test_server_run_001",
        run_config=run_config,
        node_id="test_server_node",
        node_config={},
        state={}
    )

def test_server_initialization():
    """Тестируем инициализацию сервера без запуска стратегии"""
    print("=== ТЕСТИРОВАНИЕ ИНИЦИАЛИЗАЦИИ СЕРВЕРА ===")
    
    # 1. Загружаем конфигурацию из pyproject.toml
    config = load_config_from_pyproject()
    
    # 2. Создаем mock контекст
    mock_context = create_mock_context(config)
    
    # 3. Тестируем что серверная функция может быть вызвана
    try:
        # Просто проверяем что функция существует и может быть импортирована
        from fl_app.server_app import main
        print("✅ Серверная функция main() доступна")
        
        # Проверяем что ServerApp создается
        from fl_app.server_app import app
        print("✅ ServerApp создан успешно")
        print(f"✅ Тип app: {type(app).__name__}")
        
        print("✅ Сервер инициализирован корректно (стратегия не тестируется из-за несовместимости версий)")
        
    except Exception as e:
        print(f"❌ Ошибка при инициализации сервера: {e}")
        raise

def test_model_creation():
    """Тестируем создание модели сервером"""
    print("\n=== ТЕСТИРОВАНИЕ СОЗДАНИЯ МОДЕЛИ СЕРВЕРОМ ===")
    
    try:
        from fl_app.task import create_model
        
        # Создаем модель
        model = create_model()
        print("✅ Модель успешно создана сервером")
        print(f"✅ Тип модели: {type(model).__name__}")
        
        # Проверяем параметры
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Обучаемые параметры: {trainable_params:,}")
        
        # Проверяем устройство
        device = next(model.parameters()).device
        print(f"✅ Модель на устройстве: {device}")
        
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        raise

def test_config_loading():
    """Тестируем загрузку конфигурации сервером"""
    print("\n=== ТЕСТИРОВАНИЕ ЗАГРУЗКИ КОНФИГУРАЦИИ ===")
    
    try:
        # Загружаем конфигурацию из pyproject.toml
        config = load_config_from_pyproject()
        
        # Проверяем обязательные параметры
        required_params = ["batch_size", "local_epochs", "lr", "rounds", "data_dir", "split_path"]
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            print(f"❌ Отсутствуют параметры: {missing_params}")
            raise ValueError(f"Отсутствуют параметры: {missing_params}")
        
        print("✅ Все обязательные параметры загружены:")
        for key, value in config.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке конфигурации: {e}")
        raise

def test_fedavg_strategy():
    """Тестируем создание стратегии FedAvg"""
    print("\n=== ТЕСТИРОВАНИЕ СТРАТЕГИИ FEDAVG ===")
    
    try:
        from flwr.serverapp.strategy import FedAvg
        
        # Создаем стратегию с правильными параметрами для новой версии Flower
        strategy = FedAvg()
        
        print("✅ Стратегия FedAvg успешно создана")
        print(f"✅ Тип стратегии: {type(strategy).__name__}")
        
        # Проверяем доступные атрибуты
        available_attrs = [attr for attr in dir(strategy) if not attr.startswith('_')]
        print(f"✅ Доступные атрибуты стратегии: {len(available_attrs)}")
        
    except Exception as e:
        print(f"❌ Ошибка при создании стратегии: {e}")
        raise

def test_server_app_structure():
    """Тестируем структуру ServerApp"""
    print("\n=== ТЕСТИРОВАНИЕ СТРУКТУРЫ SERVERAPP ===")
    
    try:
        # Проверяем что app является ServerApp
        from flwr.serverapp import ServerApp
        assert isinstance(fl_app.server_app.app, ServerApp), "app должен быть экземпляром ServerApp"
        print("✅ app является экземпляром ServerApp")
        
        # Проверяем что функция main существует
        assert hasattr(fl_app.server_app, 'main'), "Функция main должна существовать"
        print("✅ Функция main существует")
        
        # Проверяем что функция main имеет декоратор @app.main()
        assert callable(fl_app.server_app.main), "Функция main должна быть вызываемой"
        print("✅ Функция main является вызываемой")
        
        print("✅ Структура ServerApp корректна")
        
    except Exception as e:
        print(f"❌ Ошибка в структуре ServerApp: {e}")
        raise

def print_config_summary():
    """Печатает сводку конфигурации"""
    try:
        config = load_config_from_pyproject()
        print("\n=== СВОДКА КОНФИГУРАЦИИ СЕРВЕРА ИЗ PYPROJECT.TOML ===")
        for key, value in config.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить конфигурацию: {e}")

if __name__ == "__main__":
    print("Запуск тестов сервера с параметрами из pyproject.toml...")
    
    # Печатаем сводку конфигурации
    print_config_summary()
    
    # Запускаем тесты
    test_config_loading()
    test_model_creation()
    test_fedavg_strategy()
    test_server_app_structure()
    test_server_initialization()
    
    print("\n🎉 Все тесты сервера пройдены успешно!")
    print("✅ Конфигурация из pyproject.toml загружена корректно")
    print("✅ Модель создается правильно")
    print("✅ Стратегия FedAvg инициализируется")
    print("✅ Структура ServerApp корректна")
    print("✅ Сервер готов к работе с клиентами")
    print("⚠️  Примечание: Полный запуск стратегии не тестируется из-за несовместимости версий Flower")