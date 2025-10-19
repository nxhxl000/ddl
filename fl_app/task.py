import json
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import flwr as fl
from collections import defaultdict
import numpy as np
import logging
import os
from datetime import datetime

# Настроим логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используем {'GPU' if torch.cuda.is_available() else 'CPU'} для тренировки.")

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
    """Загружаем данные для конкретного клиента с аугментацией."""
    logger.info(f"Загружаем данные для клиента {client_id}...")
    client_indices = split_data["indices"][client_id]  # Получаем индексы для клиента
    
    # Трансформации данных с аугментацией
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Загрузка датасета CIFAR-10
    try:
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        logger.info(f"Датасет CIFAR-10 успешно загружен из папки {data_dir}.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета CIFAR-10: {e}")
        return

    # Субсет данных для клиента
    client_data = Subset(trainset, client_indices)
    
    # Разделяем данные на train и val
    train_size = int((1 - val_ratio) * len(client_data))
    val_size = len(client_data) - train_size
    train_data, val_data = random_split(client_data, [train_size, val_size])

    # DataLoaders для train и val
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    logger.info(f"Для клиента {client_id} загружены данные: {len(train_data)} для тренировки и {len(val_data)} для валидации.")
    
    return train_loader, val_loader

# Используем ResNet-18, адаптированный для CIFAR-10
def create_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Адаптация для CIFAR-10 (32x32): Изменяем conv1 и удаляем maxpool
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Убираем maxpool для маленьких изображений
    
    # Упростить и унифицировать dropout в backbone
    model.layer3[1].add_module('dropout', torch.nn.Dropout(0.7))  # было 0.8
    model.layer4[1].add_module('dropout', torch.nn.Dropout(0.8))  # было 0.9

    # УПРОСТИТЬ классификатор - это ключевое изменение!
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.7),  # было 0.8
        torch.nn.Linear(model.fc.in_features, 128),  # УМЕНЬШИТЬ с 256 до 128
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.6),  # было 0.7
        torch.nn.Linear(128, 10)
    )
    
    # Инициализация весов (оставить как есть)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    model = model.to(device)
    
    # Логирование
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Модель ResNet-18 адаптирована для CIFAR-10 и успешно создана.")
    logger.info(f"Обучаемые параметры: {trainable_params:,} из {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model

# Функция тренировки модели
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, batch_size=64, writer=None, round_num=1):
    logger.info("Перед началом обучения:")
    logger.info(f"Количество эпох: {epochs}")
    logger.info(f"Скорость обучения (lr): {lr}")
    logger.info(f"Размер батча (batch size): {batch_size}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    # Для ранней остановки - отслеживаем accuracy вместо loss
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%, LR: {current_lr:.6f}")
        

        # ИСПРАВЛЕНИЕ: Сохраняем модель когда улучшается accuracy, а не loss
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Ранняя остановка на эпохе {epoch+1}")
            logger.info(f"Лучшая Val Accuracy: {best_val_accuracy*100:.2f}%")
            break
    
    return model

# Функция для оценки модели
def evaluate_model(model, data_loader):
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():  # Выключаем градиенты для ускорения вычислений
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Перемещаем данные на GPU (если доступен)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Вычисляем итоговый loss и accuracy
    loss = running_loss / len(data_loader)
    accuracy = correct / total
    return loss, accuracy

