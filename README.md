# Репозиторий для создания системы федеративного обучения

## 1) Клонирование репозитория:

Склонируй проект и перейди в папку:

```bash
git clone https://github.com/nxhxl000/ddl.git
cd ./ddl/
```

## 2) Создание и активация окружения (.venv)

Зависимости находятся в requirements.txt

Для создания виртуального окружения необходимо перейти в корневую папку проекта и выполнить следующие команды:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Если хотите использовать GPU c CUDA, установите сответсвуюшие вашей версии CUDA библиотеки torch, torchvision, torchaudio.

Команды установки стабильных версий библиотек для CUDA 12.8:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```

Для активации виртуального окружения:
```bash
source .venv/bin/activate
```

## 3) Скрипт для загрузки датасетов CIFAR-10 и MNIST 

Путь к скрипту: scripts/download_datasets.py

Для загрузки обоих датасетов выполни команду:
```bash
python -m scripts.download_datasets
```
## 4) Запуск скриптов из папки 

Все скрипты, находящиеся в папке scripts/ лучше запускать и корня репозитория как модули Python
Примеры:
```bash
python -m scripts.split_check
python -m scripts.cuda_check
```
## 5) Загрузка моделей для эксперементов

Скрипт находится по пути: scripts/load_models.py 
Загружает и подготавливает модели для дальнейших экспериментов
Параметр "--pretrained" позволяет загрузить предобученные модели для ResNet/MobileNet 
Модели сохраняются в папки: models/cifar10; models/mnist
```bash
python -m scripts.load_models
python -m scripts.load_models --pretrained 
```
## 6) Проведение локальных эксперементов

Скрипт находится по пути scripts/local_train.py
Локальные эксперименты сохраняются в отдельную папку local_exp/
Запуск:
 ```bash
python -m scripts.local_train
```

## 7) Скрипты для федеративного обучения

Основыне скрипты, отвечающие за федеративное обучение находятся в папке fl_app/

Скрипт task.py является утилитарным, в нем расположены функции загрузки и разбиения датасетов, создания модели, dataloaders, функции тренировки и оценки

Скрипт client_app отвечает за клиентскую часть приложения, на строне клиентов происходит обучение модели.

Скрипт server_app отвечает за серверную часть приложения, на строне сервера происходит валидационная оценка модели каждый раунд обучения.

В файле pyproject.toml находится конфиг федеративного обучения

Для запуска необходжимо в корне репозитория выполнить:
```bash
flwr run .
```

Логи эесперемента появятся в папке logs/ с именем (название датасета + схеме разбиения + число клиентов)


