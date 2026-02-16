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

Команда установки стабильных версий библиотек для CUDA 12.8:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
```

Для активации виртуального окружения:
```bash
source .venv/bin/activate
```

## 3)