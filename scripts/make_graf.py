import json
import matplotlib.pyplot as plt

# Функция для построения графиков
def plot_metrics(json_file_path):
    # Чтение данных из JSON файла
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Извлечение данных для графиков
    rounds = list(data.keys())
    train_loss = [data[str(round)]["train_loss"] for round in rounds]
    train_accuracy = [data[str(round)]["train_accuracy"] for round in rounds]
    val_loss = [data[str(round)]["val_loss"] for round in rounds]
    val_accuracy = [data[str(round)]["val_accuracy"] for round in rounds]
    
    # Создание графиков
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # График для Train Loss
    axs[0, 0].plot(rounds, train_loss, marker='o', color='r')
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Rounds')
    axs[0, 0].set_ylabel('Loss')

    # График для Train Accuracy
    axs[0, 1].plot(rounds, train_accuracy, marker='o', color='g')
    axs[0, 1].set_title('Train Accuracy')
    axs[0, 1].set_xlabel('Rounds')
    axs[0, 1].set_ylabel('Accuracy')

    # График для Validation Loss
    axs[1, 0].plot(rounds, val_loss, marker='o', color='b')
    axs[1, 0].set_title('Validation Loss')
    axs[1, 0].set_xlabel('Rounds')
    axs[1, 0].set_ylabel('Loss')

    # График для Validation Accuracy
    axs[1, 1].plot(rounds, val_accuracy, marker='o', color='orange')
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Rounds')
    axs[1, 1].set_ylabel('Accuracy')

    # Автоматическое выравнивание меток
    plt.tight_layout()

    # Сохранение графиков в ту же папку, что и JSON файл
    output_path = json_file_path.rsplit('/', 1)[0]  # Получаем путь к папке
    plt.savefig(f"{output_path}/training_metrics.png")
    plt.show()

# Пример использования
# Укажите путь к вашему JSON файлу
json_file_path = "/home/nxhxl/work/ddl/runs/cifar10_iid_K10_seed42/aggregated_metrics.json"  # Замените на путь к вашему файлу
plot_metrics(json_file_path)
