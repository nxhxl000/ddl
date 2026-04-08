"""
CIFAR-100 — Exploratory Data Analysis (EDA)
Для методологической части ВКР по федеративному обучению.

Анализы:
  1. Статистика по каналам RGB (средние, std) — обоснование нормализации
  2. Распределение классов и суперклассов
  3. Межклассовое сходство: матрица расстояний + t-SNE

Запуск:
  pip install torch torchvision matplotlib seaborn scikit-learn numpy
  python cifar100_eda.py

Результат: 4 PNG-файла в папке ./eda_results/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

import torchvision
import torchvision.transforms as transforms

# ─── Настройки ───────────────────────────────────────────────
OUTPUT_DIR = './eda_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ─── Загрузка данных ─────────────────────────────────────────
print("Загрузка CIFAR-100...")
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True
)

# Массив изображений: (N, 32, 32, 3), uint8
train_images = np.array([np.array(img) for img, _ in train_dataset])
train_labels = np.array(train_dataset.targets)

fine_classes = train_dataset.classes  # 100 классов

# Маппинг суперклассов CIFAR-100
SUPERCLASS_NAMES = [
    'aquatic mammals', 'fish', 'flowers', 'food containers',
    'fruit and vegetables', 'household electrical devices',
    'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes',
    'large omnivores and herbivores', 'medium-sized mammals',
    'non-insect invertebrates', 'people', 'reptiles',
    'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
]

SUPERCLASS_MAPPING = {
    'aquatic mammals':              ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':                         ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':                      ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers':              ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables':         ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture':          ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':                      ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores':             ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores':['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals':         ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates':     ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':                       ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles':                     ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals':                ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees':                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1':                   ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2':                   ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

# Построить маппинг fine_label -> coarse_label
fine_to_coarse = {}
for sc_idx, sc_name in enumerate(SUPERCLASS_NAMES):
    for cls_name in SUPERCLASS_MAPPING[sc_name]:
        if cls_name in fine_classes:
            fine_idx = fine_classes.index(cls_name)
            fine_to_coarse[fine_idx] = sc_idx

coarse_labels = np.array([fine_to_coarse[l] for l in train_labels])

print(f"Загружено: {len(train_images)} обучающих изображений, {len(test_dataset)} тестовых")
print(f"Форма изображения: {train_images[0].shape}")
print(f"Классов: {len(fine_classes)}, Суперклассов: {len(SUPERCLASS_NAMES)}")

# ═══════════════════════════════════════════════════════════════
# 1. СТАТИСТИКА ПО КАНАЛАМ RGB
# ═══════════════════════════════════════════════════════════════
print("\n[1/3] Статистика по каналам RGB...")

# Нормализуем к [0,1]
images_float = train_images.astype(np.float32) / 255.0

channel_names = ['Red', 'Green', 'Blue']
channel_colors = ['#e74c3c', '#2ecc71', '#3498db']

# Средние и std по каждому каналу (по всем пикселям всех изображений)
means = [images_float[:, :, :, c].mean() for c in range(3)]
stds  = [images_float[:, :, :, c].std()  for c in range(3)]

print(f"  Средние по каналам (R, G, B): ({means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f})")
print(f"  Std по каналам    (R, G, B): ({stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f})")

# Распределение средней яркости по каналам для каждого изображения
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Распределение средней яркости по каналам RGB\n(CIFAR-100, обучающая выборка)', 
             fontsize=14, fontweight='bold')

for c in range(3):
    per_image_means = images_float[:, :, :, c].reshape(len(images_float), -1).mean(axis=1)
    axes[c].hist(per_image_means, bins=80, color=channel_colors[c], alpha=0.75, edgecolor='white', linewidth=0.3)
    axes[c].axvline(means[c], color='black', linestyle='--', linewidth=1.5, 
                    label=f'μ = {means[c]:.4f}')
    axes[c].axvline(means[c] - stds[c], color='gray', linestyle=':', linewidth=1)
    axes[c].axvline(means[c] + stds[c], color='gray', linestyle=':', linewidth=1,
                    label=f'σ = {stds[c]:.4f}')
    axes[c].set_title(f'Канал {channel_names[c]}')
    axes[c].set_xlabel('Средняя яркость')
    axes[c].set_ylabel('Количество изображений')
    axes[c].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_rgb_channel_stats.png'), bbox_inches='tight')
plt.close()
print("  → Сохранено: 1_rgb_channel_stats.png")


# ═══════════════════════════════════════════════════════════════
# 2. РАСПРЕДЕЛЕНИЕ КЛАССОВ И СУПЕРКЛАССОВ
# ═══════════════════════════════════════════════════════════════
print("\n[2/3] Распределение классов и суперклассов...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Распределение изображений по классам и суперклассам\n(CIFAR-100, обучающая выборка)',
             fontsize=14, fontweight='bold')

# 2a. Распределение по 100 fine-классам
class_counts = np.bincount(train_labels, minlength=100)
colors_fine = plt.cm.tab20(np.array([fine_to_coarse[i] for i in range(100)]) / 20.0)
ax1.bar(range(100), class_counts, color=colors_fine, edgecolor='white', linewidth=0.2)
ax1.axhline(y=500, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Ожидаемое (500)')
ax1.set_xlabel('Индекс класса (fine label)')
ax1.set_ylabel('Количество изображений')
ax1.set_title('Распределение по 100 классам (цвет = суперкласс)')
ax1.set_xlim(-1, 100)
ax1.legend()

# 2b. Распределение по 20 суперклассам
coarse_counts = np.bincount(coarse_labels, minlength=20)
bars = ax2.barh(range(20), coarse_counts, color=plt.cm.tab20(np.arange(20)/20.0), 
                edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(20))
ax2.set_yticklabels(SUPERCLASS_NAMES, fontsize=9)
ax2.set_xlabel('Количество изображений')
ax2.set_title('Распределение по 20 суперклассам')
ax2.axvline(x=2500, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Ожидаемое (2500)')
for bar, count in zip(bars, coarse_counts):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2, 
             str(count), va='center', fontsize=8)
ax2.legend()
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_class_distribution.png'), bbox_inches='tight')
plt.close()
print("  → Сохранено: 2_class_distribution.png")


# ═══════════════════════════════════════════════════════════════
# 3. МЕЖКЛАССОВОЕ СХОДСТВО
# ═══════════════════════════════════════════════════════════════
print("\n[3/3] Межклассовое сходство...")

# 3a. Вычислить средние представления (mean image) для каждого класса
class_means = np.zeros((100, 32*32*3))
for c in range(100):
    mask = train_labels == c
    class_means[c] = images_float[mask].reshape(-1, 32*32*3).mean(axis=0)

# 3b. Матрица косинусных расстояний между классами (сгруппировано по суперклассам)
# Сортируем классы по суперклассу для наглядности
sorted_indices = sorted(range(100), key=lambda i: (fine_to_coarse[i], i))
sorted_class_means = class_means[sorted_indices]
sorted_class_names = [fine_classes[i] for i in sorted_indices]

dist_matrix = cdist(sorted_class_means, sorted_class_means, metric='cosine')

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(dist_matrix, ax=ax, cmap='RdYlBu_r', 
            xticklabels=False, yticklabels=sorted_class_names,
            cbar_kws={'label': 'Косинусное расстояние'})
ax.set_title('Матрица косинусных расстояний между 100 классами CIFAR-100\n'
             '(классы сгруппированы по суперклассам)', fontsize=13, fontweight='bold')
ax.set_ylabel('Класс')
ax.set_xlabel('Класс')
ax.tick_params(axis='y', labelsize=5)

# Нарисовать границы суперклассов
superclass_sizes = []
current_sc = fine_to_coarse[sorted_indices[0]]
count = 0
for idx in sorted_indices:
    sc = fine_to_coarse[idx]
    if sc != current_sc:
        superclass_sizes.append(count)
        current_sc = sc
        count = 1
    else:
        count += 1
superclass_sizes.append(count)

pos = 0
for size in superclass_sizes:
    ax.axhline(y=pos, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(x=pos, color='black', linewidth=0.5, alpha=0.5)
    pos += size

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3a_cosine_distance_matrix.png'), bbox_inches='tight')
plt.close()
print("  → Сохранено: 3a_cosine_distance_matrix.png")

# 3c. t-SNE визуализация (на средних представлениях классов)
print("  Запуск t-SNE...")
tsne = TSNE(n_components=2, perplexity=15, random_state=42, max_iter=2000)
embeddings = tsne.fit_transform(class_means)

fig, ax = plt.subplots(figsize=(14, 10))
sc_colors = plt.cm.tab20(np.arange(20) / 20.0)

for sc_idx, sc_name in enumerate(SUPERCLASS_NAMES):
    mask = np.array([fine_to_coarse[i] == sc_idx for i in range(100)])
    ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
               c=[sc_colors[sc_idx]], s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
               label=sc_name, zorder=3)
    # Подписи классов
    for i in np.where(mask)[0]:
        ax.annotate(fine_classes[i], (embeddings[i, 0], embeddings[i, 1]),
                    fontsize=5, alpha=0.7, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')

ax.set_title('t-SNE визуализация средних представлений 100 классов CIFAR-100\n'
             '(цвет = суперкласс)', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE компонента 1')
ax.set_ylabel('t-SNE компонента 2')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, 
          ncol=1, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3b_tsne_class_embeddings.png'), bbox_inches='tight')
plt.close()
print("  → Сохранено: 3b_tsne_class_embeddings.png")


# ═══════════════════════════════════════════════════════════════
# СВОДКА
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("СВОДНАЯ СТАТИСТИКА CIFAR-100")
print("="*60)
print(f"Обучающая выборка:    {len(train_images)} изображений")
print(f"Тестовая выборка:     {len(test_dataset)} изображений")
print(f"Размер изображения:   32 × 32 × 3 (RGB)")
print(f"Количество классов:   {len(fine_classes)} (fine) / {len(SUPERCLASS_NAMES)} (coarse)")
print(f"Изображений на класс: {class_counts[0]} (train) / {10000//100} (test)")
print(f"")
print(f"Средние по каналам (R, G, B): ({means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f})")
print(f"Std по каналам    (R, G, B): ({stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f})")
print(f"")
print(f"Мин. расстояние между классами:  {dist_matrix[dist_matrix > 0].min():.4f}")
print(f"Макс. расстояние между классами: {dist_matrix.max():.4f}")
print(f"Среднее расстояние:              {dist_matrix[np.triu_indices(100, k=1)].mean():.4f}")
print(f"")
print(f"Файлы сохранены в: {os.path.abspath(OUTPUT_DIR)}/")
print("="*60)