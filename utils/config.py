"""config.py

Модуль с конфигурацией проекта: константы, пути, словари.
"""
import os
import torch
# Пути к данным
DATA_DIR = 'data/samples'  # Путь к папке с изображениями капчи
WEIGHTS_DIR = 'weights'
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'best_model.pth')

# Параметры архитектуры и данных
SEQ_LEN = 5                # Фиксированная длина текста на капче (по примеру 2b827)

# Гиперпараметры обучения модели
BATCH_SIZE = 64            # Размер пакета данных (батча)
EPOCHS = 20                # Количество эпох обучения
LEARNING_RATE = 0.005      # Скорость обучения (шаг градиентного спуска)
IMG_HEIGHT = 40
IMG_WIDTH = 140
TRAIN_RATIO = 0.8         # 80% train / 20% test

# Параметры токенизации и словаря
VOCAB = "0123456789abcdefghijklmnopqrstuvwxyz"           # все символы (цифры и буквы), которые могут встретиться на капче
CHAR2IDX = {char: idx for idx, char in enumerate(VOCAB)} # Создаем словарь перевода БУКВ В ЦИФРЫ {'0': 0, '1': 1, ..., 'a': 10, 'b': 11, ...}
IDX2CHAR = {idx: char for char, idx in CHAR2IDX.items()} # Создаем обратный словарь: ЦИФРЫ В БУКВЫ {0: '0', 1: '1', ..., 10: 'a', 11: 'b', ...}
NUM_CLASSES = len(VOCAB)                                 # Считаем, сколько всего уникальных символов в алфавите (в данном случае их 36). Это число определяет, сколько вариантов ответа будет у нейросети для каждой буквы.
PAD_TOKEN = -100                                         # Игнорируется в CrossEntropyLoss

# Автоматический выбор устройства: GPU (если доступно) или CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создаем папку для весов, если её нет
os.makedirs(WEIGHTS_DIR, exist_ok=True)
