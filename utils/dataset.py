"""dataset.py
Модуль для загрузки и предобработки датасета капчи.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from utils import config


class CaptchaDataset(Dataset):
    """Класс-датасет для подготовки изображений капчи к обучению в PyTorch."""

    def __init__(self, img_paths, transform=None):
        """Инициализирует объект датасета списком путей к изображениям."""
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        """Возвращает общее количество элементов в данном датасете."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Загружает, обрабатывает и возвращает один элемент данных по его индексу."""
        path = self.img_paths[idx]
        
        # Выделяем имя файла без расширения и приводим к нижнему регистру
        filename = os.path.basename(path)
        label_str = os.path.splitext(filename)[0].lower()
        
        # Чтение изображения через Pillow в оттенках серого
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
            
        # Кодирование символов строки в числовые индексы
        target = [config.CHAR2IDX.get(char, 0) for char in label_str]
        
        # Валидация и защита от нестандартной длины текста
        if len(target) < config.SEQ_LEN:
            target += [-100] * (config.SEQ_LEN - len(target))
        else:
            target = target[:config.SEQ_LEN]
            
        return img, torch.tensor(target, dtype=torch.long)


def get_transform():
    """Возвращает конвейер предобработки изображений."""
    return T.Compose([
        T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])


def get_dataloaders():
    """Сканирует директорию, разбивает выборку и возвращает train и test DataLoader."""
    # Проверка директории
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(
            f"Директория с датасетом не найдена: {os.path.abspath(config.DATA_DIR)}. "
            f"Проверьте переменную DATA_DIR в файле utils/config.py"
        )

    # Сбор только файлов изображений
    all_files = [
        os.path.join(config.DATA_DIR, f) 
        for f in os.listdir(config.DATA_DIR) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not all_files:
        raise ValueError(f"В указанной папке {config.DATA_DIR} отсутствуют файлы изображений!")

    # Создаем ОДИН базовый датасет без трансформаций (или с базовыми)
    transform_fn = get_transform()
    full_dataset = CaptchaDataset(all_files, transform=transform_fn)

    # Расчет пропорций разделения выборки
    train_size = int(config.TRAIN_RATIO * len(all_files))
    test_size = len(all_files) - train_size

    # Разделение Самого датасета с фиксацией сида
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Проверяем, является ли выбранное в конфиге устройство видеокартой CUDA
    is_cuda = config.DEVICE.type == 'cuda'
    
    # Безопасный расчет num_workers для Windows/CPU
    # Если проект запускается на CPU, ставим 0 (самый безопасный режим без лагов и падений), если на GPU, оставляем 2 для параллельной загрузки.
    num_workers = 2 if is_cuda else 0
    
    # Инициализация многопоточных загрузчиков
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,      # Применит 0 для CPU и 2 для GPU
        pin_memory=is_cuda  # Включится автоматически только для CUDA и ускоряет передачу данных на GPU
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,       # Применит 0 для CPU и 2 для GPU
        pin_memory=is_cuda   # Включится автоматически только для CUDA и ускоряет передачу данных на GPU
    )
    
    return train_loader, test_loader


# Блок для тестирования модуля (выполняется только при прямом запуске dataset.py)
if __name__ == '__main__':
    try:
        train_loader, test_loader = get_dataloaders()
        print(f"Успешно создано батчей для обучения: {len(train_loader)}")
        print(f"Успешно создано батчей для валидации: {len(test_loader)}")
        
        # Проверяем один батч данных
        images, labels = next(iter(train_loader))
        print(f"Размер батча картинок: {images.shape}")  # Ожидается: [batch_size, 1, 40, 140]
        print(f"Размер батча меток: {labels.shape}")    # Ожидается: [batch_size, seq_len]
    except Exception as e:
        print(f"Ошибка при тестировании датасета: {e}")
