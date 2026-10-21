"""visualization.py

Модуль для визуализации ошибок модели OCR.
Собирает данные о предсказаниях и отображает топ-N худших результатов.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import config  # Исправлен относительный импорт
from utils.metrics import calculate_cer  # Исправлен путь к метрикам


def collect_errors_data(model, test_loader, criterion_none, device):
    """Проход по тестовой выборке и сбор статистики ошибок для каждой картинки.
    
    Args:
        model: Обученная модель
        test_loader: DataLoader с тестовыми данными
        criterion_none: CrossEntropyLoss(reduction='none', ignore_index=config.PAD_TOKEN)
        device: Устройство (cuda/cpu)
    
    Returns:
        list[dict]: Список словарей с данными об ошибках
    """
    errors_data = []
    model.eval()
    
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            
            # Считаем лосс для каждого символа отдельно
            # Форма логитов: (B, SEQ_LEN, NUM_CLASSES) -> (B, NUM_CLASSES, SEQ_LEN)
            losses = criterion_none(logits.permute(0, 2, 1), targets)
            img_losses = losses.mean(dim=1)  # Средний лосс на картинку
            
            preds = logits.argmax(dim=2)
            
            for i in range(imgs.size(0)):
                cer = calculate_cer(preds[i], targets[i])
                errors_data.append({
                    'img': imgs[i].cpu(),
                    'true': targets[i].cpu(),
                    'pred': preds[i].cpu(),
                    'cer': cer,
                    'loss': img_losses[i].item()
                })
    
    # Сортируем: сначала по максимальному CER (худшие), затем по Loss
    errors_data.sort(key=lambda x: (-x['cer'], -x['loss']))
    return errors_data


def visualize_top_errors(errors_data, top_n=5, title='Top Worst Predictions'):
    """Визуализация N худших предсказаний модели (по CER)."""
    if not errors_data:
        print("[Внимание] Нет данных для визуализации.")
        return

    n_cols = min(top_n, len(errors_data))
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
    
    # Если показывается только 1 пример, axes не будет списком, делаем из него список
    if n_cols == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes):
        if i >= len(errors_data):
            ax.axis('off')
            continue
            
        data = errors_data[i]
        img = data['img'].squeeze(0).numpy()
        
        # Декодируем тензоры обратно в текстовые строки, используя config.PAD_TOKEN
        true_str = "".join([config.IDX2CHAR[idx.item()] 
                            for idx in data['true'] if idx.item() != config.PAD_TOKEN])
        pred_str = "".join([config.IDX2CHAR[idx.item()] 
                            for idx in data['pred']])
        
        # Инвертируем нормализацию обратно в [0, 1] для корректного отображения imshow
        img = (img * 0.5) + 0.5
        
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {true_str}\nPred: {pred_str}\nCER: {data['cer']:.2f}\nLoss: {data['loss']:.3f}",
            fontsize=10
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_worst_predictions(model, test_loader, top_n=5):
    """Главная функция-обёртка: инициализирует критерий, собирает ошибки и выводит графики.
    
    Args:
        model: Обученная модель
        test_loader: DataLoader с тестовыми данными
        top_n: Количество примеров для отображения
    
    Returns:
        list[dict]: Список с данными об ошибках
    """
    device = config.DEVICE
    
    # Создаем локальный критерий с отключенным усреднением по батчу
    criterion_none = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN, reduction='none')
    
    # Собираем данные
    errors_data = collect_errors_data(model, test_loader, criterion_none, device)
    
    # Визуализируем
    visualize_top_errors(errors_data, top_n=top_n, 
                         title=f'Top {top_n} Worst Predictions (Highest CER)')
    
    return errors_data
