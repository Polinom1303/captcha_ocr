"""train.py

Основной скрипт обучения модели AdvancedCRNN для распознавания капчи с визуализацией прогресса.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import config
from utils.metrics import calculate_cer
from utils.dataset import get_dataloaders
from utils.model import AdvancedCRNN


def train_epoch(model, loader, criterion, optimizer, device):
    """Выполняет одну эпоху обучения нейросети с полосой прогресса."""
    model.train()
    total_loss, total_cer = 0.0, 0.0
    
    # Использование tqdm для красивого отображения прогресса батчей
    for imgs, targets in tqdm(loader, desc="  Train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)  # Выход модели: (B, SEQ_LEN, NUM_CLASSES)
        
        # CrossEntropyLoss ожидает форму (B, NUM_CLASSES, SEQ_LEN)
        loss = criterion(logits.permute(0, 2, 1), targets)
        loss.backward()
        
        # Защита от взрыва градиентов в LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=2)
        total_cer += calculate_cer(preds, targets) * imgs.size(0)
        
    return total_loss / len(loader), total_cer / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Выполняет валидацию модели на тестовой выборке."""
    model.eval()
    total_loss, total_cer, exact_matches = 0.0, 0.0, 0
    
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="  Eval ", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            
            loss = criterion(logits.permute(0, 2, 1), targets)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=2)
            total_cer += calculate_cer(preds, targets) * imgs.size(0)
            
            # Считаем количество капч, которые угаданы абсолютно полностью (без единой ошибки)
            exact_matches += (preds == targets).all(dim=1).sum().item()
            
    return (
        total_loss / len(loader),
        total_cer / len(loader.dataset),
        (exact_matches / len(loader.dataset)) * 100  # Точность полного совпадения в %
    )


def main():
    """Основная функция для запуска полного цикла обучения."""
    device = config.DEVICE
    print(f"[Train] Starting training process on device: {device.type.upper()}")
    
    # Загружаем даталоадеры через безопасную функцию из датасета
    train_loader, test_loader = get_dataloaders()
    
    # Инициализируем модель, используя параметры из вашего конфига
    model = AdvancedCRNN(
        num_classes=config.NUM_CLASSES, 
        seq_len=config.SEQ_LEN
    ).to(device)
    
    # Используем config.PAD_TOKEN вместо жестко зашитого числа -100
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5 factor=0.5
    )
    
    best_cer = 1.0
    history = {
        'train_loss': [], 'train_cer': [], 
        'val_loss': [], 'val_cer': [], 'val_acc': []
    }
    
    print(f"[Train] Starting training for {config.EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        
        train_loss, train_cer = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_cer, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}") 
        scheduler.step(val_loss)
        
        # Сохраняем метрики в историю обучения
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   CER: {val_cer:.4f} | Val Accuracy (Full Match): {val_acc:.2f}%")
        
        # Автоматическое сохранение весов лучшей модели по пути из конфига
        if val_cer < best_cer:
            best_cer = val_cer
            # Гарантируем наличие папки weights перед сохранением
            os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  ✓ Saved best model to: {config.BEST_MODEL_PATH} (Best CER: {best_cer:.4f})")
            
    print(f"\n[Train] Finished! Best Validation CER: {best_cer:.4f}")
    return history


if __name__ == '__main__':
    main()
