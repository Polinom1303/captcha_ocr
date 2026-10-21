"""metrics.py

Модуль с метриками для оценки качества модели OCR.
"""
import torch
from utils import config


def calculate_cer(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Расчет Character Error Rate (CER) — доли неверно угаданных символов.
    
    Поддерживает как одиночные последовательности, так и целые батчи данных.
    
    Args:
        pred (torch.Tensor): Тензор предсказанных индексов символов формы (SEQ_LEN,) или (B, SEQ_LEN)
        target (torch.Tensor): Тензор истинных индексов символов формы (SEQ_LEN,) или (B, SEQ_LEN)
                               Значение config.PAD_TOKEN игнорируется при расчете.
    
    Returns:
        float: Доля неверно распознанных символов (от 0.0 до 1.0)
    """
    # Выделяем маску полезных символов, игнорируя паддинг из конфига
    valid_mask = (target != config.PAD_TOKEN)
    
    # Защита от деления на ноль, если батч/строка полностью состоит из паддингов
    if valid_mask.sum() == 0:
        return 0.0
    
    # Считаем количество несовпадений только там, где есть реальные символы капчи
    incorrect = (pred[valid_mask] != target[valid_mask]).sum().item()
    
    # Возвращаем отношение ошибочных символов к общему числу валидных символов
    return incorrect / valid_mask.sum().item()


# Блок для изолированного тестирования метрики
if __name__ == '__main__':
    print("Тестирование модуля метрик...")
    
    # Тест 1: Идеальное совпадение на батче из 2 элементов
    mock_target = torch.tensor([[1, 2, 3, -100, -100], [4, 5, 6, 7, -100]])
    mock_pred_ok = torch.tensor([[1, 2, 3, 99, 99], [4, 5, 6, 7, 88]])
    cer_ok = calculate_cer(mock_pred_ok, mock_target)
    print(f"Тест 1 (Ожидается 0.0): CER = {cer_ok:.4f}")
    
    # Тест 2: Ошибки в батче (2 ошибки из 7 валидных символов)
    # В первой строке ошиблись в '3' (написали '9'), во второй строке ошиблись в '7' (написали '0')
    mock_pred_err = torch.tensor([[1, 2, 9, 99, 99], [4, 5, 6, 0, 88]])
    cer_err = calculate_cer(mock_pred_err, mock_target)
    print(f"Тест 2 (Ожидается 2/7 = 0.2857): CER = {cer_err:.4f}")
