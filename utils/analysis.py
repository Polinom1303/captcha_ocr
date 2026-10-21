"""analytics.py

Модуль для статистического анализа ошибок модели OCR.
Считает распределение ошибок по позициям, частоту путаницы символов и т.д.
"""

from collections import Counter
import torch
from utils import config  # Исправлен импорт под единый стандарт проекта


def _get_val(item):
    """Вспомогательный метод для безопасного извлечения значения.
    
    Работает корректно, даже если на вход подан тензор PyTorch или обычное число.
    """
    return item.item() if isinstance(item, torch.Tensor) else item


def print_general_statistics(errors_data):
    """Вывод общей статистики: количество ошибок и топ-5 худших примеров."""
    total_errors = sum(1 for x in errors_data if x['cer'] > 0)
    print(f"[Статистика] Всего картинок в тесте с ошибками: {total_errors} из {len(errors_data)}")
    
    if not errors_data:
        return

    print("\nДетальный разбор ТОП-5 худших предсказаний:")
    print("-" * 60)
    # Сортируем по убыванию CER и Loss, чтобы показать действительно худшие результаты
    sorted_worst = sorted(errors_data, key=lambda x: (-x['cer'], -x['loss']))
    
    for i in range(min(5, len(sorted_worst))):
        data = sorted_worst[i]
        true_str = "".join([config.IDX2CHAR[_get_val(idx)] 
                            for idx in data['true'] if _get_val(idx) != config.PAD_TOKEN])
        pred_str = "".join([config.IDX2CHAR[_get_val(idx)] 
                            for idx in data['pred']])
        print(f"Пример {i+1} | Истинный: '{true_str}' | "
              f"Предсказано: '{pred_str}' | CER: {data['cer']:.2f} | Loss: {data['loss']:.4f}")


def analyze_position_and_confusion_errors(errors_data):
    """Анализ распределения ошибок по позициям и построение матрицы путаницы."""
    position_errors = [0] * config.SEQ_LEN
    total_errors_counted = 0
    confusion_pairs = []
    
    for data in errors_data:
        if data['cer'] > 0:
            for pos, (t, p) in enumerate(zip(data['true'], data['pred'])):
                true_val = _get_val(t)
                pred_val = _get_val(p)
                
                # Игнорируем токены заполнения (pad) и верные предсказания
                if true_val != config.PAD_TOKEN and true_val != pred_val:
                    position_errors[pos] += 1
                    total_errors_counted += 1
                    
                    char_t = config.IDX2CHAR[true_val]
                    char_p = config.IDX2CHAR[pred_val]
                    confusion_pairs.append((char_t, char_p))
    
    print("\n Распределение ошибок по позициям в тексте (1-5 символы):")
    for pos, count in enumerate(position_errors):
        percentage = (count / total_errors_counted * 100) if total_errors_counted > 0 else 0
        print(f"   Позиция {pos+1}: {count} ошиб(ок) ({percentage:.1f}%)")
    
    print("\n Матрица путаницы — Топ-5 пар частых замещений:")
    pair_counts = Counter(confusion_pairs)
    for (true_c, pred_c), count in pair_counts.most_common(5):
        print(f"   Истинный символ '{true_c}' -> ошибочно предсказан как '{pred_c}': {count} раз(а)")
    
    return position_errors, pair_counts


def full_statistical_analysis(errors_data):
    """Функция-обёртка: запускает все виды статистического анализа."""
    if not errors_data:
        print("[Внимание] Список данных об ошибках пуст. Анализ невозможен.")
        return
        
    print_general_statistics(errors_data)
    print("-" * 60)
    analyze_position_and_confusion_errors(errors_data)


# Блок для изолированного тестирования логики анализа
if __name__ == '__main__':
    print("Запуск изолированного теста модуля аналитики...")
    
    # Имитируем структуру данных из валидатора train.py
    # Параметры капчи: истинные токены, предсказанные, CER и Loss
    mock_errors_data = [
        {
            'true': torch.tensor([1, 2, 3, 4, 5]), 
            'pred': torch.tensor([1, 9, 3, 4, 0]),  # Ошибки на 2-й и 5-й позициях
            'cer': 0.4, 
            'loss': 0.85
        },
        {
            'true': [1, 2, 3, -100, -100],  # Тест поддержки обычных списков Python
            'pred': [1, 9, 3, -100, -100],  # Ошибка на 2-й позиции
            'cer': 0.33, 
            'loss': 0.52
        }
    ]
    
    full_statistical_analysis(mock_errors_data)
