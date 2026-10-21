"""utils/__init__.py

Пакет утилит для анализа, визуализации и оценки качества модели OCR.
Обеспечивает чистый импорт функций напрямую из папки utils.
"""

# Импортируем конфигурацию
from utils.config import *

# Импортируем метрику оценки
from utils.metrics import calculate_cer

# Импортируем модули визуализации ошибок
from utils.visualization import (
    collect_errors_data, 
    visualize_top_errors, 
    show_worst_predictions
)

# Импортируем модули статистического анализа (исправлены названия функций)
from utils.analysis import (
    print_general_statistics,
    analyze_position_and_confusion_errors,
    full_statistical_analysis
)
