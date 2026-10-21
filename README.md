# OCR для распознавания капчи
## Описание проекта
Решение финального задания по курсу **Deep Learning Basic**. Реализована модель архитектуры **CRNN** (Convolutional Recurrent Neural Network) для распознавания символов на изображениях капчи.

## Структура проекта
DL project/
├── data/
├── utils/
│   ├── __init__.py
│   ├── config.py           # Константы и словари
│   ├── dataset.py          # CaptchaDataset и DataLoader'ы
│   ├── model.py            # AdvancedCRNN
│   ├── train.py            # Цикл обучения
│   ├── metrics.py          # Функция calculate_cer
│   ├── visualization.py    # Визуализация ошибок (топ-5 худших)
│   └── analysis.py         # Статистический анализ ошибок
├── weights/
│   └── best_model.pth      # Лучшие веса
├── captcha_ocr_pipeline.ipyn          # Главный ноутбук
└── README.md

