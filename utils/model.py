"""model.py
Модуль архитектуры нейронной сети AdvancedCRNN для распознавания капчи.
"""

import torch
import torch.nn as nn

from utils import config


class AdvancedCRNN(nn.Module):
    """Сверточно-рекуррентная нейронная сеть (CRNN).

    Объединяет глубокие сверточные слои (VGG-style) для извлечения визуальных 
    признаков, двунаправленную LSTM (Bi-LSTM) для анализа последовательности 
    символов и полносвязный классификатор.
    """

    def __init__(self, num_classes, seq_len):
        """Инициализирует слои и блоки нейронной сети."""
        super().__init__()
        self.seq_len = seq_len
        
        # Глубокая CNN (VGG-style) с батч-нормализацией (BatchNorm)
        self.cnn = nn.Sequential(
            # Блок 1: Вход (B, 1, 40, 140) -> Выход (B, 64, 20, 70)
            nn.Conv2d(1, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 2: Вход (B, 64, 20, 70) -> Выход (B, 128, 10, 35)
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Блок 3: Уменьшаем только высоту. Выход (B, 256, 5, 35)
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Блок 4: Уменьшаем только высоту. Выход (B, 512, 2, 35)
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        # Вместо жесткого сжатия до seq_len мы сначала аккуратно убираем высоту до 1
        self.height_pool = nn.AdaptiveAvgPool2d((1, 35)) 

        # Двунаправленная RNN (Bi-LSTM) для захвата контекста слева-направо и справа-налево
        self.rnn = nn.LSTM(
            input_size=512, 
            hidden_size=256, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Финальный классификатор. Входной размер удвоен (256 * 2), так как LSTM двунаправленная
        self.fc = nn.Linear(256 * 2, num_classes)

        # Автоматически применяем инициализацию весов при создании объекта класса
        self.apply(self._init_weights)

    def forward(self, x):
        """Выполняет прямой проход данных через слои сети."""
        # Извлечение визуальных признаков через CNN: (B, 1, 40, 140) -> (B, 512, 2, 35)
        conv_out = self.cnn(x)               
        
        # Убираем высоту до 1. На выходе получаем: (B, 512, 1, 35)
        pooled = self.height_pool(conv_out) 

        # Убираем лишнюю единичную ось высоты: (B, 512, 35)
        rnn_in = pooled.squeeze(2).permute(0, 2, 1)
        
        # Проход через рекуррентные слои: (B, 5, 512) -> (B, 5, 512)
        rnn_out, _ = self.rnn(rnn_in)        
        
        # Классификация каждого временного шага: (B, 5, 512) -> (B, 5, NUM_CLASSES)
        logits = self.fc(rnn_out)   
        
        # Сжатие времени ПОСЛЕ классификации: блок из 7 колонок -> 1 символ
        b = logits.size(0)
        logits = logits.view(b, self.seq_len, 35 // self.seq_len, -1).mean(dim=2)              
        
        return logits

    def _init_weights(self, m):
        """Внутренний метод инициализации весов для стабильного старта обучения."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)


# Блок для изолированного тестирования архитектуры модели
if __name__ == '__main__':
    # Создаем модель, используя параметры из общего конфига
    model = AdvancedCRNN(
        num_classes=config.NUM_CLASSES, 
        seq_len=config.SEQ_LEN
    ).to(config.DEVICE) # Берем девайс напрямую из config
    
    print("Архитектура модели успешно инициализирована!")
    print(f"Модель отправлена на устройство: {config.DEVICE}")
    
    # Фейковый тестовый батч для проверки работоспособности слоев forward()
    # Размер: [батч=4, каналы=1 (серый), высота=40, ширина=140]
    dummy_input = torch.randn(4, 1, 40, 140).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Входной размер тензора: {dummy_input.shape}")
    print(f"Выходной размер тензора (логиты): {output.shape}") 
    # Ожидается: [4, config.SEQ_LEN, config.NUM_CLASSES]
