# GeoJEPA 🛰️

**Определение местоположения и генерация описаний по спутниковым снимкам на базе JEPA-архитектуры.**

```
┌─────────────────────────────────────────┐
│  GeoJEPA                                │
├─────────────────────────────────────────┤
│  Vision Encoder (ViT-S, 384d, 12L)     │
│  ├── JEPA Pretraining (маскирование 75%)│
│  ├── EMA Target Encoder                 │
│  └── Predictor (cross-attention)        │
├─────────────────────────────────────────┤
│  Downstream:                            │
│  ├── GeoLocator (иерархическая класс.)  │
│  └── GeoDescriber (GPT-2 decoder)       │
└─────────────────────────────────────────┘
```

## ⚡ Быстрый старт (RTX 3090 24GB)

```bash
# 1. Клонировать и установить
git clone <repo>
cd geojepa
pip install -e ".[all]"

# 2. Быстрый тест (5 минут, синтетические данные)
python quickstart.py

# 3. Полное обучение
python scripts/train.py --synthetic --num-samples 100000 --epochs 100 --batch-size 64
```

## 📊 Время обучения на RTX 3090 24GB

| Этап | Данные | Batch | Эпох | Время/эпоху | Итого |
|------|--------|-------|------|-------------|-------|
| **JEPA Pretraining** | 100K снимков | 64 | 100 | ~30 мин | **~50 часов** |
| **JEPA (тест)** | 10K снимков | 64 | 2 | ~3 мин | **~5 мин** |
| **GeoLocator** | 100K снимков | 64 | 10 | ~10 мин | **~1.5 часа** |
| **GeoDescriber** | 100K снимков | 64 | 10 | ~15 мин | **~2.5 часа** |
| **ИТОГО** | | | | | **~54 часа** |

**VRAM использование:** ViT-Small (384d, 12L) + batch 64 = ~8GB. Остаётся запас для mixed precision и больших батчей.

## 🚀 Использование

### Предобучение JEPA (self-supervised)

```bash
# С реальными данными Sentinel-2 (требуются credentials)
export SH_CLIENT_ID="your_id"
export SH_CLIENT_SECRET="your_secret"
python scripts/train.py --auto-download --num-samples 100000 --epochs 100

# С синтетическими данными (для теста)
python scripts/train.py --synthetic --num-samples 10000 --epochs 10
```

### Обучение downstream задач

```bash
python scripts/train.py --downstream --pretrained checkpoints/best_model.pt
```

### Инференс

```bash
python scripts/predict.py satellite_image.tif --checkpoint checkpoints/best_model.pt
```

### Веб-интерфейс (Gradio)

```bash
python scripts/demo.py --checkpoint checkpoints/best_model.pt --share
```

## 📁 Структура проекта

```
geojepa/
├── src/
│   ├── config.py              # Конфигурация
│   ├── models/
│   │   ├── encoder.py         # ViT энкодер + EMA
│   │   ├── predictor.py       # JEPA предиктор
│   │   ├── geolocator.py      # Иерархическая геолокация
│   │   ├── geodescriber.py    # GPT-2 декодер для описаний
│   │   └── geojepa.py         # Основная модель
│   ├── data/
│   │   ├── dataset.py         # Dataset + аугментации
│   │   └── downloader.py      # Авто-загрузка Sentinel-2
│   ├── training/
│   │   └── trainer.py         # Лупы обучения
│   └── inference/
│       └── predictor.py       # Инференс + форматирование
├── scripts/
│   ├── train.py               # Скрипт обучения
│   ├── predict.py             # Скрипт предсказания
│   └── demo.py                # Gradio демо
├── configs/
│   └── default.yaml           # YAML конфиг
├── quickstart.py              # Быстрый старт
├── setup.py                   # Установка
├── requirements.txt           # Зависимости
└── README.md                  # Этот файл
```

## 🔑 Получение Sentinel Hub credentials

1. Зарегистрируйтесь на [dataspace.copernicus.eu](https://dataspace.copernicus.eu/)
2. Создайте OAuth client в личном кабинете
3. Установите переменные окружения:
   ```bash
   export SH_CLIENT_ID="your_client_id"
   export SH_CLIENT_SECRET="your_client_secret"
   ```

## 📡 Источники данных

| Источник | Разрешение | Покрытие | Доступ |
|----------|-----------|----------|--------|
| Sentinel-2 (ESA) | 10 м | Глобальное | Бесплатно (требуется регистрация) |
| Landsat 8/9 (USGS) | 30 м | Глобальное | Бесплатно |
| Синтетические | — | — | Локально |

## 🧠 Архитектура JEPA для спутниковых снимков

### Почему JEPA?

- **Предсказание пикселей** бессмысленно: один ландшафт выглядит по-разному в разные сезоны
- **Предсказание представлений** (embeddings) заставляет модель выучивать **инвариантные признаки**: рельеф, гидрография, тип застройки

### Процесс обучения

```
1. Маскируем 75% патчей снимка
2. Контекстный энкодер обрабатывает видимые патчи
3. Предиктор пытается предсказать embedding маскированных патчей
4. Target encoder (EMA) даёт ground truth embeddings
5. Контрастивная потеря: близкие координаты → близкие embeddings
```

## 🎯 Точность

| Уровень | Точность | Как достичь |
|---------|----------|-------------|
| **Континент** | >95% | По растительности и рельефу |
| **Страна** | ~80% | По типу ландшафта + дорогам |
| **Регион** | ~60% | По уникальным конфигурациям |
| **Точка (<1°)** | ~30% | Требует сверхвысокого разрешения |

## 📦 Установка

```bash
# Базовая установка
pip install -e .

# С поддержкой загрузки реальных данных
pip install -e ".[data]"

# С демо-интерфейсом
pip install -e ".[demo]"

# Всё сразу
pip install -e ".[all]"
```

## 📝 Лицензия

MIT

## 🙏 Благодарности

- [Sentinel Hub](https://www.sentinel-hub.com/) — API для спутниковых данных
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) — ESA данные
- [OpenEO](https://openeo.org/) — Стандартизированный доступ к EO данным
