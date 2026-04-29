# GIS Pipeline — Извлечение объектов из космоснимков

Учебный кейс NURIS: автоматическое выявление объектов по данным ДЗЗ
с формированием ГИС-слоёв.

---

## Архитектура

```
main.py                  ← точка входа, оркестрация
pipeline/
  loader.py              ← загрузка GeoTIFF, AOI, репроекция, NDVI/NDWI/NDBI
  tiler.py               ← разбивка на тайлы с перекрытием
  detector.py            ← мульти-классовая детекция (4 детектора)
  postprocess.py         ← NMS, слияние, геовалидация
  exporter.py            ← GeoJSON / GeoPackage / Shapefile
  metrics.py             ← P/R/F1/mAP, зональная статистика
  visualizer.py          ← PNG-карты для отчёта
config.yaml              ← все параметры
requirements.txt
Dockerfile
```

## Классы объектов

| Класс        | Геометрия   | Метод детекции                         |
|--------------|-------------|----------------------------------------|
| `vegetation` | Polygon     | NDVI > порог + морфология              |
| `water`      | Polygon     | NDWI > порог + морфология              |
| `building`   | Polygon     | Яркость + NDBI + компактность          |
| `road`       | LineString  | Canny edges + Hough Transform          |

---

## Быстрый старт

### 1. Установка зависимостей

```bash
# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate.bat       # Windows

# Установить зависимости
pip install -r requirements.txt
```

### 2. Демо-режим (без реальных данных)

```bash
python main.py --demo --output output_demo
```

Создаёт синтетический снимок (512×512, 4 полосы, Алматы) и прогоняет полный пайплайн.

### 3. Реальные данные

```bash
python main.py \
  --image  data/scene.tif \
  --aoi    data/aoi.geojson \
  --output output/ \
  --config config.yaml
```

### 4. С оценкой качества по GT

```bash
python main.py \
  --image  data/scene.tif \
  --output output/ \
  --gt     data/ground_truth.geojson
```

### 5. Docker

```bash
# Сборка
docker build -t gis-pipeline .

# Демо
docker run -v $(pwd)/output:/output gis-pipeline --demo

# Реальные данные
docker run \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  gis-pipeline --image /data/scene.tif --aoi /data/aoi.geojson
```

---

## Входные данные

| Файл               | Формат              | Описание                              |
|--------------------|---------------------|---------------------------------------|
| `scene.tif`        | GeoTIFF с геопривязкой | Космоснимок (RGB, 4-полосный и т.д.) |
| `aoi.geojson`      | GeoJSON Polygon     | Контур территории интереса (опц.)     |
| `ground_truth.geojson` | GeoJSON         | Разметка для оценки качества (опц.)   |

**Без геопривязки:** если снимок не содержит CRS, нужно указать её вручную в `config.yaml`:
```yaml
crs:
  processing: "EPSG:32642"
```
Или передать снимок через `rasterio` с явным `transform` и `crs`.

---

## Результаты (output/)

```
output/
  detections.geojson          ← все объекты (обязательный по ТЗ)
  detections_vegetation.geojson
  detections_water.geojson
  detections_building.geojson
  detections_road.geojson
  detections.gpkg             ← GeoPackage
  shapefile/                  ← Shapefile (по типам геометрии)
  summary.json                ← сводка по классам
  zone_statistics.json        ← количество / площадь / плотность
  zone_statistics.csv
  quality_report.json         ← метрики качества
  manual_sample.csv           ← контрольная выборка (без GT)
  overview.png                ← обзорная карта
  class_vegetation.png
  class_water.png
  class_building.png
  class_road.png
  density_map.png             ← тепловая карта плотности
  pipeline.log
```

### Структура GeoJSON (по ТЗ)

```json
{
  "type": "Feature",
  "properties": {
    "id":         "building_demo_000042",
    "class":      "building",
    "confidence": 78.3,
    "source":     "demo_almaty",
    "date":       "2025-04-26",
    "area_m2":    342.5,
    "attr_aspect_ratio": 1.42,
    "attr_compactness":  0.61
  },
  "geometry": { "type": "Polygon", "coordinates": [...] }
}
```

---

## Конфигурация (config.yaml)

Основные параметры:

```yaml
tiling:
  size:    256    # размер тайла (пиксели)
  overlap:  32    # перекрытие (пиксели)

detection:
  vegetation:
    ndvi_threshold: 0.20   # чем ниже — больше объектов, больше шума
  building:
    min_compactness: 0.15  # фильтр нитеобразных ложных позитивов

postprocess:
  min_confidence:     25.0
  nms_iou_threshold:   0.4  # NMS — удаление дублей тайловых границ
  merge_touching:     true   # слияние касающихся полигонов
```

---

## Метрики качества

**При наличии GT (`--gt`):**
- Precision / Recall / F1 по каждому классу
- mAP (mean Average Precision) при IoU ≥ 0.5
- PR-кривая

**Без GT:**
- Контрольная выборка (50 объектов, стратифицировано по классам)
- Автоматическая эвристическая оценка (форма / размер / уверенность)
- Типичные ошибки с объяснением причин

---

## Ограничения прототипа

1. Детектор зданий основан на яркости — плохо работает на снимках с тенями
   или снегом. Решение: добавить текстурные признаки (GLCM) или нейросеть.
2. NDVI/NDWI требуют NIR-полосы. Для чисто RGB-снимков детекция растительности
   и воды менее точна.
3. Детектор дорог (Hough) выдаёт фрагменты, не непрерывные линии.
   Решение: использовать D-LinkNet или аналог.
4. Прототип обрабатывает один снимок за раз (нет оценки изменений).
5. Нет поддержки облачной маскировки.
