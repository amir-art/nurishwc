# Dockerfile — GIS Pipeline (NURIS)
FROM python:3.11-slim

LABEL maintainer="NURIS GIS Team"
LABEL description="Автоматическое извлечение объектов из космоснимков"

# Системные зависимости GDAL/GEOS/PROJ
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY . .

# Папки для данных и результатов
RUN mkdir -p /data /output

# Точка входа
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config.yaml", "--output", "/output", "--demo"]

# Использование:
#   docker build -t gis-pipeline .
#   docker run -v $(pwd)/data:/data -v $(pwd)/output:/output \
#     gis-pipeline --image /data/scene.tif --aoi /data/aoi.geojson
#
#   Демо (без реальных данных):
#   docker run -v $(pwd)/output:/output gis-pipeline --demo
