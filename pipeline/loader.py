"""
pipeline/loader.py
Загрузка и подготовка космических снимков (GeoTIFF).
Поддерживает: обрезку по AOI, нормализацию, геопривязку.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from shapely.geometry import box, mapping

log = logging.getLogger(__name__)

# Стандартные имена полос для типовых спутников
BAND_CONFIGS = {
    "sentinel2": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    "landsat8":  ["Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    "rgb":       ["Red", "Green", "Blue"],
    "default":   None,  # auto-numbered
}


class ImageLoader:
    """Загружает GeoTIFF, приводит к целевой CRS, обрезает по AOI."""

    TARGET_CRS = "EPSG:4326"   # WGS84 для итоговых слоёв
    PROC_CRS   = "EPSG:32642"  # UTM зона 42N (Казахстан) — для метрических вычислений

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.satellite = cfg.get("satellite", "default")
        self.band_names = BAND_CONFIGS.get(self.satellite) or []
        self.target_crs = cfg.get("crs", {}).get("target", self.TARGET_CRS)
        self.proc_crs   = cfg.get("crs", {}).get("processing", self.PROC_CRS)

    # ── Публичный API ─────────────────────────────────────────────────────────

    def load(self, image_path: str, aoi_path: Optional[str] = None) -> dict:
        """Загружает снимок, при необходимости обрезает по AOI."""
        image_path = str(image_path)
        log.info(f"Открываем снимок: {image_path}")

        with rasterio.open(image_path) as src:
            self._validate_source(src)
            src_crs = src.crs or CRS.from_epsg(32642)

            # Обрезка по AOI
            if aoi_path:
                aoi_geom = self._load_aoi(aoi_path, src_crs)
                data, transform = self._clip_to_aoi(src, aoi_geom)
                crs = src_crs
            else:
                data = src.read()
                transform = src.transform
                crs = src_crs

            meta = src.meta.copy()
            meta.update({"count": data.shape[0], "transform": transform,
                         "height": data.shape[1], "width": data.shape[2]})

        # Приводим к единой CRS для обработки
        data, transform, crs = self._reproject_if_needed(
            data, transform, crs, self.proc_crs
        )

        band_names = self.band_names or [f"B{i+1}" for i in range(data.shape[0])]

        scene = {
            "data":        data,          # np.ndarray (C, H, W)
            "transform":   transform,
            "crs":         crs,
            "meta":        meta,
            "band_names":  band_names,
            "shape":       (data.shape[1], data.shape[2]),
            "n_bands":     data.shape[0],
            "source_path": image_path,
            "source_id":   Path(image_path).stem,
        }

        # Вычисляем спектральные индексы
        scene["indices"] = self._compute_indices(data, band_names)

        log.info(f"Снимок подготовлен: shape={scene['shape']}, "
                 f"CRS={crs}, bands={band_names}")
        return scene

    def load_demo_scene(self, output_dir: Path) -> dict:
        """
        Генерирует синтетический снимок для демо-запуска.
        Создаёт реальный GeoTIFF с геопривязкой.
        """
        log.info("Генерируем демо-снимок (синтетика Алматы)...")
        H, W = 512, 512
        n_bands = 4  # R, G, B, NIR

        # Синтетика: фоновый шум + структуры (здания, вода, растительность)
        rng = np.random.default_rng(42)
        data = rng.integers(200, 3000, (n_bands, H, W), dtype=np.uint16)

        # Здания — прямоугольные зоны с высоким отражением
        for _ in range(30):
            r, c = rng.integers(20, H-40), rng.integers(20, W-40)
            h_, w_ = rng.integers(15, 45), rng.integers(15, 45)
            data[:3, r:r+h_, c:c+w_] = rng.integers(2500, 3500, (3, h_, w_))
            data[3, r:r+h_, c:c+w_] = rng.integers(800, 1200, (h_, w_))

        # Вода — низкое NIR, высокое Blue
        for _ in range(5):
            r, c = rng.integers(50, H-60), rng.integers(50, W-60)
            h_, w_ = rng.integers(20, 60), rng.integers(40, 100)
            data[0, r:r+h_, c:c+w_] = rng.integers(400, 700, (h_, w_))
            data[1, r:r+h_, c:c+w_] = rng.integers(600, 900, (h_, w_))
            data[2, r:r+h_, c:c+w_] = rng.integers(1200, 1800, (h_, w_))
            data[3, r:r+h_, c:c+w_] = rng.integers(100, 300, (h_, w_))

        # Растительность — высокое NIR
        for _ in range(15):
            r, c = rng.integers(10, H-50), rng.integers(10, W-50)
            h_, w_ = rng.integers(20, 80), rng.integers(20, 80)
            data[0, r:r+h_, c:c+w_] = rng.integers(300, 600, (h_, w_))
            data[1, r:r+h_, c:c+w_] = rng.integers(800, 1200, (h_, w_))
            data[2, r:r+h_, c:c+w_] = rng.integers(300, 600, (h_, w_))
            data[3, r:r+h_, c:c+w_] = rng.integers(2000, 3500, (h_, w_))

        # Алматы примерные координаты (UTM 42N)
        west,  south = 770000.0, 4820000.0
        east,  north = west + W * 10, south + H * 10  # 10м/пиксель
        transform = from_bounds(west, south, east, north, W, H)
        crs = CRS.from_epsg(32642)

        # Сохраняем GeoTIFF
        demo_path = output_dir / "demo_scene.tif"
        with rasterio.open(
            demo_path, "w",
            driver="GTiff", height=H, width=W, count=n_bands,
            dtype="uint16", crs=crs, transform=transform,
        ) as dst:
            dst.write(data)

        log.info(f"Демо-снимок сохранён: {demo_path}")
        band_names = ["Red", "Green", "Blue", "NIR"]
        return {
            "data":        data,
            "transform":   transform,
            "crs":         crs,
            "meta":        {},
            "band_names":  band_names,
            "shape":       (H, W),
            "n_bands":     n_bands,
            "source_path": str(demo_path),
            "source_id":   "demo_almaty",
            "indices":     self._compute_indices(data, band_names),
        }

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _validate_source(self, src) -> None:
        if src.count == 0:
            raise ValueError("Снимок не содержит полос данных")
        if src.width < 64 or src.height < 64:
            raise ValueError(f"Снимок слишком маленький: {src.width}×{src.height}")

    def _load_aoi(self, aoi_path: str, target_crs) -> list:
        gdf = gpd.read_file(aoi_path)
        if str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
        return [mapping(geom) for geom in gdf.geometry]

    def _clip_to_aoi(self, src, aoi_geoms: list):
        data, transform = rio_mask(src, aoi_geoms, crop=True)
        return data, transform

    def _reproject_if_needed(self, data, transform, src_crs, dst_crs_str):
        dst_crs = CRS.from_string(dst_crs_str)
        if src_crs == dst_crs:
            return data, transform, src_crs

        log.info(f"Перепроецируем: {src_crs} → {dst_crs}")
        n, H, W = data.shape
        new_transform, new_W, new_H = calculate_default_transform(
            src_crs, dst_crs, W, H, *rasterio.transform.array_bounds(H, W, transform)
        )
        reprojected = np.zeros((n, new_H, new_W), dtype=data.dtype)
        for band_i in range(n):
            reproject(
                source=data[band_i], destination=reprojected[band_i],
                src_transform=transform, src_crs=src_crs,
                dst_transform=new_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        return reprojected, new_transform, dst_crs

    def _compute_indices(self, data: np.ndarray, band_names: list) -> dict:
        """Вычисляет NDVI, NDWI, NDBI по доступным полосам."""
        indices = {}
        bn = [b.upper() for b in band_names]
        eps = 1e-8

        def safe_norm(a, b):
            fa, fb = a.astype(np.float32), b.astype(np.float32)
            return (fa - fb) / (fa + fb + eps)

        def get(name):
            for variant in [name, name.capitalize(), name.lower()]:
                if variant in bn:
                    return data[bn.index(variant)].astype(np.float32)
            return None

        nir, red = get("NIR"), get("RED")
        green, swir = get("GREEN"), get("SWIR1")

        if nir is not None and red is not None:
            indices["NDVI"] = np.clip(safe_norm(nir, red), -1, 1)
        if nir is not None and green is not None:
            indices["NDWI"] = np.clip(safe_norm(green, nir), -1, 1)
        if nir is not None and swir is not None:
            indices["NDBI"] = np.clip(safe_norm(swir, nir), -1, 1)

        log.info(f"Вычислены индексы: {list(indices.keys())}")
        return indices
