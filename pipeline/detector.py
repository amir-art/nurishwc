"""
pipeline/detector.py
Мульти-классовая детекция объектов по спектральным индексам и морфологии.

Классы:
  - vegetation   : NDVI-порог + морфология
  - water        : NDWI-порог
  - building     : яркость + текстура + NDBI
  - road         : линейная морфология

Для каждого тайла возвращает список RawDetection с геометрией в CRS снимка.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import cv2
from scipy import ndimage
from shapely.geometry import Point, Polygon, LineString, mapping
from shapely.affinity import scale

from .tiler import Tile, TileManager

log = logging.getLogger(__name__)

# ── Структура детекции ─────────────────────────────────────────────────────────

@dataclass
class RawDetection:
    det_id:     str
    class_name: str
    confidence: float          # 0–100
    geometry:   object         # shapely Point / Polygon / LineString
    geom_type:  str            # "Point" | "Polygon" | "LineString"
    area_m2:    Optional[float] = None
    length_m:   Optional[float] = None
    source_id:  str = ""
    tile_id:    str = ""
    attributes: dict = field(default_factory=dict)


# ── Базовый класс детектора ────────────────────────────────────────────────────

class BaseClassDetector:
    """Шаблон детектора одного класса."""

    CLASS_NAME = "unknown"
    GEOM_TYPE  = "Polygon"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._counter = 0

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        raise NotImplementedError

    def _next_id(self, tile: Tile) -> str:
        self._counter += 1
        return f"{self.CLASS_NAME}_{tile.source_id}_{self._counter:06d}"

    def _px_to_geo_polygon(
        self, tile: Tile,
        contour_px: np.ndarray,
        simplify_tol: float = 1.0
    ) -> Optional[Polygon]:
        """Контур в пикселях → Polygon в координатах CRS."""
        if len(contour_px) < 3:
            return None
        coords = []
        for pt in contour_px.reshape(-1, 2):
            col, row = pt
            x = tile.transform.c + col * tile.transform.a
            y = tile.transform.f + row * tile.transform.e
            coords.append((x, y))
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if simplify_tol:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            return poly if poly.is_valid and not poly.is_empty else None
        except Exception:
            return None

    def _px_to_geo_point(self, tile: Tile, row_px: float, col_px: float) -> Point:
        x = tile.transform.c + col_px * tile.transform.a
        y = tile.transform.f + row_px * tile.transform.e
        return Point(x, y)

    def _px_to_geo_line(self, tile: Tile, points_px: list) -> Optional[LineString]:
        coords = []
        for row, col in points_px:
            x = tile.transform.c + col * tile.transform.a
            y = tile.transform.f + row * tile.transform.e
            coords.append((x, y))
        return LineString(coords) if len(coords) >= 2 else None


# ── Детектор растительности ────────────────────────────────────────────────────

class VegetationDetector(BaseClassDetector):
    CLASS_NAME = "vegetation"
    GEOM_TYPE  = "Polygon"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dc = cfg.get("detection", {}).get("vegetation", {})
        self.ndvi_threshold   = dc.get("ndvi_threshold", 0.2)
        self.min_area_px      = dc.get("min_area_px", 50)
        self.max_area_px      = dc.get("max_area_px", 50000)

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        ndvi = tile.indices.get("NDVI")
        if ndvi is None:
            return []

        mask = (ndvi > self.ndvi_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_px or area_px > self.max_area_px:
                continue
            poly = self._px_to_geo_polygon(tile, cnt)
            if poly is None:
                continue
            # Уверенность на основе среднего NDVI в маске объекта
            obj_mask = np.zeros(ndvi.shape, np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 1, -1)
            mean_ndvi = float(ndvi[obj_mask == 1].mean()) if obj_mask.sum() > 0 else 0
            conf = min(100.0, max(0.0, (mean_ndvi - self.ndvi_threshold) / (1 - self.ndvi_threshold) * 100))

            px_size = abs(tile.transform.a)
            area_m2 = area_px * px_size ** 2

            detections.append(RawDetection(
                det_id    = self._next_id(tile),
                class_name= self.CLASS_NAME,
                confidence= round(conf, 1),
                geometry  = poly,
                geom_type = "Polygon",
                area_m2   = round(area_m2, 1),
                source_id = tile.source_id,
                tile_id   = tile.tile_id,
                attributes= {"mean_ndvi": round(mean_ndvi, 3)},
            ))
        return detections


# ── Детектор водоёмов ──────────────────────────────────────────────────────────

class WaterDetector(BaseClassDetector):
    CLASS_NAME = "water"
    GEOM_TYPE  = "Polygon"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dc = cfg.get("detection", {}).get("water", {})
        self.ndwi_threshold = dc.get("ndwi_threshold", 0.1)
        self.min_area_px    = dc.get("min_area_px", 30)

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        ndwi = tile.indices.get("NDWI")
        if ndwi is None:
            return []

        mask = (ndwi > self.ndwi_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_px:
                continue
            poly = self._px_to_geo_polygon(tile, cnt)
            if poly is None:
                continue

            obj_mask = np.zeros(ndwi.shape, np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 1, -1)
            mean_ndwi = float(ndwi[obj_mask == 1].mean()) if obj_mask.sum() > 0 else 0
            conf = min(100.0, max(0.0, mean_ndwi * 120))

            px_size = abs(tile.transform.a)
            area_m2 = area_px * px_size ** 2

            detections.append(RawDetection(
                det_id    = self._next_id(tile),
                class_name= self.CLASS_NAME,
                confidence= round(conf, 1),
                geometry  = poly,
                geom_type = "Polygon",
                area_m2   = round(area_m2, 1),
                source_id = tile.source_id,
                tile_id   = tile.tile_id,
                attributes= {"mean_ndwi": round(mean_ndwi, 3)},
            ))
        return detections


# ── Детектор зданий ────────────────────────────────────────────────────────────

class BuildingDetector(BaseClassDetector):
    CLASS_NAME = "building"
    GEOM_TYPE  = "Polygon"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dc = cfg.get("detection", {}).get("building", {})
        self.bright_pct    = dc.get("brightness_percentile", 80)
        self.min_area_px   = dc.get("min_area_px", 25)
        self.max_area_px   = dc.get("max_area_px", 8000)
        self.compactness   = dc.get("min_compactness", 0.15)

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        data = tile.data
        # Используем первые 3 полосы (RGB-подобные) или всё
        n = min(data.shape[0], 3)
        rgb = data[:n].astype(np.float32)
        brightness = rgb.mean(axis=0)

        ndvi = tile.indices.get("NDVI")
        ndwi = tile.indices.get("NDWI")

        # Яркие зоны — кандидаты на крыши
        thresh_val = np.percentile(brightness, self.bright_pct)
        mask_bright = (brightness > thresh_val).astype(np.uint8)

        # Исключаем воду и растительность
        if ndvi is not None:
            mask_bright[ndvi > 0.2] = 0
        if ndwi is not None:
            mask_bright[ndwi > 0.1] = 0

        # Морфология
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask_bright, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_px or area_px > self.max_area_px:
                continue

            # Фильтр по компактности (круглость) — отсекаем длинные нити
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compact = 4 * np.pi * area_px / (perimeter ** 2)
                if compact < self.compactness:
                    continue

            poly = self._px_to_geo_polygon(tile, cnt, simplify_tol=0.5)
            if poly is None:
                continue

            # Прямоугольник минимальной площади (ориентированный bbox)
            rect = cv2.minAreaRect(cnt)
            _, (rw, rh), angle = rect
            aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

            mean_bright = float(brightness[mask == 1].mean()) if mask.sum() > 0 else 0
            norm_bright = float(np.percentile(brightness, 98)) + 1e-5
            conf = min(100.0, max(0.0, mean_bright / norm_bright * 80 + 20))

            px_size = abs(tile.transform.a)
            area_m2 = area_px * px_size ** 2

            detections.append(RawDetection(
                det_id    = self._next_id(tile),
                class_name= self.CLASS_NAME,
                confidence= round(conf, 1),
                geometry  = poly,
                geom_type = "Polygon",
                area_m2   = round(area_m2, 1),
                source_id = tile.source_id,
                tile_id   = tile.tile_id,
                attributes= {
                    "aspect_ratio": round(aspect, 2),
                    "orientation":  round(float(angle), 1),
                    "compactness":  round(float(
                        4 * np.pi * area_px / (perimeter**2) if perimeter > 0 else 0
                    ), 3),
                },
            ))
        return detections


# ── Детектор дорог (линейные объекты) ─────────────────────────────────────────

class RoadDetector(BaseClassDetector):
    CLASS_NAME = "road"
    GEOM_TYPE  = "LineString"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dc = cfg.get("detection", {}).get("road", {})
        self.canny_low    = dc.get("canny_low", 50)
        self.canny_high   = dc.get("canny_high", 150)
        self.hough_thresh = dc.get("hough_threshold", 40)
        self.min_len_px   = dc.get("min_line_length_px", 30)
        self.max_gap_px   = dc.get("max_line_gap_px", 10)

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        data = tile.data
        n = min(data.shape[0], 3)
        rgb = data[:n].astype(np.float32)
        gray = rgb.mean(axis=0)

        # Нормализация в 8 бит для OpenCV
        g_min, g_max = gray.min(), gray.max()
        if g_max == g_min:
            return []
        gray8 = ((gray - g_min) / (g_max - g_min) * 255).astype(np.uint8)

        # Выравнивание гистограммы
        gray8 = cv2.equalizeHist(gray8)

        # Edges
        edges = cv2.Canny(gray8, self.canny_low, self.canny_high)

        # Probabilistic Hough
        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.min_len_px,
            maxLineGap=self.max_gap_px,
        )
        if lines is None:
            return []

        detections = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            geo_line = self._px_to_geo_line(tile, [(y1, x1), (y2, x2)])
            if geo_line is None:
                continue
            length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            px_size   = abs(tile.transform.a)
            length_m  = length_px * px_size
            conf      = min(100.0, 40 + length_px / self.min_len_px * 5)

            detections.append(RawDetection(
                det_id    = self._next_id(tile),
                class_name= self.CLASS_NAME,
                confidence= round(conf, 1),
                geometry  = geo_line,
                geom_type = "LineString",
                length_m  = round(length_m, 1),
                source_id = tile.source_id,
                tile_id   = tile.tile_id,
                attributes= {"angle_deg": round(
                    float(np.degrees(np.arctan2(y2-y1, x2-x1))), 1
                )},
            ))
        return detections


# ── Агрегатор всех детекторов ──────────────────────────────────────────────────

class MultiClassDetector:
    """Запускает все детекторы и объединяет результаты."""

    def __init__(self, cfg: dict):
        enabled = cfg.get("detection", {}).get("enabled_classes",
                          ["vegetation", "water", "building", "road"])
        self._detectors = []
        registry = {
            "vegetation": VegetationDetector,
            "water":      WaterDetector,
            "building":   BuildingDetector,
            "road":       RoadDetector,
        }
        for cls in enabled:
            if cls in registry:
                self._detectors.append(registry[cls](cfg))
                log.info(f"Детектор подключён: {cls}")

    def detect(self, tile: Tile, scene: dict) -> List[RawDetection]:
        results = []
        for det in self._detectors:
            try:
                results.extend(det.detect(tile, scene))
            except Exception as e:
                log.warning(f"Ошибка детектора {det.CLASS_NAME} "
                            f"на тайле {tile.tile_id}: {e}")
        return results
