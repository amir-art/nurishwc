"""
pipeline/change_detection.py
Обнаружение изменений между двумя снимками одной территории.

Методы:
  - Image Differencing (разность нормализованных каналов)
  - NDVI Change (для растительности)
  - CVA — Change Vector Analysis
  - Бинарная маска изменений + полигонизация

Результат: список ChangeDetection объектов с флагом direction (gain/loss/change).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RioResampling
from shapely.geometry import Polygon, mapping

log = logging.getLogger(__name__)


@dataclass
class ChangeDetection:
    """Один полигон изменений между двумя датами."""
    det_id:     str
    class_name: str          # "change_gain" | "change_loss" | "change"
    confidence: float
    geometry:   object       # shapely Polygon
    geom_type:  str = "Polygon"
    area_m2:    Optional[float] = None
    source_id:  str = ""
    tile_id:    str = ""
    attributes: dict = field(default_factory=dict)


class ChangeDetector:
    """
    Сравнивает два снимка (before/after) и выявляет зоны изменений.
    Оба снимка должны быть загружены через ImageLoader и иметь одинаковую CRS.
    При несовпадении размеров — after перепроецируется под before.
    """

    def __init__(self, cfg: dict):
        cc = cfg.get("change_detection", {})
        self.method          = cc.get("method", "ndvi_diff")  # ndvi_diff | cva | diff
        self.threshold       = cc.get("threshold", 0.15)
        self.min_area_px     = cc.get("min_area_px", 30)
        self.morph_kernel    = cc.get("morph_kernel_size", 5)
        self._counter        = 0

    # ── Публичный API ─────────────────────────────────────────────────────────

    def detect(self, scene_before: dict, scene_after: dict) -> List[ChangeDetection]:
        """
        Запускает детекцию изменений.
        Возвращает список ChangeDetection в CRS scene_before.
        """
        log.info(f"Анализ изменений: метод={self.method}")

        data_b = scene_before["data"].astype(np.float32)
        data_a = scene_after["data"].astype(np.float32)
        transform = scene_before["transform"]
        crs       = scene_before["crs"]

        # Выравниваем размеры
        data_a = self._align(data_a, data_b, scene_after, scene_before)

        if self.method == "ndvi_diff":
            change_map = self._ndvi_diff(
                data_b, data_a,
                scene_before.get("band_names", []),
                scene_after.get("band_names", []),
            )
        elif self.method == "cva":
            change_map = self._cva(data_b, data_a)
        else:
            change_map = self._band_diff(data_b, data_a)

        # Разделяем на gain / loss
        mask_gain = (change_map > self.threshold).astype(np.uint8)
        mask_loss = (change_map < -self.threshold).astype(np.uint8)

        detections: List[ChangeDetection] = []
        src_id = f"{scene_before.get('source_id','T1')}_vs_{scene_after.get('source_id','T2')}"

        detections += self._polygonize(
            mask_gain, transform, crs, "change_gain", src_id, change_map
        )
        detections += self._polygonize(
            mask_loss, transform, crs, "change_loss", src_id, change_map
        )

        log.info(
            f"Изменений обнаружено: {len(detections)} "
            f"(gain={sum(1 for d in detections if d.class_name=='change_gain')}, "
            f"loss={sum(1 for d in detections if d.class_name=='change_loss')})"
        )
        return detections

    # ── Методы вычисления карты изменений ─────────────────────────────────────

    def _ndvi_diff(
        self,
        before: np.ndarray, after: np.ndarray,
        bn_b: list, bn_a: list
    ) -> np.ndarray:
        """ΔNDVI = NDVI_after - NDVI_before."""
        eps = 1e-8

        def get_band(data, band_names, name):
            upper = [b.upper() for b in band_names]
            if name in upper:
                return data[upper.index(name)]
            # Fallback: первая полоса
            return data[0]

        nir_b = get_band(before, bn_b, "NIR")
        red_b = get_band(before, bn_b, "RED")
        nir_a = get_band(after,  bn_a, "NIR")
        red_a = get_band(after,  bn_a, "RED")

        ndvi_b = (nir_b - red_b) / (nir_b + red_b + eps)
        ndvi_a = (nir_a - red_a) / (nir_a + red_a + eps)
        return np.clip(ndvi_a - ndvi_b, -1, 1)

    def _cva(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Change Vector Analysis: magnitude изменения по всем каналам.
        Возвращает знаковую карту (+ / -) по первому каналу.
        """
        n = min(before.shape[0], after.shape[0])
        # Нормализация
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)

        nb = np.stack([norm(before[i]) for i in range(n)])
        na = np.stack([norm(after[i])  for i in range(n)])
        diff = na - nb
        magnitude = np.sqrt((diff ** 2).sum(axis=0))
        sign = np.sign(diff[0])  # знак по первому каналу
        return np.clip(magnitude * sign, -1, 1)

    def _band_diff(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Простая разность нормализованных средних по каналам."""
        n = min(before.shape[0], after.shape[0])
        mn_b = before[:n].mean(axis=0)
        mn_a = after[:n].mean(axis=0)
        p98 = max(mn_b.max(), mn_a.max()) + 1e-8
        return np.clip((mn_a - mn_b) / p98, -1, 1)

    # ── Полигонизация ─────────────────────────────────────────────────────────

    def _polygonize(
        self,
        mask: np.ndarray,
        transform,
        crs,
        cls: str,
        src_id: str,
        change_map: np.ndarray,
    ) -> List[ChangeDetection]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_px:
                continue

            poly = self._cnt_to_poly(cnt, transform)
            if poly is None:
                continue

            # Средний модуль изменения внутри контура
            obj_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 1, -1)
            mean_mag = float(np.abs(change_map[obj_mask == 1]).mean())
            conf = min(100.0, mean_mag / max(self.threshold, 1e-8) * 60)

            px_size = abs(transform.a)
            area_m2 = area_px * px_size ** 2

            self._counter += 1
            results.append(ChangeDetection(
                det_id    = f"{cls}_{src_id}_{self._counter:06d}",
                class_name= cls,
                confidence= round(conf, 1),
                geometry  = poly,
                geom_type = "Polygon",
                area_m2   = round(area_m2, 1),
                source_id = src_id,
                attributes= {
                    "mean_change_magnitude": round(mean_mag, 4),
                    "direction": "gain" if "gain" in cls else "loss",
                },
            ))
        return results

    def _cnt_to_poly(self, contour, transform) -> Optional[Polygon]:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            return None
        coords = []
        for col, row in pts:
            x = transform.c + col * transform.a
            y = transform.f + row * transform.e
            coords.append((x, y))
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.is_valid and not poly.is_empty else None
        except Exception:
            return None

    # ── Выравнивание ──────────────────────────────────────────────────────────

    def _align(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        scene_a: dict,
        scene_b: dict,
    ) -> np.ndarray:
        """Перепроецирует data_a под grid scene_b."""
        if data_a.shape == data_b.shape:
            return data_a

        log.info(
            f"Выравнивание снимков: {data_a.shape} → {data_b.shape}"
        )
        n   = min(data_a.shape[0], data_b.shape[0])
        H_b, W_b = data_b.shape[1], data_b.shape[2]
        out = np.zeros((n, H_b, W_b), dtype=data_a.dtype)
        for i in range(n):
            reproject(
                source      = data_a[i],
                destination = out[i],
                src_transform = scene_a["transform"],
                src_crs       = scene_a["crs"],
                dst_transform = scene_b["transform"],
                dst_crs       = scene_b["crs"],
                resampling    = RioResampling.bilinear,
            )
        return out
