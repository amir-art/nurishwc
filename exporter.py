"""
pipeline/exporter.py
Экспорт детекций в GIS-форматы: GeoJSON (обязательно), GeoPackage, Shapefile.

Структура каждого объекта по ТЗ:
  id, class, confidence, source, geometry [+ area_m2, length_m, date, ...]
CRS: задаётся из конфига, по умолчанию EPSG:4326.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from pyproj import Transformer

from .detector import RawDetection

log = logging.getLogger(__name__)

# Стандартные поля по ТЗ
REQUIRED_FIELDS = ["id", "class", "confidence", "source", "geometry"]
OPTIONAL_FIELDS = ["area_m2", "length_m", "date", "change_flag"]


class GeoExporter:
    """Конвертирует список RawDetection в GIS-файлы."""

    def __init__(self, cfg: dict, scene: dict):
        self.cfg       = cfg
        self.scene     = scene
        self.src_crs   = scene["crs"]
        export_crs_str = cfg.get("export", {}).get("crs", "EPSG:4326")
        self.export_crs = export_crs_str

        # Трансформер для перевода координат в экспортную CRS
        src_epsg = self._epsg_from_crs(self.src_crs)
        dst_epsg = int(export_crs_str.split(":")[-1])
        if src_epsg != dst_epsg:
            self.transformer = Transformer.from_crs(
                src_epsg, dst_epsg, always_xy=True
            )
        else:
            self.transformer = None

        self.today = str(date.today())

    # ── Публичный API ─────────────────────────────────────────────────────────

    def export_all(self, detections: List[RawDetection], output_dir: Path) -> dict:
        """Экспортирует во все включённые форматы."""
        if not detections:
            log.warning("Нет детекций для экспорта")
            return {}

        gdf = self._to_geodataframe(detections)
        paths = {}

        # ── GeoJSON (обязательный) ─────────────────────────────────────────
        gj_path = output_dir / "detections.geojson"
        self._save_geojson(gdf, gj_path)
        paths["GeoJSON"] = str(gj_path)

        # ── GeoJSON по классам ─────────────────────────────────────────────
        for cls in gdf["class"].unique():
            cls_gdf  = gdf[gdf["class"] == cls].copy()
            cls_path = output_dir / f"detections_{cls}.geojson"
            self._save_geojson(cls_gdf, cls_path)
            paths[f"GeoJSON_{cls}"] = str(cls_path)

        # ── GeoPackage (рекомендуется) ─────────────────────────────────────
        gpkg_path = output_dir / "detections.gpkg"
        self._save_gpkg(gdf, gpkg_path)
        paths["GeoPackage"] = str(gpkg_path)

        # ── Shapefile (опционально) ────────────────────────────────────────
        if self.cfg.get("export", {}).get("shapefile", True):
            shp_dir = output_dir / "shapefile"
            shp_dir.mkdir(exist_ok=True)
            shp_path = shp_dir / "detections.shp"
            self._save_shapefile(gdf, shp_path)
            paths["Shapefile"] = str(shp_path)

        # ── Сводный JSON (summary) ─────────────────────────────────────────
        summary = self._build_summary(gdf)
        sum_path = output_dir / "summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        paths["Summary"] = str(sum_path)

        log.info(f"Экспорт завершён: {len(gdf)} объектов в {len(paths)} файлов")
        return paths

    # ── Построение GeoDataFrame ───────────────────────────────────────────────

    def _to_geodataframe(self, detections: List[RawDetection]) -> gpd.GeoDataFrame:
        records = []
        for d in detections:
            geom = self._reproject_geometry(d.geometry)
            row = {
                "id":         d.det_id,
                "class":      d.class_name,
                "confidence": d.confidence,
                "source":     d.source_id,
                "date":       self.today,
                "geometry":   geom,
            }
            if d.area_m2 is not None:
                row["area_m2"] = d.area_m2
            if d.length_m is not None:
                row["length_m"] = d.length_m
            # Дополнительные атрибуты из детектора
            for k, v in (d.attributes or {}).items():
                row[f"attr_{k}"] = v

            records.append(row)

        gdf = gpd.GeoDataFrame(records, crs=self.export_crs)
        return gdf

    def _reproject_geometry(self, geom):
        """Перепроецирует геометрию в export_crs."""
        if self.transformer is None:
            return geom
        from shapely.ops import transform as shp_transform
        def _tr(x, y, z=None):
            xp, yp = self.transformer.transform(x, y)
            return (xp, yp) if z is None else (xp, yp, z)
        return shp_transform(_tr, geom)

    # ── Сохранение форматов ───────────────────────────────────────────────────

    def _save_geojson(self, gdf: gpd.GeoDataFrame, path: Path) -> None:
        """
        Сохраняет в GeoJSON с явным указанием CRS (name члена FeatureCollection).
        """
        gdf_out = gdf.copy()
        # Конвертируем все несериализуемые типы
        for col in gdf_out.columns:
            if col == "geometry":
                continue
            gdf_out[col] = gdf_out[col].apply(
                lambda v: float(v) if isinstance(v, (np.floating,)) else v
            )
        gdf_out.to_file(str(path), driver="GeoJSON")
        log.info(f"GeoJSON сохранён: {path} ({len(gdf_out)} объектов)")

    def _save_gpkg(self, gdf: gpd.GeoDataFrame, path: Path) -> None:
        try:
            gdf.to_file(str(path), driver="GPKG", layer="detections")
            log.info(f"GeoPackage сохранён: {path}")
        except Exception as e:
            log.warning(f"Не удалось сохранить GeoPackage: {e}")

    def _save_shapefile(self, gdf: gpd.GeoDataFrame, path: Path) -> None:
        try:
            # Shapefile не поддерживает смешанные типы геометрии — разделяем
            for geom_type, group in gdf.groupby(gdf.geometry.geom_type):
                type_path = path.parent / f"{path.stem}_{geom_type.lower()}.shp"
                group.to_file(str(type_path))
            log.info(f"Shapefile(s) сохранены: {path.parent}")
        except Exception as e:
            log.warning(f"Не удалось сохранить Shapefile: {e}")

    # ── Сводная таблица ────────────────────────────────────────────────────────

    def _build_summary(self, gdf: gpd.GeoDataFrame) -> dict:
        summary: dict = {
            "crs":            self.export_crs,
            "total_objects":  int(len(gdf)),
            "date":           self.today,
            "source":         self.scene.get("source_id", "unknown"),
            "classes":        {},
        }
        for cls in gdf["class"].unique():
            sub = gdf[gdf["class"] == cls]
            entry: dict = {
                "count": int(len(sub)),
                "mean_confidence": round(float(sub["confidence"].mean()), 1),
            }
            if "area_m2" in sub.columns:
                valid_area = sub["area_m2"].dropna()
                if len(valid_area) > 0:
                    entry["total_area_m2"]  = round(float(valid_area.sum()), 1)
                    entry["mean_area_m2"]   = round(float(valid_area.mean()), 1)
                    entry["total_area_ha"]  = round(float(valid_area.sum()) / 10000, 3)
            if "length_m" in sub.columns:
                valid_len = sub["length_m"].dropna()
                if len(valid_len) > 0:
                    entry["total_length_m"] = round(float(valid_len.sum()), 1)
                    entry["total_length_km"] = round(float(valid_len.sum()) / 1000, 3)
            summary["classes"][cls] = entry

        return summary

    # ── Утилиты ───────────────────────────────────────────────────────────────

    @staticmethod
    def _epsg_from_crs(crs) -> int:
        try:
            return int(str(crs).split(":")[-1])
        except Exception:
            return 32642

# Импорт numpy нужен для лямбды в _save_geojson
import numpy as np
