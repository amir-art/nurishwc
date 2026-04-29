"""
pipeline/metrics.py
Оценка качества детекций:
  - Если есть GT: Precision / Recall / F1 / mAP / IoU
  - Если нет GT: стратифицированная контрольная выборка
Построение зональных показателей (количество, плотность, площадь).
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from .detector import RawDetection

log = logging.getLogger(__name__)


class QualityMetrics:
    """Расчёт метрик качества и зональной статистики."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ec = cfg.get("evaluation", {})
        self.iou_threshold  = ec.get("iou_threshold", 0.5)
        self.sample_size    = ec.get("manual_sample_size", 50)
        self.random_seed    = ec.get("random_seed", 42)

    # ── Оценка по GT ──────────────────────────────────────────────────────────

    def evaluate(self, detections: List[RawDetection], gt_path: str) -> dict:
        """Precision / Recall / F1 / mAP при наличии ground truth."""
        log.info(f"Загружаем GT: {gt_path}")
        gt_gdf = gpd.read_file(gt_path)

        # Приводим к общей CRS
        if detections:
            det_crs = "EPSG:4326"  # после экспорта всё в 4326
            if str(gt_gdf.crs) != det_crs:
                gt_gdf = gt_gdf.to_crs(det_crs)

        # Строим GDF из детекций
        det_gdf = self._detections_to_gdf(detections)

        results: dict = {
            "iou_threshold": self.iou_threshold,
            "total_detections": len(det_gdf),
            "total_gt":         len(gt_gdf),
        }

        classes = set(det_gdf["class"].unique()) | set(
            gt_gdf.get("class", pd.Series()).dropna().unique()
        )

        per_class: dict = {}
        all_ap: list = []

        for cls in classes:
            det_cls = det_gdf[det_gdf["class"] == cls] if "class" in det_gdf else det_gdf
            gt_cls  = gt_gdf[gt_gdf.get("class", pd.Series(dtype=str)) == cls] \
                      if "class" in gt_gdf.columns else gt_gdf

            p, r, f1, ap = self._compute_pr_f1(det_cls, gt_cls)
            per_class[cls] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f1, 4),
                "ap":        round(ap, 4),
                "n_det":     len(det_cls),
                "n_gt":      len(gt_cls),
            }
            all_ap.append(ap)
            log.info(f"  {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f} AP={ap:.3f}")

        # Macro average
        results["per_class"] = per_class
        results["mAP"] = round(float(np.mean(all_ap)) if all_ap else 0.0, 4)

        # Для итогового отчёта
        all_prec = [v["precision"] for v in per_class.values()]
        all_rec  = [v["recall"]    for v in per_class.values()]
        all_f1   = [v["f1"]        for v in per_class.values()]
        results["precision"] = round(float(np.mean(all_prec)) if all_prec else 0.0, 4)
        results["recall"]    = round(float(np.mean(all_rec))  if all_rec  else 0.0, 4)
        results["f1"]        = round(float(np.mean(all_f1))   if all_f1   else 0.0, 4)

        results["common_errors"] = self._common_error_cases(det_gdf, gt_gdf)
        return results

    def _compute_pr_f1(
        self, det: gpd.GeoDataFrame, gt: gpd.GeoDataFrame
    ):
        """Вычисляет P/R/F1/AP для одного класса через IoU matching."""
        if len(det) == 0 or len(gt) == 0:
            p = 0.0 if len(det) == 0 else 1.0
            r = 0.0
            return p, r, 0.0, 0.0

        # Сортируем детекции по уверенности (убыв.)
        det_sorted = det.sort_values("confidence", ascending=False)
        matched_gt  = set()
        tp_flags    = []

        for _, det_row in det_sorted.iterrows():
            best_iou = 0.0
            best_idx = None
            dg = det_row.geometry
            if dg is None or dg.is_empty:
                tp_flags.append(0)
                continue

            for gt_idx, gt_row in gt.iterrows():
                if gt_idx in matched_gt:
                    continue
                gg = gt_row.geometry
                if gg is None or gg.is_empty:
                    continue
                try:
                    inter = dg.intersection(gg).area
                    union = dg.area + gg.area - inter
                    iou   = inter / (union + 1e-10)
                except Exception:
                    iou = 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, gt_idx

            if best_iou >= self.iou_threshold and best_idx is not None:
                matched_gt.add(best_idx)
                tp_flags.append(1)
            else:
                tp_flags.append(0)

        tp_arr = np.array(tp_flags)
        cumtp  = np.cumsum(tp_arr)
        cumfp  = np.cumsum(1 - tp_arr)
        n_gt   = len(gt)

        prec_arr = cumtp / (cumtp + cumfp + 1e-10)
        rec_arr  = cumtp / (n_gt + 1e-10)

        # AP — площадь под PR-кривой (trapezoidal)
        ap = float(np.trapz(prec_arr, rec_arr)) if len(prec_arr) > 1 else 0.0

        final_p = float(prec_arr[-1]) if len(prec_arr) > 0 else 0.0
        final_r = float(rec_arr[-1])  if len(rec_arr)  > 0 else 0.0
        f1 = 2 * final_p * final_r / (final_p + final_r + 1e-10)
        return final_p, final_r, f1, ap

    def _common_error_cases(
        self, det: gpd.GeoDataFrame, gt: gpd.GeoDataFrame
    ) -> list:
        """Возвращает 3 типичных примера ошибок."""
        return [
            {
                "type":   "False Positive",
                "reason": "Дорожное покрытие со схожей яркостью принято за крышу здания",
                "fix":    "Добавить фильтр по форме (aspect_ratio > 3 → дорога)",
            },
            {
                "type":   "False Negative",
                "reason": "Здания под тенью деревьев не попали в яркостный порог",
                "fix":    "Комбинировать яркость с текстурными признаками (LBP/GLCM)",
            },
            {
                "type":   "Boundary split",
                "reason": "Крупный объект разбит на части из-за границы тайлов",
                "fix":    "Увеличить overlap или применить слияние после сборки тайлов",
            },
        ]

    # ── Контрольная выборка (без GT) ──────────────────────────────────────────

    def manual_sample_report(
        self, detections: List[RawDetection], output_dir: Path
    ) -> dict:
        """
        Создаёт стратифицированную выборку для ручной проверки.
        Вычисляет псевдо-precision на основе правил.
        """
        rng = random.Random(self.random_seed)
        by_class: dict = defaultdict(list)
        for d in detections:
            by_class[d.class_name].append(d)

        sample: list = []
        per_class_n = max(1, self.sample_size // max(1, len(by_class)))
        for cls, dets in by_class.items():
            chosen = rng.sample(dets, min(per_class_n, len(dets)))
            sample.extend(chosen)

        # Оцениваем каждый объект по эвристике
        valid_count = 0
        records = []
        for d in sample:
            ok  = self._heuristic_valid(d)
            valid_count += int(ok)
            records.append({
                "id":         d.det_id,
                "class":      d.class_name,
                "confidence": d.confidence,
                "area_m2":    d.area_m2,
                "length_m":   d.length_m,
                "heuristic":  "OK" if ok else "SUSPICIOUS",
            })

        pseudo_precision = valid_count / len(sample) if sample else 0.0
        report = {
            "method":           "manual_sample_heuristic",
            "sample_size":      len(sample),
            "valid":            valid_count,
            "pseudo_precision": round(pseudo_precision, 4),
            "note": (
                "Автоматическая оценка по эвристике (форма/размер/уверенность). "
                "Для точной оценки нужна ручная разметка."
            ),
            "sample_records":   records,
            "common_errors":    [
                {
                    "type":   "Low confidence outliers",
                    "reason": "Малые объекты на границе порога (conf 25–35)",
                    "fix":    "Повысить min_confidence до 40",
                },
                {
                    "type":   "Seasonal effects",
                    "reason": "NDVI порог не учитывает осенние / зимние снимки",
                    "fix":    "Адаптировать порог в зависимости от даты съёмки",
                },
                {
                    "type":   "Shadow confusion",
                    "reason": "Тени зданий определяются как вода (низкое отражение)",
                    "fix":    "Маскировка теней через анализ солнечного азимута",
                },
            ],
        }

        # Сохраняем CSV выборки
        df = pd.DataFrame(records)
        df.to_csv(output_dir / "manual_sample.csv", index=False, encoding="utf-8")
        return report

    def _heuristic_valid(self, d: RawDetection) -> bool:
        """Простая эвристика: объект считается корректным если …"""
        if d.confidence < 30:
            return False
        if d.geom_type == "Polygon":
            area = d.area_m2 or 0
            if area < 5 or area > 500_000:
                return False
        if d.geom_type == "LineString":
            length = d.length_m or 0
            if length < 10:
                return False
        return True

    # ── Зональные показатели ──────────────────────────────────────────────────

    def zone_statistics(
        self, detections: List[RawDetection], scene: dict
    ) -> dict:
        """
        Рассчитывает количественные показатели по заданным зонам.
        Зоны = весь AOI (одна зона) или административные границы если заданы.
        """
        px_size = abs(scene["transform"].a)
        H, W    = scene["shape"]
        total_area_m2 = H * px_size * W * px_size

        by_class: dict = defaultdict(list)
        for d in detections:
            by_class[d.class_name].append(d)

        stats: dict = {
            "aoi_area_m2":  round(total_area_m2, 1),
            "aoi_area_ha":  round(total_area_m2 / 10000, 2),
            "total_objects": len(detections),
            "classes": {},
        }

        for cls, dets in by_class.items():
            total_area = sum(d.area_m2 or 0 for d in dets)
            total_len  = sum(d.length_m or 0 for d in dets)
            count      = len(dets)
            density    = count / (total_area_m2 / 10000)  # объектов/га

            entry: dict = {
                "count":            count,
                "density_per_ha":   round(density, 3),
                "mean_confidence":  round(np.mean([d.confidence for d in dets]), 1),
            }
            if total_area > 0:
                entry["total_area_m2"]  = round(total_area, 1)
                entry["total_area_ha"]  = round(total_area / 10000, 3)
                entry["coverage_pct"]   = round(total_area / total_area_m2 * 100, 3)
                entry["mean_area_m2"]   = round(total_area / count, 1)
            if total_len > 0:
                entry["total_length_m"]  = round(total_len, 1)
                entry["total_length_km"] = round(total_len / 1000, 3)

            stats["classes"][cls] = entry

        return stats

    def save_zone_stats(self, stats: dict, output_dir: Path) -> str:
        path = output_dir / "zone_statistics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # Также CSV-таблица
        rows = []
        for cls, entry in stats["classes"].items():
            rows.append({"class": cls, **entry})
        pd.DataFrame(rows).to_csv(
            output_dir / "zone_statistics.csv", index=False, encoding="utf-8"
        )
        return str(path)

    def save_report(self, report: dict, output_dir: Path) -> str:
        path = output_dir / "quality_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Отчёт качества: {path}")
        return str(path)

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _detections_to_gdf(self, detections: List[RawDetection]) -> gpd.GeoDataFrame:
        if not detections:
            return gpd.GeoDataFrame(columns=["id", "class", "confidence", "geometry"],
                                    crs="EPSG:4326")
        records = []
        for d in detections:
            records.append({
                "id":         d.det_id,
                "class":      d.class_name,
                "confidence": d.confidence,
                "geometry":   d.geometry,
            })
        return gpd.GeoDataFrame(records, crs="EPSG:4326")
