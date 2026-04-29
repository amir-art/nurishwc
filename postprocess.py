"""
pipeline/postprocess.py
Постобработка детекций:
  - фильтрация по уверенности / размеру
  - Non-Maximum Suppression (NMS) для полигонов — удаление дублей
  - пространственная фильтрация выбросов
  - финальное присвоение глобальных ID
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

from .detector import RawDetection

log = logging.getLogger(__name__)


class PostProcessor:
    """Применяет цепочку фильтров к сырым детекциям."""

    def __init__(self, cfg: dict):
        ppc = cfg.get("postprocess", {})
        self.min_confidence  = ppc.get("min_confidence", 25.0)
        self.iou_threshold   = ppc.get("nms_iou_threshold", 0.4)
        self.min_area_m2     = ppc.get("min_area_m2", 10.0)
        self.min_length_m    = ppc.get("min_length_m", 5.0)
        self.max_area_m2     = ppc.get("max_area_m2", 1_000_000.0)
        self.merge_overlap   = ppc.get("merge_touching", True)

    # ── Публичный API ─────────────────────────────────────────────────────────

    def run(self, detections: List[RawDetection], scene: dict) -> List[RawDetection]:
        n0 = len(detections)
        if not detections:
            return []

        # 1. Фильтр по уверенности
        detections = self._filter_confidence(detections)
        log.info(f"После фильтра уверенности: {len(detections)} (было {n0})")

        # 2. Фильтр по размеру / длине
        detections = self._filter_size(detections)
        log.info(f"После фильтра размера: {len(detections)}")

        # 3. NMS по каждому классу (удаляем дубли из перекрытий тайлов)
        detections = self._nms_per_class(detections)
        log.info(f"После NMS: {len(detections)}")

        # 4. Объединение касающихся полигонов одного класса
        if self.merge_overlap:
            detections = self._merge_touching(detections)
            log.info(f"После слияния касающихся: {len(detections)}")

        # 5. Финальная геометрическая валидация
        detections = self._validate_geometry(detections)
        log.info(f"После геовалидации: {len(detections)}")

        # 6. Пересчёт площади/длины и присвоение глобальных ID
        detections = self._assign_ids_and_fix_metrics(detections, scene)

        return detections

    # ── Шаги постобработки ────────────────────────────────────────────────────

    def _filter_confidence(self, dets: List[RawDetection]) -> List[RawDetection]:
        return [d for d in dets if d.confidence >= self.min_confidence]

    def _filter_size(self, dets: List[RawDetection]) -> List[RawDetection]:
        result = []
        for d in dets:
            if d.geom_type == "Polygon":
                area = d.area_m2 or 0
                if self.min_area_m2 <= area <= self.max_area_m2:
                    result.append(d)
            elif d.geom_type == "LineString":
                length = d.length_m or 0
                if length >= self.min_length_m:
                    result.append(d)
            else:
                result.append(d)
        return result

    def _nms_per_class(self, dets: List[RawDetection]) -> List[RawDetection]:
        """IoU-based NMS для полигонов каждого класса отдельно."""
        by_class: Dict[str, List[RawDetection]] = defaultdict(list)
        non_poly: List[RawDetection] = []

        for d in dets:
            if d.geom_type == "Polygon":
                by_class[d.class_name].append(d)
            else:
                non_poly.append(d)

        survived = list(non_poly)
        for cls, cls_dets in by_class.items():
            survived.extend(self._nms_polygons(cls_dets))
        return survived

    def _nms_polygons(self, dets: List[RawDetection]) -> List[RawDetection]:
        if not dets:
            return []
        # Сортируем по убыванию уверенности
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, d_i in enumerate(dets):
            if i in suppressed:
                continue
            keep.append(d_i)
            geom_i = d_i.geometry
            if not geom_i.is_valid:
                continue
            area_i = geom_i.area + 1e-10

            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue
                geom_j = dets[j].geometry
                if not geom_j.is_valid:
                    continue
                try:
                    inter = geom_i.intersection(geom_j).area
                    union = area_i + geom_j.area - inter
                    iou = inter / (union + 1e-10)
                    if iou >= self.iou_threshold:
                        suppressed.add(j)
                except Exception:
                    pass
        return keep

    def _merge_touching(self, dets: List[RawDetection]) -> List[RawDetection]:
        """
        Объединяет перекрывающиеся / касающиеся полигоны одного класса.
        Буфер 1м перед объединением чтобы захватить касающиеся.
        """
        by_class: Dict[str, List[RawDetection]] = defaultdict(list)
        non_poly: List[RawDetection] = []

        for d in dets:
            if d.geom_type == "Polygon":
                by_class[d.class_name].append(d)
            else:
                non_poly.append(d)

        merged = list(non_poly)
        for cls, cls_dets in by_class.items():
            merged.extend(self._merge_class_polys(cls, cls_dets))
        return merged

    def _merge_class_polys(
        self, cls: str, dets: List[RawDetection]
    ) -> List[RawDetection]:
        if not dets:
            return []

        buf = 1.0  # 1 метр буфер
        buffered = [d.geometry.buffer(buf) for d in dets]

        # Итеративное объединение касающихся групп
        groups: List[List[int]] = []
        assigned = [False] * len(dets)

        for i in range(len(dets)):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, len(dets)):
                if assigned[j]:
                    continue
                if any(buffered[k].intersects(buffered[j]) for k in group):
                    group.append(j)
                    assigned[j] = True
            groups.append(group)

        result = []
        for g in groups:
            if len(g) == 1:
                result.append(dets[g[0]])
            else:
                polys     = [dets[k].geometry for k in g]
                merged_g  = unary_union(polys).buffer(0)
                if merged_g.is_empty:
                    continue
                # Берём максимальную уверенность группы
                best      = max(g, key=lambda k: dets[k].confidence)
                best_det  = dets[best]
                result.append(RawDetection(
                    det_id    = best_det.det_id,
                    class_name= cls,
                    confidence= best_det.confidence,
                    geometry  = merged_g,
                    geom_type = "Polygon",
                    area_m2   = round(merged_g.area, 1),
                    source_id = best_det.source_id,
                    tile_id   = best_det.tile_id,
                    attributes= {**best_det.attributes, "merged_from": len(g)},
                ))
        return result

    def _validate_geometry(self, dets: List[RawDetection]) -> List[RawDetection]:
        result = []
        for d in dets:
            g = d.geometry
            if g is None or g.is_empty:
                continue
            if not g.is_valid:
                g = g.buffer(0)
                if not g.is_valid or g.is_empty:
                    continue
                d.geometry = g
            result.append(d)
        return result

    def _assign_ids_and_fix_metrics(
        self, dets: List[RawDetection], scene: dict
    ) -> List[RawDetection]:
        """Назначает глобальные ID и пересчитывает area/length."""
        for i, d in enumerate(dets, 1):
            d.det_id = f"{d.class_name}_{d.source_id}_{i:06d}"
            if d.geom_type == "Polygon" and d.geometry is not None:
                d.area_m2 = round(d.geometry.area, 1)
            elif d.geom_type == "LineString" and d.geometry is not None:
                d.length_m = round(d.geometry.length, 1)
        return dets
