"""
pipeline/tiler.py
Разбивка снимка на тайлы с перекрытием и сборка результатов.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine

log = logging.getLogger(__name__)


@dataclass
class Tile:
    """Один тайл снимка с геометрическим контекстом."""
    tile_id:    str
    data:       np.ndarray        # (C, H, W) uint16/float32
    transform:  Affine
    crs:        object
    row_off:    int               # смещение в пикселях от начала снимка
    col_off:    int
    height:     int
    width:      int
    overlap:    int               # перекрытие в пикселях (для NMS)
    indices:    dict = field(default_factory=dict)  # NDVI и пр. для этого тайла
    source_id:  str = ""


class TileManager:
    """Нарезает сцену на перекрывающиеся тайлы и собирает результаты."""

    def __init__(self, cfg: dict):
        tc = cfg.get("tiling", {})
        self.tile_size = tc.get("size", 256)
        self.overlap   = tc.get("overlap", 32)
        self.min_valid = tc.get("min_valid_px_ratio", 0.1)  # минимум valid пикселей

    # ── Публичный API ─────────────────────────────────────────────────────────

    def split(self, scene: dict) -> List[Tile]:
        """Нарезает сцену на тайлы с перекрытием."""
        data       = scene["data"]        # (C, H, W)
        transform  = scene["transform"]
        crs        = scene["crs"]
        indices    = scene.get("indices", {})
        source_id  = scene.get("source_id", "unknown")
        _, H, W    = data.shape

        step = self.tile_size - self.overlap
        tiles: List[Tile] = []

        rows = list(range(0, H - self.overlap, step))
        cols = list(range(0, W - self.overlap, step))

        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(cols):
                r_end = min(row + self.tile_size, H)
                c_end = min(col + self.tile_size, W)
                r_start, c_start = r_end - self.tile_size, c_end - self.tile_size
                r_start = max(0, r_start)
                c_start = max(0, c_start)

                tile_data = data[:, r_start:r_end, c_start:c_end]

                # Пропускаем тайлы с преобладанием nodata
                if not self._is_valid(tile_data):
                    continue

                # Пересчитываем аффинное преобразование для тайла
                tile_transform = _tile_transform(transform, r_start, c_start)

                # Вырезаем соответствующие индексы
                tile_indices = {
                    k: v[r_start:r_end, c_start:c_end]
                    for k, v in indices.items()
                }

                tile = Tile(
                    tile_id   = f"{source_id}_r{r_idx:04d}_c{c_idx:04d}",
                    data      = tile_data,
                    transform = tile_transform,
                    crs       = crs,
                    row_off   = r_start,
                    col_off   = c_start,
                    height    = r_end - r_start,
                    width     = c_end - c_start,
                    overlap   = self.overlap,
                    indices   = tile_indices,
                    source_id = source_id,
                )
                tiles.append(tile)

        log.info(
            f"Тайлинг: {H}×{W} px → {len(tiles)} тайлов "
            f"({self.tile_size}px, overlap={self.overlap}px)"
        )
        return tiles

    def pixel_to_geo(self, tile: Tile, row_px: int, col_px: int) -> Tuple[float, float]:
        """Конвертирует пиксельные координаты внутри тайла в географические (x, y)."""
        x = tile.transform.c + col_px * tile.transform.a
        y = tile.transform.f + row_px * tile.transform.e
        return x, y

    def pixel_bbox_to_geo_polygon(
        self, tile: Tile,
        r1: int, c1: int, r2: int, c2: int
    ):
        """
        Конвертирует bbox в пикселях [r1,c1,r2,c2] → координаты углов в CRS тайла.
        Возвращает список (x, y) — четыре угла прямоугольника.
        """
        corners_px = [(r1, c1), (r1, c2), (r2, c2), (r2, c1)]
        return [self.pixel_to_geo(tile, r, c) for r, c in corners_px]

    def pixel_point_to_geo(self, tile: Tile, row_px: int, col_px: int):
        """Центр объекта в пикселях → географические координаты."""
        return self.pixel_to_geo(tile, row_px, col_px)

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _is_valid(self, tile_data: np.ndarray) -> bool:
        """True если тайл содержит достаточно ненулевых пикселей."""
        total   = tile_data.shape[1] * tile_data.shape[2]
        nonzero = np.count_nonzero(tile_data[0])
        return nonzero / total >= self.min_valid


def _tile_transform(src_transform: Affine, row_off: int, col_off: int) -> Affine:
    """Сдвигает аффинное преобразование на (col_off, row_off) пикселей."""
    return src_transform * Affine.translation(col_off, row_off)
