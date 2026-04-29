"""
pipeline/visualizer.py
Визуализация результатов:
  - обзорная карта (RGB + контуры детекций)
  - карта по каждому классу
  - карта плотности (heatmap)
  - скриншоты для отчёта
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import cv2

from .detector import RawDetection

log = logging.getLogger(__name__)

# Палитра классов
CLASS_COLORS: Dict[str, str] = {
    "vegetation": "#2ecc71",   # зелёный
    "water":      "#3498db",   # синий
    "building":   "#e74c3c",   # красный
    "road":       "#f39c12",   # оранжевый
    "unknown":    "#95a5a6",   # серый
}

DPI = 150


class ResultVisualizer:
    """Создаёт PNG-скриншоты результатов."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_objects = cfg.get("visualization", {}).get("max_objects_per_plot", 5000)

    # ── Публичный API ─────────────────────────────────────────────────────────

    def render_overview(
        self, scene: dict, detections: List[RawDetection], output_dir: Path
    ) -> str:
        """Обзорная карта: RGB + контуры всех классов."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        rgb = self._get_rgb(scene)
        ax.imshow(rgb, extent=self._extent(scene), aspect="equal")

        shown = 0
        for d in detections:
            if shown >= self.max_objects:
                break
            color = CLASS_COLORS.get(d.class_name, "#95a5a6")
            try:
                self._draw_detection(ax, d, color, alpha=0.4, scene=scene)
                shown += 1
            except Exception:
                pass

        # Легенда
        legend_patches = []
        classes = {d.class_name for d in detections}
        for cls in sorted(classes):
            col = CLASS_COLORS.get(cls, "#95a5a6")
            legend_patches.append(mpatches.Patch(color=col, label=cls))
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

        ax.set_title(
            f"Обзор: {scene.get('source_id', 'сцена')} | "
            f"объектов: {len(detections)}", fontsize=11
        )
        ax.set_xlabel(f"X ({scene['crs']})")
        ax.set_ylabel("Y")
        ax.tick_params(labelsize=7)

        path = output_dir / "overview.png"
        fig.tight_layout()
        fig.savefig(str(path), dpi=DPI)
        plt.close(fig)
        log.info(f"Обзор сохранён: {path}")
        return str(path)

    def render_per_class(
        self, scene: dict, detections: List[RawDetection], output_dir: Path
    ) -> List[str]:
        """Отдельная карта для каждого класса."""
        by_class: Dict[str, List[RawDetection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name, []).append(d)

        paths = []
        for cls, cls_dets in by_class.items():
            path = self._render_single_class(scene, cls, cls_dets, output_dir)
            paths.append(path)
        return paths

    def render_density_map(
        self, detections: List[RawDetection], scene: dict, output_dir: Path
    ) -> str:
        """Тепловая карта плотности объектов (kernel density)."""
        H, W = scene["shape"]
        transform = scene["transform"]

        density = np.zeros((H, W), dtype=np.float32)

        for d in detections:
            if d.geometry is None:
                continue
            try:
                centroid = d.geometry.centroid
                col = int((centroid.x - transform.c) / transform.a)
                row = int((centroid.y - transform.f) / transform.e)
                if 0 <= row < H and 0 <= col < W:
                    density[row, col] += 1.0
            except Exception:
                pass

        # Gaussian blur как KDE
        if density.max() > 0:
            sigma_px = max(5, min(H, W) // 40)
            density  = cv2.GaussianBlur(density, (0, 0), sigma_px)

        fig, ax = plt.subplots(1, 1, figsize=(11, 9))
        ext = self._extent(scene)
        im = ax.imshow(
            density, cmap="hot_r", extent=ext, aspect="equal",
            vmin=0, vmax=density.max() or 1
        )
        plt.colorbar(im, ax=ax, label="Плотность объектов")
        ax.set_title("Карта плотности объектов", fontsize=12)
        ax.set_xlabel(f"X ({scene['crs']})")
        ax.set_ylabel("Y")

        path = output_dir / "density_map.png"
        fig.tight_layout()
        fig.savefig(str(path), dpi=DPI)
        plt.close(fig)
        log.info(f"Карта плотности: {path}")
        return str(path)

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _render_single_class(
        self, scene: dict, cls: str,
        dets: List[RawDetection], output_dir: Path
    ) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Левая: RGB + объекты класса
        rgb = self._get_rgb(scene)
        axes[0].imshow(rgb, extent=self._extent(scene), aspect="equal")
        color = CLASS_COLORS.get(cls, "#95a5a6")
        for d in dets[:self.max_objects]:
            try:
                self._draw_detection(axes[0], d, color, alpha=0.5, scene=scene)
            except Exception:
                pass
        axes[0].set_title(f"{cls} — {len(dets)} объектов", fontsize=10)
        axes[0].set_xlabel(f"X ({scene['crs']})")
        axes[0].set_ylabel("Y")

        # Правая: гистограмма уверенности
        confs = [d.confidence for d in dets]
        axes[1].hist(confs, bins=20, color=color, edgecolor="white", alpha=0.85)
        axes[1].axvline(np.mean(confs), color="black", linestyle="--",
                        label=f"среднее={np.mean(confs):.1f}")
        axes[1].set_title(f"Распределение уверенности ({cls})", fontsize=10)
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Количество")
        axes[1].legend(fontsize=8)

        path = output_dir / f"class_{cls}.png"
        fig.tight_layout()
        fig.savefig(str(path), dpi=DPI)
        plt.close(fig)
        log.info(f"Карта класса {cls}: {path}")
        return str(path)

    def _get_rgb(self, scene: dict) -> np.ndarray:
        """Извлекает RGB из сцены и нормализует в [0,1]."""
        data = scene["data"]
        bnames = [b.upper() for b in scene.get("band_names", [])]

        def idx(name):
            return bnames.index(name) if name in bnames else None

        r_i = idx("RED")   or 0
        g_i = idx("GREEN") or min(1, data.shape[0]-1)
        b_i = idx("BLUE")  or min(2, data.shape[0]-1)

        r = data[r_i].astype(np.float32)
        g = data[g_i].astype(np.float32)
        b = data[b_i].astype(np.float32)

        def norm(ch):
            p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
            if p98 == p2:
                return np.zeros_like(ch)
            return np.clip((ch - p2) / (p98 - p2), 0, 1)

        return np.dstack([norm(r), norm(g), norm(b)])

    def _extent(self, scene: dict):
        """Вычисляет extent для imshow."""
        t = scene["transform"]
        H, W = scene["shape"]
        left  = t.c
        right = t.c + W * t.a
        top   = t.f
        bot   = t.f + H * t.e
        return [left, right, bot, top]

    def _draw_detection(
        self, ax, d: RawDetection, color: str, alpha: float, scene: dict
    ) -> None:
        """Рисует объект на осях matplotlib."""
        geom = d.geometry
        if geom is None or geom.is_empty:
            return

        gtype = d.geom_type

        if gtype == "Point":
            ax.plot(*geom.coords[0], "o", color=color,
                    markersize=4, alpha=alpha)

        elif gtype == "Polygon":
            coords = np.array(geom.exterior.coords)
            patch = mpatches.Polygon(
                coords, closed=True,
                facecolor=color, edgecolor=color,
                alpha=alpha, linewidth=0.5
            )
            ax.add_patch(patch)

        elif gtype == "LineString":
            coords = np.array(geom.coords)
            ax.plot(coords[:, 0], coords[:, 1],
                    "-", color=color, linewidth=1.2, alpha=alpha + 0.2)

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords)
                patch = mpatches.Polygon(
                    coords, closed=True,
                    facecolor=color, edgecolor=color,
                    alpha=alpha, linewidth=0.5
                )
                ax.add_patch(patch)
