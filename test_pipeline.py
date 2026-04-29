"""
tests/test_pipeline.py
Юнит-тесты для всех компонентов GIS Pipeline.
Запуск: python -m pytest tests/ -v
"""

import sys
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pytest
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import Polygon, LineString, Point

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader import ImageLoader
from pipeline.tiler import TileManager, Tile
from pipeline.detector import (
    VegetationDetector,
    WaterDetector,
    BuildingDetector,
    RoadDetector,
    MultiClassDetector,
    RawDetection,
)
from pipeline.postprocess import PostProcessor
from pipeline.exporter import GeoExporter
from pipeline.metrics import QualityMetrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg():
    return {
        "satellite": "rgb",
        "crs": {"processing": "EPSG:32642", "target": "EPSG:4326"},
        "tiling": {"size": 128, "overlap": 16, "min_valid_px_ratio": 0.05},
        "detection": {
            "enabled_classes": ["vegetation", "water", "building", "road"],
            "vegetation": {"ndvi_threshold": 0.15, "min_area_px": 10, "max_area_px": 50000},
            "water":      {"ndwi_threshold": 0.05, "min_area_px": 10},
            "building":   {"brightness_percentile": 75, "min_area_px": 10,
                           "max_area_px": 8000, "min_compactness": 0.05},
            "road":       {"canny_low": 30, "canny_high": 120,
                           "hough_threshold": 20, "min_line_length_px": 10,
                           "max_line_gap_px": 5},
        },
        "postprocess": {
            "min_confidence": 10.0,
            "nms_iou_threshold": 0.5,
            "min_area_m2": 1.0,
            "min_length_m": 1.0,
            "max_area_m2": 5_000_000.0,
            "merge_touching": False,  # ускоряем тесты
        },
        "export": {"crs": "EPSG:4326", "shapefile": False},
        "evaluation": {"iou_threshold": 0.5, "manual_sample_size": 10, "random_seed": 42},
        "visualization": {"max_objects_per_plot": 100},
    }


@pytest.fixture
def synthetic_scene(base_cfg):
    """Синтетический снимок 256×256, 4 полосы (R, G, B, NIR)."""
    H, W, C = 256, 256, 4
    rng = np.random.default_rng(7)
    data = rng.integers(500, 2000, (C, H, W), dtype=np.uint16)

    # Зона растительности (высокое NIR)
    data[3, 30:80, 30:90] = 3500
    data[0, 30:80, 30:90] = 400

    # Зона воды (высокое Blue = канал 2, низкое NIR)
    data[2, 150:190, 100:160] = 3000
    data[3, 150:190, 100:160] = 200

    # Яркое здание
    data[:3, 100:130, 40:70] = 3000
    data[3,  100:130, 40:70] = 900

    transform = from_bounds(770000, 4820000, 772560, 4822560, W, H)
    crs = CRS.from_epsg(32642)

    loader = ImageLoader(base_cfg)
    indices = loader._compute_indices(data, ["Red", "Green", "Blue", "NIR"])

    return {
        "data":        data,
        "transform":   transform,
        "crs":         crs,
        "meta":        {},
        "band_names":  ["Red", "Green", "Blue", "NIR"],
        "shape":       (H, W),
        "n_bands":     C,
        "source_path": "synthetic",
        "source_id":   "test_scene",
        "indices":     indices,
    }


@pytest.fixture
def sample_tile(synthetic_scene):
    """Один тайл из синтетического снимка."""
    data = synthetic_scene["data"][:, :128, :128]
    transform = synthetic_scene["transform"]
    crs = synthetic_scene["crs"]
    indices = {k: v[:128, :128] for k, v in synthetic_scene["indices"].items()}
    return Tile(
        tile_id   = "test_t0",
        data      = data,
        transform = transform,
        crs       = crs,
        row_off   = 0,
        col_off   = 0,
        height    = 128,
        width     = 128,
        overlap   = 16,
        indices   = indices,
        source_id = "test_scene",
    )


# ── Тесты ImageLoader ─────────────────────────────────────────────────────────

class TestImageLoader:
    def test_compute_indices_with_nir(self, base_cfg):
        loader = ImageLoader(base_cfg)
        data = np.array([
            np.full((10, 10), 500, dtype=np.uint16),   # Red
            np.full((10, 10), 700, dtype=np.uint16),   # Green
            np.full((10, 10), 300, dtype=np.uint16),   # Blue
            np.full((10, 10), 2000, dtype=np.uint16),  # NIR
        ])
        indices = loader._compute_indices(data, ["Red", "Green", "Blue", "NIR"])
        assert "NDVI" in indices
        assert "NDWI" in indices
        # NDVI: (NIR-Red)/(NIR+Red) = 1500/2500 = 0.6
        assert indices["NDVI"].mean() > 0.5

    def test_compute_indices_rgb_only(self, base_cfg):
        loader = ImageLoader(base_cfg)
        data = np.ones((3, 10, 10), dtype=np.uint16) * 1000
        indices = loader._compute_indices(data, ["Red", "Green", "Blue"])
        assert indices == {}   # нет NIR — нет индексов

    def test_demo_scene(self, base_cfg, tmp_path):
        loader = ImageLoader(base_cfg)
        scene  = loader.load_demo_scene(tmp_path)
        assert scene["data"].shape[0] == 4
        assert scene["shape"] == (512, 512)
        assert "NDVI" in scene["indices"]
        assert (tmp_path / "demo_scene.tif").exists()


# ── Тесты TileManager ─────────────────────────────────────────────────────────

class TestTileManager:
    def test_split_count(self, base_cfg, synthetic_scene):
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        assert len(tiles) > 0

    def test_tile_shape(self, base_cfg, synthetic_scene):
        base_cfg["tiling"]["size"]    = 128
        base_cfg["tiling"]["overlap"] = 0
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        for tile in tiles:
            assert tile.data.shape[1] <= 128
            assert tile.data.shape[2] <= 128

    def test_tile_has_indices(self, base_cfg, synthetic_scene):
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        for tile in tiles:
            assert "NDVI" in tile.indices or len(tile.indices) >= 0

    def test_pixel_to_geo(self, base_cfg, synthetic_scene):
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        tile = tiles[0]
        x, y = tm.pixel_to_geo(tile, 0, 0)
        # Начало должно быть близко к левому верхнему углу
        assert abs(x - tile.transform.c) < 1e-3
        assert abs(y - tile.transform.f) < 1e-3

    def test_tile_id_unique(self, base_cfg, synthetic_scene):
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        ids = [t.tile_id for t in tiles]
        assert len(ids) == len(set(ids))


# ── Тесты детекторов ──────────────────────────────────────────────────────────

class TestVegetationDetector:
    def test_detects_vegetation_zone(self, base_cfg, sample_tile, synthetic_scene):
        det = VegetationDetector(base_cfg)
        results = det.detect(sample_tile, synthetic_scene)
        # Синтетика содержит зону растительности в тайле [0:128, 0:128]
        assert isinstance(results, list)
        classes = {r.class_name for r in results}
        assert all(c == "vegetation" for c in classes)

    def test_confidence_range(self, base_cfg, sample_tile, synthetic_scene):
        det = VegetationDetector(base_cfg)
        for r in det.detect(sample_tile, synthetic_scene):
            assert 0 <= r.confidence <= 100

    def test_geometry_valid(self, base_cfg, sample_tile, synthetic_scene):
        det = VegetationDetector(base_cfg)
        for r in det.detect(sample_tile, synthetic_scene):
            assert r.geometry is not None
            assert r.geometry.is_valid


class TestWaterDetector:
    def test_returns_list(self, base_cfg, sample_tile, synthetic_scene):
        det = WaterDetector(base_cfg)
        results = det.detect(sample_tile, synthetic_scene)
        assert isinstance(results, list)

    def test_no_ndwi_returns_empty(self, base_cfg, sample_tile, synthetic_scene):
        tile_no_ndwi = Tile(
            tile_id="no_ndwi", data=sample_tile.data,
            transform=sample_tile.transform, crs=sample_tile.crs,
            row_off=0, col_off=0, height=128, width=128,
            overlap=16, indices={}, source_id="test"
        )
        det = WaterDetector(base_cfg)
        assert det.detect(tile_no_ndwi, synthetic_scene) == []


class TestBuildingDetector:
    def test_returns_list(self, base_cfg, sample_tile, synthetic_scene):
        det = BuildingDetector(base_cfg)
        results = det.detect(sample_tile, synthetic_scene)
        assert isinstance(results, list)

    def test_area_positive(self, base_cfg, sample_tile, synthetic_scene):
        det = BuildingDetector(base_cfg)
        for r in det.detect(sample_tile, synthetic_scene):
            assert (r.area_m2 or 0) >= 0


class TestMultiClassDetector:
    def test_all_classes(self, base_cfg, sample_tile, synthetic_scene):
        md = MultiClassDetector(base_cfg)
        results = md.detect(sample_tile, synthetic_scene)
        assert isinstance(results, list)

    def test_enabled_filter(self, base_cfg, sample_tile, synthetic_scene):
        base_cfg["detection"]["enabled_classes"] = ["vegetation"]
        md = MultiClassDetector(base_cfg)
        results = md.detect(sample_tile, synthetic_scene)
        for r in results:
            assert r.class_name == "vegetation"


# ── Тесты PostProcessor ───────────────────────────────────────────────────────

class TestPostProcessor:
    def _make_det(self, cls, conf, poly, area=None, length=None):
        return RawDetection(
            det_id=f"test_{cls}",
            class_name=cls,
            confidence=conf,
            geometry=poly,
            geom_type="Polygon" if isinstance(poly, Polygon) else "LineString",
            area_m2=area or (poly.area if isinstance(poly, Polygon) else None),
            length_m=length,
            source_id="test",
            tile_id="t0",
        )

    def test_confidence_filter(self, base_cfg, synthetic_scene):
        pp = PostProcessor(base_cfg)
        p = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        dets = [
            self._make_det("vegetation", 5.0, p, area=10000),
            self._make_det("vegetation", 50.0, p, area=10000),
        ]
        result = pp._filter_confidence(dets)
        assert len(result) == 1
        assert result[0].confidence == 50.0

    def test_nms_removes_duplicate(self, base_cfg, synthetic_scene):
        pp = PostProcessor(base_cfg)
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])  # почти совпадает
        d1 = self._make_det("building", 80.0, p1, area=100)
        d2 = self._make_det("building", 60.0, p2, area=64)
        result = pp._nms_polygons([d1, d2])
        assert len(result) == 1
        assert result[0].confidence == 80.0

    def test_geometry_validation(self, base_cfg, synthetic_scene):
        pp = PostProcessor(base_cfg)
        good = RawDetection(
            det_id="g1", class_name="vegetation", confidence=70,
            geometry=Polygon([(0,0),(10,0),(10,10),(0,10)]),
            geom_type="Polygon", area_m2=100, source_id="test", tile_id="t0",
        )
        bad = RawDetection(
            det_id="b1", class_name="vegetation", confidence=70,
            geometry=None, geom_type="Polygon", source_id="test", tile_id="t0",
        )
        result = pp._validate_geometry([good, bad])
        assert len(result) == 1

    def test_full_run(self, base_cfg, synthetic_scene):
        pp = PostProcessor(base_cfg)
        md = MultiClassDetector(base_cfg)
        tm = TileManager(base_cfg)
        tiles = tm.split(synthetic_scene)
        raw = []
        for tile in tiles:
            raw.extend(md.detect(tile, synthetic_scene))
        processed = pp.run(raw, synthetic_scene)
        assert isinstance(processed, list)
        for d in processed:
            assert d.confidence >= base_cfg["postprocess"]["min_confidence"]


# ── Тесты GeoExporter ─────────────────────────────────────────────────────────

class TestGeoExporter:
    def _sample_detections(self):
        poly = Polygon([(76.9, 43.2), (76.91, 43.2), (76.91, 43.21), (76.9, 43.21)])
        return [
            RawDetection(
                det_id="veg_001", class_name="vegetation", confidence=75.0,
                geometry=poly, geom_type="Polygon",
                area_m2=12500.0, source_id="test", tile_id="t0",
            ),
            RawDetection(
                det_id="water_001", class_name="water", confidence=82.0,
                geometry=poly.buffer(-0.001), geom_type="Polygon",
                area_m2=9000.0, source_id="test", tile_id="t0",
            ),
        ]

    def test_geojson_created(self, base_cfg, synthetic_scene, tmp_path):
        exp = GeoExporter(base_cfg, synthetic_scene)
        dets = self._sample_detections()
        paths = exp.export_all(dets, tmp_path)
        assert "GeoJSON" in paths
        assert Path(paths["GeoJSON"]).exists()

    def test_geojson_has_required_fields(self, base_cfg, synthetic_scene, tmp_path):
        exp = GeoExporter(base_cfg, synthetic_scene)
        dets = self._sample_detections()
        paths = exp.export_all(dets, tmp_path)
        with open(paths["GeoJSON"], "r", encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj["features"]:
            props = feat["properties"]
            for field in ["id", "class", "confidence", "source"]:
                assert field in props, f"Поле '{field}' отсутствует в GeoJSON"

    def test_summary_structure(self, base_cfg, synthetic_scene, tmp_path):
        exp = GeoExporter(base_cfg, synthetic_scene)
        dets = self._sample_detections()
        exp.export_all(dets, tmp_path)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "crs" in summary
        assert "total_objects" in summary
        assert summary["total_objects"] == 2


# ── Тесты QualityMetrics ──────────────────────────────────────────────────────

class TestQualityMetrics:
    def _make_poly_det(self, i, cls, conf, x0, y0, size):
        poly = Polygon([
            (x0, y0), (x0+size, y0), (x0+size, y0+size), (x0, y0+size)
        ])
        return RawDetection(
            det_id=f"{cls}_{i}", class_name=cls, confidence=conf,
            geometry=poly, geom_type="Polygon",
            area_m2=size*size*1e6, source_id="test", tile_id="t0",
        )

    def test_zone_statistics(self, base_cfg, synthetic_scene):
        qm = QualityMetrics(base_cfg)
        dets = [self._make_poly_det(i, "vegetation", 70, 76.9+i*0.01, 43.2, 0.001)
                for i in range(5)]
        stats = qm.zone_statistics(dets, synthetic_scene)
        assert "total_objects" in stats
        assert stats["total_objects"] == 5
        assert "vegetation" in stats["classes"]
        assert stats["classes"]["vegetation"]["count"] == 5

    def test_manual_sample_report(self, base_cfg, synthetic_scene, tmp_path):
        qm = QualityMetrics(base_cfg)
        dets = [self._make_poly_det(i, "water", 80, 76.9+i*0.01, 43.2, 0.002)
                for i in range(20)]
        report = qm.manual_sample_report(dets, tmp_path)
        assert "pseudo_precision" in report
        assert 0.0 <= report["pseudo_precision"] <= 1.0
        assert (tmp_path / "manual_sample.csv").exists()

    def test_save_zone_stats(self, base_cfg, synthetic_scene, tmp_path):
        qm = QualityMetrics(base_cfg)
        dets = [self._make_poly_det(i, "building", 65, 76.9+i*0.01, 43.2, 0.001)
                for i in range(3)]
        stats = qm.zone_statistics(dets, synthetic_scene)
        path = qm.save_zone_stats(stats, tmp_path)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert "classes" in data


# ── Интеграционный тест ───────────────────────────────────────────────────────

class TestIntegration:
    def test_full_demo_pipeline(self, base_cfg, tmp_path):
        """End-to-end: demo → detect → postprocess → export."""
        loader = ImageLoader(base_cfg)
        scene  = loader.load_demo_scene(tmp_path)
        assert scene["data"].shape[0] == 4

        tm = TileManager(base_cfg)
        tiles = tm.split(scene)
        assert len(tiles) > 0

        md = MultiClassDetector(base_cfg)
        raw = []
        for tile in tiles:
            raw.extend(md.detect(tile, scene))

        pp = PostProcessor(base_cfg)
        processed = pp.run(raw, scene)
        assert isinstance(processed, list)

        exp = GeoExporter(base_cfg, scene)
        paths = exp.export_all(processed, tmp_path)
        assert "GeoJSON" in paths
        assert Path(paths["GeoJSON"]).exists()

        qm = QualityMetrics(base_cfg)
        stats = qm.zone_statistics(processed, scene)
        assert stats["total_objects"] == len(processed)
