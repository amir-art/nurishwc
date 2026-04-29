"""
GIS Pipeline — Автоматическое извлечение объектов из космических снимков
Учебный кейс NURIS | Вся тех. часть по ТЗ
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from pipeline.loader import ImageLoader
from pipeline.tiler import TileManager
from pipeline.detector import MultiClassDetector
from pipeline.postprocess import PostProcessor
from pipeline.exporter import GeoExporter
from pipeline.metrics import QualityMetrics
from pipeline.visualizer import ResultVisualizer

# ─── Логгер ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger("main")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GIS Pipeline: объекты из космоснимков → GeoJSON"
    )
    parser.add_argument("--config", default="config.yaml", help="Путь к конфигу")
    parser.add_argument("--image", help="Путь к GeoTIFF (переопределяет конфиг)")
    parser.add_argument("--aoi", help="Путь к AOI GeoJSON (переопределяет конфиг)")
    parser.add_argument("--output", default="output", help="Папка для результатов")
    parser.add_argument(
        "--gt", default=None, help="Ground truth GeoJSON для расчёта метрик"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Запустить в демо-режиме (синтетика)"
    )
    return parser.parse_args()


def run_pipeline(cfg: dict, args) -> dict:
    t0 = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Загрузка и подготовка снимков ────────────────────────────────────
    log.info("=== Шаг 1: Загрузка снимка ===")
    loader = ImageLoader(cfg)
    if args.demo:
        scene = loader.load_demo_scene(output_dir)
    else:
        image_path = args.image or cfg["data"]["image"]
        aoi_path = args.aoi or cfg["data"].get("aoi")
        scene = loader.load(image_path, aoi_path)

    log.info(f"Снимок загружен: {scene['shape']} px, CRS={scene['crs']}, "
             f"bands={scene['band_names']}")

    # ── 2. Тайлинг ──────────────────────────────────────────────────────────
    log.info("=== Шаг 2: Разбивка на тайлы ===")
    tiler = TileManager(cfg)
    tiles = tiler.split(scene)
    log.info(f"Создано тайлов: {len(tiles)}")

    # ── 3. Детекция объектов по тайлам ──────────────────────────────────────
    log.info("=== Шаг 3: Детекция объектов ===")
    detector = MultiClassDetector(cfg)
    raw_detections = []
    for i, tile in enumerate(tiles):
        dets = detector.detect(tile, scene)
        raw_detections.extend(dets)
        if (i + 1) % 10 == 0 or (i + 1) == len(tiles):
            log.info(f"  Обработано тайлов: {i+1}/{len(tiles)}, "
                     f"детекций: {len(raw_detections)}")

    log.info(f"Всего сырых детекций: {len(raw_detections)}")

    # ── 4. Постобработка ─────────────────────────────────────────────────────
    log.info("=== Шаг 4: Постобработка ===")
    pp = PostProcessor(cfg)
    detections = pp.run(raw_detections, scene)
    log.info(f"После постобработки: {len(detections)} объектов")

    # ── 5. Экспорт в GIS-форматы ─────────────────────────────────────────────
    log.info("=== Шаг 5: Экспорт ===")
    exporter = GeoExporter(cfg, scene)
    export_paths = exporter.export_all(detections, output_dir)
    for fmt, path in export_paths.items():
        log.info(f"  {fmt}: {path}")

    # ── 6. Расчёт показателей по зонам ───────────────────────────────────────
    log.info("=== Шаг 6: Показатели по зонам ===")
    metrics_engine = QualityMetrics(cfg)
    zone_stats = metrics_engine.zone_statistics(detections, scene)
    stats_path = metrics_engine.save_zone_stats(zone_stats, output_dir)
    log.info(f"Сводная таблица: {stats_path}")

    # ── 7. Метрики качества (если есть GT) ───────────────────────────────────
    gt_path = args.gt or cfg["evaluation"].get("ground_truth")
    quality_report = {}
    if gt_path:
        log.info("=== Шаг 7: Оценка качества по GT ===")
        quality_report = metrics_engine.evaluate(detections, gt_path)
        metrics_engine.save_report(quality_report, output_dir)
    else:
        log.info("=== Шаг 7: Контрольная выборка (без GT) ===")
        quality_report = metrics_engine.manual_sample_report(detections, output_dir)

    # ── 8. Визуализация ───────────────────────────────────────────────────────
    log.info("=== Шаг 8: Визуализация ===")
    viz = ResultVisualizer(cfg)
    viz.render_overview(scene, detections, output_dir)
    viz.render_per_class(scene, detections, output_dir)
    viz.render_density_map(detections, scene, output_dir)

    elapsed = time.time() - t0
    log.info(f"=== Пайплайн завершён за {elapsed:.1f}с ===")
    log.info(f"Результаты сохранены в: {output_dir.resolve()}")

    return {
        "detections": len(detections),
        "tiles": len(tiles),
        "export_paths": export_paths,
        "zone_stats": zone_stats,
        "quality": quality_report,
        "elapsed_s": elapsed,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    results = run_pipeline(cfg, args)

    print("\n" + "=" * 60)
    print("ИТОГО:")
    print(f"  Объектов выделено : {results['detections']}")
    print(f"  Тайлов обработано : {results['tiles']}")
    print(f"  Время работы      : {results['elapsed_s']:.1f}с")
    if results["quality"]:
        q = results["quality"]
        print(f"  Precision         : {q.get('precision', 'N/A')}")
        print(f"  Recall            : {q.get('recall', 'N/A')}")
        print(f"  F1                : {q.get('f1', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
