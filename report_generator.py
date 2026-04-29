"""
pipeline/report_generator.py
Генерирует самодостаточный HTML-отчёт с:
  - метриками качества
  - зональной статистикой (таблица)
  - встроенными PNG-картами (base64)
  - сводкой по классам
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _img_to_b64(path: Path) -> str:
    """Читает PNG и возвращает data-URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def generate_report(
    output_dir: Path,
    zone_stats: dict,
    quality_report: dict,
    summary: Optional[dict] = None,
) -> str:
    """Создаёт HTML-отчёт и возвращает путь."""
    od = Path(output_dir)

    # Загружаем summary.json если не передан
    if summary is None:
        sp = od / "summary.json"
        summary = json.loads(sp.read_text("utf-8")) if sp.exists() else {}

    # Изображения
    imgs = {
        "overview":    _img_to_b64(od / "overview.png"),
        "density":     _img_to_b64(od / "density_map.png"),
        "vegetation":  _img_to_b64(od / "class_vegetation.png"),
        "water":       _img_to_b64(od / "class_water.png"),
        "building":    _img_to_b64(od / "class_building.png"),
        "road":        _img_to_b64(od / "class_road.png"),
    }

    # CSS-цвета классов
    class_colors = {
        "vegetation": "#2ecc71",
        "water":      "#3498db",
        "building":   "#e74c3c",
        "road":       "#f39c12",
    }

    # ── Генерация таблицы зон ─────────────────────────────────────────────────
    zone_rows = ""
    for cls, entry in zone_stats.get("classes", {}).items():
        color = class_colors.get(cls, "#95a5a6")
        area  = entry.get("total_area_ha", entry.get("total_length_km", "—"))
        unit  = "га" if "total_area_ha" in entry else "км"
        zone_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{cls}</span></td>
          <td>{entry.get('count', 0)}</td>
          <td>{entry.get('density_per_ha', '—')}</td>
          <td>{area} {unit}</td>
          <td>{entry.get('mean_confidence', '—')}</td>
        </tr>"""

    # ── Таблица метрик качества ───────────────────────────────────────────────
    metrics_rows = ""
    if "per_class" in quality_report:
        for cls, m in quality_report["per_class"].items():
            color = class_colors.get(cls, "#95a5a6")
            metrics_rows += f"""
            <tr>
              <td><span class="badge" style="background:{color}">{cls}</span></td>
              <td>{m.get('precision','—')}</td>
              <td>{m.get('recall','—')}</td>
              <td>{m.get('f1','—')}</td>
              <td>{m.get('ap','—')}</td>
              <td>{m.get('n_det','—')}</td>
              <td>{m.get('n_gt','—')}</td>
            </tr>"""
    elif "pseudo_precision" in quality_report:
        metrics_rows = f"""
        <tr>
          <td colspan="7">
            <b>Псевдо-Precision (контрольная выборка, {quality_report.get('sample_size',0)} объектов):</b>
            {quality_report.get('pseudo_precision', '—')}
          </td>
        </tr>"""

    # ── Типичные ошибки ───────────────────────────────────────────────────────
    errors_html = ""
    errors = (quality_report.get("common_errors") or
              quality_report.get("sample_records") and
              quality_report.get("common_errors", []))
    for err in (quality_report.get("common_errors") or []):
        errors_html += f"""
        <div class="error-card">
          <h4>⚠ {err.get('type','')}</h4>
          <p><b>Причина:</b> {err.get('reason','')}</p>
          <p><b>Решение:</b> {err.get('fix','')}</p>
        </div>"""

    # ── Миниатюры карт ────────────────────────────────────────────────────────
    def img_block(label, key):
        src = imgs.get(key, "")
        if not src:
            return ""
        return f"""
        <div class="img-card">
          <p class="img-label">{label}</p>
          <img src="{src}" alt="{label}" />
        </div>"""

    images_html = "".join([
        img_block("Обзор: все классы", "overview"),
        img_block("Карта плотности", "density"),
        img_block("Растительность", "vegetation"),
        img_block("Вода", "water"),
        img_block("Здания", "building"),
        img_block("Дороги", "road"),
    ])

    # ── AOI ───────────────────────────────────────────────────────────────────
    aoi_area = zone_stats.get("aoi_area_ha", "—")
    total_obj = zone_stats.get("total_objects", "—")
    mAP_val   = quality_report.get("mAP", quality_report.get("pseudo_precision", "—"))
    source_id = summary.get("source", zone_stats.get("source", "—"))
    crs_val   = summary.get("crs", zone_stats.get("crs", "—"))

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GIS Pipeline — Отчёт</title>
<style>
  :root {{
    --bg: #f4f6f9; --card: #fff; --accent: #2c3e50;
    --text: #333; --muted: #888; --border: #e0e0e0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg);
          color: var(--text); font-size: 14px; }}
  header {{ background: var(--accent); color: #fff; padding: 24px 40px; }}
  header h1 {{ font-size: 22px; font-weight: 600; }}
  header p  {{ opacity: 0.75; font-size: 13px; margin-top: 4px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px; }}
  .section {{ background: var(--card); border-radius: 8px;
              box-shadow: 0 1px 4px rgba(0,0,0,.07);
              padding: 24px; margin-bottom: 24px; }}
  .section h2 {{ font-size: 16px; color: var(--accent);
                 border-bottom: 2px solid var(--border);
                 padding-bottom: 10px; margin-bottom: 16px; }}
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .kpi {{ background: var(--card); border-radius: 8px; padding: 18px 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,.07); flex: 1; min-width: 140px; }}
  .kpi .val {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .kpi .lbl {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--accent); color: #fff; padding: 9px 12px;
        text-align: left; font-weight: 500; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #f9f9f9; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
            color: #fff; font-size: 12px; font-weight: 600; }}
  .img-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .img-card {{ flex: 1; min-width: 260px; text-align: center; }}
  .img-card img {{ width: 100%; border-radius: 6px;
                   border: 1px solid var(--border); }}
  .img-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
  .error-card {{ background: #fff9f0; border-left: 4px solid #f39c12;
                 padding: 12px 16px; margin-bottom: 12px; border-radius: 4px; }}
  .error-card h4 {{ font-size: 13px; margin-bottom: 6px; }}
  .error-card p {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .meta-item {{ background: var(--bg); padding: 10px 14px; border-radius: 6px; }}
  .meta-item .key {{ font-size: 11px; color: var(--muted); }}
  .meta-item .value {{ font-size: 13px; font-weight: 600; color: var(--accent); }}
  footer {{ text-align: center; color: var(--muted); font-size: 12px;
            padding: 24px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>🛰 GIS Pipeline — Отчёт об извлечении объектов</h1>
  <p>Учебный кейс NURIS | Автоматическое выявление объектов по космоснимкам</p>
</header>

<div class="container">

  <!-- KPI -->
  <div class="kpi-row">
    <div class="kpi">
      <div class="val">{total_obj}</div>
      <div class="lbl">Объектов выявлено</div>
    </div>
    <div class="kpi">
      <div class="val">{aoi_area} га</div>
      <div class="lbl">Площадь AOI</div>
    </div>
    <div class="kpi">
      <div class="val">{mAP_val}</div>
      <div class="lbl">mAP / Pseudo-Precision</div>
    </div>
    <div class="kpi">
      <div class="val">{crs_val}</div>
      <div class="lbl">Координатная система</div>
    </div>
  </div>

  <!-- Метаданные -->
  <div class="section">
    <h2>📋 Метаданные сцены</h2>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="key">Источник</div>
        <div class="value">{source_id}</div>
      </div>
      <div class="meta-item">
        <div class="key">Дата обработки</div>
        <div class="value">{summary.get('date', '—')}</div>
      </div>
      <div class="meta-item">
        <div class="key">CRS экспорта</div>
        <div class="value">{crs_val}</div>
      </div>
      <div class="meta-item">
        <div class="key">Площадь AOI</div>
        <div class="value">{aoi_area} га</div>
      </div>
    </div>
  </div>

  <!-- Зональная статистика -->
  <div class="section">
    <h2>📊 Зональная статистика</h2>
    <table>
      <thead>
        <tr>
          <th>Класс</th>
          <th>Количество</th>
          <th>Плотность (на га)</th>
          <th>Площадь / Длина</th>
          <th>Средняя уверенность</th>
        </tr>
      </thead>
      <tbody>{zone_rows}</tbody>
    </table>
  </div>

  <!-- Метрики качества -->
  <div class="section">
    <h2>🎯 Метрики качества</h2>
    {"<p style='color:#888;font-size:12px'>mAP = " + str(quality_report.get('mAP','—')) + "</p>" if 'mAP' in quality_report else ""}
    <table>
      <thead>
        <tr>
          <th>Класс</th><th>Precision</th><th>Recall</th>
          <th>F1</th><th>AP</th><th>Детекций</th><th>GT</th>
        </tr>
      </thead>
      <tbody>{metrics_rows}</tbody>
    </table>
  </div>

  <!-- Типичные ошибки -->
  {"<div class='section'><h2>⚠ Типичные ошибки</h2>" + errors_html + "</div>" if errors_html else ""}

  <!-- Карты -->
  <div class="section">
    <h2>🗺 Карты результатов</h2>
    <div class="img-grid">
      {images_html}
    </div>
  </div>

  <!-- Файлы -->
  <div class="section">
    <h2>📁 Файлы результатов</h2>
    <table>
      <thead>
        <tr><th>Формат</th><th>Файл</th><th>Описание</th></tr>
      </thead>
      <tbody>
        <tr><td>GeoJSON</td><td>detections.geojson</td><td>Все объекты (обязательный по ТЗ)</td></tr>
        <tr><td>GeoPackage</td><td>detections.gpkg</td><td>ГИС-формат для QGIS</td></tr>
        <tr><td>Shapefile</td><td>shapefile/</td><td>По типам геометрии</td></tr>
        <tr><td>JSON</td><td>zone_statistics.json</td><td>Зональные показатели</td></tr>
        <tr><td>CSV</td><td>zone_statistics.csv</td><td>Сводная таблица</td></tr>
        <tr><td>JSON</td><td>quality_report.json</td><td>Метрики качества</td></tr>
        <tr><td>PNG</td><td>overview.png</td><td>Обзорная карта</td></tr>
        <tr><td>PNG</td><td>density_map.png</td><td>Тепловая карта плотности</td></tr>
      </tbody>
    </table>
  </div>

</div>
<footer>GIS Pipeline — NURIS учебный кейс | CRS: {crs_val}</footer>
</body>
</html>"""

    path = od / "report.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"HTML-отчёт: {path}")
    return str(path)
