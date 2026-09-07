"""Render a persisted analysis into a self-contained, downloadable HTML report.

Design constraints, each for a reason:

- **Deterministic and pure.** ``render_analysis_report`` is a function of the stored record
  (plus the input assets), so the same analysis always produces the same report and the output
  is straightforward to test.
- **Real data only.** Every field comes from the persisted record. Where something was not
  recorded, the report says "not available" rather than inventing it.
- **Self-contained.** Evidence images are embedded as base64 data URIs, so a downloaded report
  opens correctly offline with no calls back to the server.
- **XSS-safe.** Every dynamic string (query, filenames, answers, trace details) is HTML-escaped;
  the only markup is the fixed template. A malicious filename or query cannot inject script.
- **No chain-of-thought.** Only the observable trace and measured evidence are shown.

The output is HTML rather than PDF deliberately: it needs no heavy native dependencies (so it
builds in the slim Docker image), and any browser can "Print → Save as PDF" for a PDF copy. The
report is styled for print.
"""

from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models_db.analysis import Analysis
from ..models_db.raster_asset import RasterAsset
from ..security.files import UnsafePathError
from ..services import storage

#: A standing caveat printed on every report. The system is honest about its evaluation state.
_BENCHMARK_CAVEAT = (
    "Model and tool outputs are real but not yet benchmark-validated or calibrated; confidence "
    "is derived from measurable signals, not from a calibrated probability of correctness. "
    "Treat results as indicative."
)

#: Matches the two evidence reference shapes the backend emits.
_EVIDENCE_TOKEN = re.compile(r"^/api/v1/evidence/([0-9a-f]{8,32})$")
_PREVIEW_REF = re.compile(r"^/api/v1/uploads/([^/]+)/preview$")

#: Cap on an embedded image so a report stays a reasonable size; previews/overlays are small.
_MAX_EMBED_BYTES = 4 * 1024 * 1024


def _esc(value: Any) -> str:
    """HTML-escape any value to a string. The single defence against injected markup."""
    if value is None:
        return "not available"
    return html.escape(str(value), quote=True)


def _fmt_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    return _esc(value)


def _resolve_evidence_file(
    reference: str | None, asset_by_id: dict[str, RasterAsset]
) -> Path | None:
    """Map an evidence ``reference`` to an on-disk PNG, confined to the storage root.

    Returns ``None`` when the reference is not an embeddable image or the file is missing.
    """
    if not reference:
        return None

    token_match = _EVIDENCE_TOKEN.match(reference)
    preview_match = _PREVIEW_REF.match(reference)
    relative: str | None = None
    if token_match:
        relative = f"{storage.EVIDENCE_DIR}/overlays/{token_match.group(1)}.png"
    elif preview_match:
        asset = asset_by_id.get(preview_match.group(1))
        relative = asset.preview_path if asset and asset.preview_path else None
    if not relative:
        return None

    try:
        path = storage.absolute_path(relative)
    except (UnsafePathError, ValueError):
        return None
    return path if path.is_file() else None


def _data_uri(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_EMBED_BYTES:
            return None
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{encoded}"


def _detail_rows(detail: dict[str, Any]) -> str:
    """Render a small, safe key/value table for an evidence item's measured detail.

    Nested structures are compacted to a single escaped line so a report never renders raw
    unescaped JSON, and long blobs do not dominate the page.
    """
    rows = []
    for key, value in detail.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (dict, list)):
            rendered = _esc(_compact(value))
        elif isinstance(value, float):
            rendered = _esc(f"{value:.4g}")
        else:
            rendered = _esc(value)
        rows.append(f'<tr><td class="k">{_esc(key.replace("_", " "))}</td><td>{rendered}</td></tr>')
    return "".join(rows)


def _compact(value: Any, limit: int = 240) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _asset_block(asset: RasterAsset) -> str:
    geo = "georeferenced" if asset.is_georeferenced else "not georeferenced"
    crs = f"EPSG:{asset.crs_epsg}" if asset.crs_epsg else (asset.crs or "not available")
    rows = [
        ("File", asset.original_filename),
        ("Asset ID", asset.id),
        ("Role", asset.role),
        ("Modality", asset.modality),
        ("Dimensions", f"{asset.width} x {asset.height} px" if asset.width else "not available"),
        ("Bands", asset.band_count),
        ("Data type", asset.dtype),
        ("Georeferencing", geo),
        ("CRS", crs),
    ]
    cells = "".join(f'<tr><td class="k">{_esc(k)}</td><td>{_esc(v)}</td></tr>' for k, v in rows)
    return f'<table class="kv">{cells}</table>'


def _provenance_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    """Pull the model-provenance lines out of the observable trace."""
    return [
        str(entry.get("detail", ""))
        for entry in trace
        if entry.get("step") in {"Model provenance", "Model selection"}
    ]


def render_analysis_report(
    record: Analysis,
    assets: list[RasterAsset] | None = None,
    *,
    embed_images: bool = True,
) -> str:
    """Render ``record`` into a complete, self-contained HTML report string.

    Args:
        record: The persisted analysis.
        assets: The input rasters, for the input-metadata section and preview embedding. May be
            empty (e.g. the uploads were later deleted); the report then omits that section.
        embed_images: When True, evidence images are embedded as data URIs. Tests may disable
            this to keep output compact.
    """
    assets = assets or []
    asset_by_id = {asset.id: asset for asset in assets}

    input_metadata = record.input_metadata or {}
    evidence = (record.evidence_refs or {}).get("items", [])
    trace = (record.trace or {}).get("entries", [])
    issues = (record.error or {}).get("issues", [])
    answer = input_metadata.get("answer")
    signals = input_metadata.get("confidence_signals", [])
    warnings = input_metadata.get("warnings", [])

    parts: list[str] = []

    # --- header -----------------------------------------------------------------
    generated = _esc(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    status = _esc(record.status)
    status_html = f'<span class="status status-{status}">{status}</span>'
    parts.append(
        f"""
        <header class="report-header">
          <div class="brand">SatQuery AI</div>
          <div class="subtitle">Remote-Sensing Analysis Report</div>
          <table class="kv">
            <tr><td class="k">Analysis ID</td><td>{_esc(record.id)}</td></tr>
            <tr><td class="k">Generated</td><td>{generated}</td></tr>
            <tr><td class="k">Analyzed at</td><td>{_fmt_timestamp(record.created_at)}</td></tr>
            <tr><td class="k">Status</td><td>{status_html}</td></tr>
            <tr><td class="k">Task</td><td>{_esc(record.task)}</td></tr>
          </table>
        </header>
        """
    )

    # --- query ------------------------------------------------------------------
    parts.append(
        f"<section><h2>Query</h2><blockquote>{_esc(record.query or 'not available')}</blockquote></section>"  # noqa: E501
    )

    # --- inputs -----------------------------------------------------------------
    if assets:
        blocks = "".join(f'<div class="asset">{_asset_block(a)}</div>' for a in assets)
        parts.append(f"<section><h2>Input imagery</h2>{blocks}</section>")
    elif input_metadata.get("asset_ids"):
        ids = ", ".join(_esc(i) for i in input_metadata["asset_ids"])
        parts.append(
            f"<section><h2>Input imagery</h2><p>Asset IDs: {ids} "
            "(metadata unavailable; the uploads may have been removed).</p></section>"
        )

    # --- selected tool ----------------------------------------------------------
    tool = _esc(record.model_id) if record.model_id else "not available"
    version = _esc(record.model_version) if record.model_version else "not available"
    parts.append(
        f'<section><h2>Agent selection</h2><table class="kv">'
        f'<tr><td class="k">Task</td><td>{_esc(record.task)}</td></tr>'
        f'<tr><td class="k">Specialist / tool</td><td>{tool}</td></tr>'
        f'<tr><td class="k">Version</td><td>{version}</td></tr>'
        f"</table></section>"
    )

    # --- answer -----------------------------------------------------------------
    if answer:
        parts.append(f'<section><h2>Answer</h2><p class="answer">{_esc(answer)}</p></section>')
    else:
        parts.append(
            '<section><h2>Answer</h2><p class="muted">No answer was produced for this request. '
            "See the status and issues below.</p></section>"
        )

    # --- issues -----------------------------------------------------------------
    if issues:
        items = "".join(
            f"<li><strong>{_esc(i.get('code'))}</strong>: {_esc(i.get('message'))}</li>"
            for i in issues
        )
        parts.append(f'<section><h2>Issues</h2><ul class="issues">{items}</ul></section>')

    # --- confidence -------------------------------------------------------------
    score = record.confidence
    score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "not available"
    signal_items = "".join(f"<li>{_esc(s)}</li>" for s in signals) or "<li>none recorded</li>"
    parts.append(
        f'<section><h2>Confidence</h2><table class="kv">'
        f'<tr><td class="k">Level</td><td>{_esc(record.confidence_level)}</td></tr>'
        f'<tr><td class="k">Score</td><td>{_esc(score_text)}</td></tr></table>'
        f'<p class="muted">Signals that produced this assessment:</p>'
        f'<ul class="signals">{signal_items}</ul></section>'
    )

    # --- warnings ---------------------------------------------------------------
    if warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
        parts.append(
            f'<section><h2>Warnings &amp; limitations</h2><ul class="warnings">{items}</ul></section>'  # noqa: E501
        )

    # --- evidence ---------------------------------------------------------------
    if evidence:
        cards = []
        for item in evidence:
            reference = item.get("reference")
            image_html = ""
            if embed_images and reference:
                path = _resolve_evidence_file(reference, asset_by_id)
                uri = _data_uri(path) if path else None
                if uri:
                    image_html = (
                        f'<img class="evidence-img" alt="{_esc(item.get("label"))}" src="{uri}" />'
                    )
                elif _EVIDENCE_TOKEN.match(reference) or _PREVIEW_REF.match(reference):
                    image_html = '<p class="muted">[image evidence could not be embedded]</p>'
            detail_rows = _detail_rows(item.get("detail") or {})
            detail_html = f'<table class="kv">{detail_rows}</table>' if detail_rows else ""
            cards.append(
                f'<div class="evidence-card"><h3>{_esc(item.get("kind"))} '
                f'<span class="muted">· {_esc(item.get("label"))}</span></h3>'
                f"{image_html}{detail_html}</div>"
            )
        parts.append(f"<section><h2>Evidence ({len(evidence)})</h2>{''.join(cards)}</section>")
    else:
        parts.append(
            '<section><h2>Evidence</h2><p class="muted">No inspectable evidence was produced.</p></section>'  # noqa: E501
        )

    # --- provenance -------------------------------------------------------------
    provenance = _provenance_from_trace(trace)
    if provenance:
        items = "".join(f"<li>{_esc(p)}</li>" for p in provenance)
        parts.append(f'<section><h2>Provenance</h2><ul class="provenance">{items}</ul></section>')

    # --- trace ------------------------------------------------------------------
    if trace:
        rows = "".join(
            f"<tr><td>{_esc(e.get('step'))}</td>"
            f'<td class="status-cell status-{_esc(e.get("status"))}">{_esc(e.get("status"))}</td>'
            f"<td>{_esc(e.get('detail'))}</td></tr>"
            for e in trace
        )
        parts.append(
            "<section><h2>Execution trace</h2>"
            '<p class="muted">An observable, ordered summary of what the system did. '
            "It contains no private reasoning.</p>"
            '<table class="trace"><thead><tr>'
            "<th>Step</th><th>Status</th><th>Detail</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>"
        )

    # --- footer -----------------------------------------------------------------
    parts.append(
        f'<footer><p class="caveat">{_esc(_BENCHMARK_CAVEAT)}</p>'
        '<p class="muted">Generated by SatQuery AI (SIH26167). No hidden evaluation data was '
        "used.</p></footer>"
    )

    body = "\n".join(parts)
    return _wrap_document(record, body)


def _wrap_document(record: Analysis, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SatQuery AI report - {_esc(record.id)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         color: #1a1f24; max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.5; }}
  .report-header {{ border-bottom: 3px solid #2563eb; padding-bottom: 12px; margin-bottom: 20px; }}
  .brand {{ font-size: 26px; font-weight: 700; color: #2563eb; }}
  .subtitle {{ font-size: 14px; color: #556; margin-bottom: 12px; }}
  h2 {{ font-size: 17px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-top: 26px; }}
  h3 {{ font-size: 14px; margin: 12px 0 6px; }}
  table.kv, table.trace {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.kv td {{ padding: 3px 8px; vertical-align: top; border-bottom: 1px solid #f1f5f9; }}
  td.k {{ color: #64748b; width: 190px; }}
  blockquote {{ margin: 0; padding: 10px 14px; background: #f8fafc;
    border-left: 3px solid #2563eb; }}
  .answer {{ font-size: 15px; }}
  .muted {{ color: #64748b; font-size: 13px; }}
  .status {{ padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
  .status-completed {{ background: #dcfce7; color: #166534; }}
  .status-rejected, .status-failed {{ background: #fee2e2; color: #991b1b; }}
  .status-unavailable {{ background: #fef9c3; color: #854d0e; }}
  .status-cell {{ font-weight: 600; }}
  .evidence-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin: 10px 0; }}
  .evidence-img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; margin: 6px 0; }}
  table.trace td, table.trace th {{ border: 1px solid #e2e8f0; padding: 5px 8px; font-size: 12px;
                                    text-align: left; vertical-align: top; }}
  ul {{ margin: 6px 0; padding-left: 20px; font-size: 13px; }}
  footer {{ margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
  .caveat {{ font-size: 12px; color: #854d0e; background: #fef9c3;
    padding: 8px 12px; border-radius: 6px; }}
  @media print {{ body {{ padding: 0; }} .evidence-card {{ break-inside: avoid; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_report_filename(record: Analysis) -> str:
    """A safe download filename derived only from the analysis id (a UUID)."""
    safe_id = re.sub(r"[^0-9a-zA-Z_-]", "", str(record.id))[:64] or "analysis"
    return f"satquery-report-{safe_id}.html"
