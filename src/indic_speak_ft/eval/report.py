"""Render panel results to a self-contained, theme-aware HTML report — the artifact the
reviewer reads (GOAL Phase 6). Per-bucket tables with WER + 95% bootstrap CIs, retention and
longform sections, and (when present) speaker similarity vs the ceiling/floor.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--head:#f5f5f5;--accent:#0b6}
@media(prefers-color-scheme:dark){:root{--bg:#151515;--fg:#eaeaea;--muted:#9a9a9a;--line:#333;--head:#1f1f1f;--accent:#3db88a}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
table{border-collapse:collapse;width:100%;margin:8px 0 24px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:var(--head)}
td.good{color:var(--accent)}caption{text-align:left;font-weight:600;margin:16px 0 6px;font-size:15px}
.ci{color:var(--muted);font-size:12px}
"""


def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, float):
        return "—" if math.isnan(x) else f"{x:.{nd}f}"  # NaN → em-dash
    return html.escape(str(x))


def _spk_sim_cell(b: dict) -> str:
    """Aggregate speaker similarity, plus a per-voice line when a bucket carries both target
    voices (D15 tightening #2 — Anagha and Chinmay are shown separately, not averaged)."""
    if "speaker_similarity" not in b:
        return "—"
    agg = _fmt(b["speaker_similarity"])
    by_voice = b.get("speaker_similarity_by_voice")
    if not by_voice:
        return agg
    detail = " · ".join(f"{html.escape(v)} {_fmt(s)}" for v, s in sorted(by_voice.items()))
    return f"{agg}<br><span class='ci'>{detail}</span>"


def _wer_table(buckets: dict[str, dict]) -> str:
    rows = []
    for name, b in buckets.items():
        if "wer" not in b:
            continue
        ci = b.get("wer_ci95", [float("nan"), float("nan")])
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{b.get('n','')}</td>"
            f"<td>{_fmt(b['wer'])} <span class='ci'>[{_fmt(ci[0])}, {_fmt(ci[1])}]</span></td>"
            f"<td>{b.get('insertions','')}</td><td>{b.get('deletions','')}</td>"
            f"<td>{b.get('substitutions','')}</td>"
            f"<td>{_fmt(b.get('hit_max_new_tokens_rate', 0.0),2)}</td>"
            f"<td>{_spk_sim_cell(b)}</td></tr>")
    if not rows:
        return ""
    return ("<caption>WER buckets (95% bootstrap CI over utterances)</caption>"
            "<table><tr><th>bucket</th><th>n</th><th>WER</th><th>ins</th><th>del</th>"
            "<th>sub</th><th>max-hit</th><th>spk-sim</th></tr>" + "".join(rows) + "</table>")


def _longform_table(buckets: dict[str, dict]) -> str:
    b = buckets.get("marathi_longform")
    if not b or "repetition_rate_3gram" not in b:
        return ""
    return ("<caption>Longform (30–60 s)</caption><table>"
            "<tr><th>n</th><th>rep 3-gram</th><th>rep 5-gram</th><th>max-hit rate</th><th>mean frames</th></tr>"
            f"<tr><td>{b.get('n','')}</td><td>{_fmt(b['repetition_rate_3gram'])}</td>"
            f"<td>{_fmt(b['repetition_rate_5gram'])}</td><td>{_fmt(b.get('hit_max_new_tokens_rate',0.0),2)}</td>"
            f"<td>{_fmt(b.get('mean_frames',0.0),1)}</td></tr></table>")


def write_html(results: dict, out_path: str, *, title: str = "Indic-Speak Marathi FT — eval panel",
               subtitle: str = "") -> str:
    buckets = results.get("buckets", {})
    body = _wer_table(buckets) + _longform_table(buckets)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
           f"<h1>{html.escape(title)}</h1><p class='sub'>{html.escape(subtitle)}</p>"
           f"{body}<details><summary>raw JSON</summary><pre>{html.escape(json.dumps(results, ensure_ascii=False, indent=2))}</pre></details>"
           f"</body></html>")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc, encoding="utf-8")
    return out_path
