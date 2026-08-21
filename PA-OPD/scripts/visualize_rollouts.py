#!/usr/bin/env python3
"""Build a self-contained, offline HTML dashboard for PA-OPD rollout JSONL files.

Example:
    python scripts/visualize_rollouts.py \
      /path/to/outputs/<run>/rollouts

The output contains all selected rows and does not need a web server or any
third-party Python package.  It is intended for inspecting format validity,
answer accuracy, rollout multiplicity, prompts, and raw model responses.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutFile:
    step: int
    path: Path


@dataclass(frozen=True)
class ImageRecord:
    """A Vision-OPD record that can be linked to a rollout prompt."""

    line_number: int
    answer: str
    original_path: Path
    crop_path: Path


_FORMAT_SUFFIX = (
    "\n\nContinue the already-open <think> block with reasoning, then emit "
    "</think><answer>the option label only</answer>."
)
_CROP_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


def _step_from_path(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _find_rollout_files(rollout_dir: Path, start_step: int | None, end_step: int | None) -> list[RolloutFile]:
    files: list[RolloutFile] = []
    for path in rollout_dir.glob("*.jsonl"):
        step = _step_from_path(path)
        if step is None:
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        files.append(RolloutFile(step=step, path=path))
    return sorted(files, key=lambda item: item.step)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalise_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _dataset_question(record: dict[str, Any]) -> str:
    """Mirror PA-OPD's dataset-side prompt cleaning."""

    problem = _normalise_text(record.get("problem", "")).replace("<image>", "").strip()
    problem = problem.replace(f"\n\n{_CROP_HINT}", "").replace(_CROP_HINT, "")
    return problem.strip()


def _rollout_question(prompt: Any) -> str:
    """Recover the MCQ text from the decoded student prompt."""

    text = _normalise_text(prompt)
    if text.startswith("user\n\n"):
        text = text[len("user\n\n") :]
    if _FORMAT_SUFFIX in text:
        text = text.split(_FORMAT_SUFFIX, 1)[0]
    return text.strip()


def _dataset_image_index(dataset_dir: Path) -> dict[str, list[ImageRecord]]:
    train_path = dataset_dir / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"Vision-OPD train.jsonl not found: {train_path}")
    index: dict[str, list[ImageRecord]] = {}
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            try:
                original_path = (dataset_dir / record["images"][0]).resolve()
                crop_path = (dataset_dir / record["teacher_images"][0]).resolve()
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"Invalid image metadata at {train_path}:{line_number}") from exc
            entry = ImageRecord(
                line_number=line_number,
                answer=str(record.get("answer", "")).strip().upper(),
                original_path=original_path,
                crop_path=crop_path,
            )
            index.setdefault(_dataset_question(record), []).append(entry)
    return index


def _attach_logged_image_paths(row: dict[str, Any]) -> str | None:
    """Prefer exact image paths written by newer PA-OPD rollout dumps."""

    original = _normalise_text(row.get("original_image_path"))
    crop = _normalise_text(row.get("crop_image_path"))
    if not original or not crop:
        return None
    original_path = Path(original).expanduser().resolve()
    crop_path = Path(crop).expanduser().resolve()
    row.update(
        {
            "image_match": "logged",
            "image_candidate_count": 1,
            "original_image_path": str(original_path),
            "crop_image_path": str(crop_path),
            "original_image_uri": original_path.as_uri(),
            "crop_image_uri": crop_path.as_uri(),
            "image_files_exist": original_path.is_file() and crop_path.is_file(),
        }
    )
    return "logged"


def _attach_image_record(row: dict[str, Any], index: dict[str, list[ImageRecord]]) -> str:
    """Attach exact paths when prompt matching is unambiguous.

    The GT label resolves the rare duplicate-question case when those duplicate
    records have different labels. If ambiguity remains, do not attach an
    arbitrary image: the dashboard displays that fact instead.
    """

    candidates = index.get(_rollout_question(row["input"]), [])
    if len(candidates) > 1 and row["gt"]:
        by_answer = [candidate for candidate in candidates if candidate.answer == row["gt"].strip().upper()]
        if len(by_answer) == 1:
            candidates = by_answer
    if len(candidates) != 1:
        row["image_match"] = "unmatched" if not candidates else "ambiguous"
        row["image_candidate_count"] = len(candidates)
        return row["image_match"]

    candidate = candidates[0]
    row.update(
        {
            "image_match": "matched",
            "image_candidate_count": 1,
            "dataset_line": candidate.line_number,
            "original_image_path": str(candidate.original_path),
            "crop_image_path": str(candidate.crop_path),
            "original_image_uri": candidate.original_path.as_uri(),
            "crop_image_uri": candidate.crop_path.as_uri(),
            "image_files_exist": candidate.original_path.is_file() and candidate.crop_path.is_file(),
        }
    )
    return "matched"


def _set_image_urls(rows: list[dict[str, Any]], html_dir: Path, url_mode: str) -> None:
    """Set portable relative URLs, or file URLs for explicitly local use."""

    for row in rows:
        if row.get("image_match") not in {"matched", "logged"}:
            continue
        for path_key, url_key in (
            ("original_image_path", "original_image_uri"),
            ("crop_image_path", "crop_image_uri"),
        ):
            path = Path(row[path_key]).expanduser().resolve()
            if url_mode == "file":
                row[url_key] = path.as_uri()
            else:
                row[url_key] = Path(os.path.relpath(path, start=html_dir)).as_posix()


def _row_payload(row: dict[str, Any], step: int, index: int) -> dict[str, Any]:
    format_valid = _as_float(row.get("pa_opd_format_valid"))
    score = _as_float(row.get("score"))
    if format_valid is None:
        # Older rollout files did not persist pa_opd_format_valid. Their score
        # is the closest available indicator and is clearly labeled in HTML.
        format_valid = score
        format_source = "score fallback"
    else:
        format_source = "pa_opd_format_valid"
    accuracy = _as_float(row.get("pa_opd_accuracy"))
    return {
        "id": f"{step}:{index}",
        "step": step,
        "index": index,
        "input": str(row.get("input", "")),
        "output": str(row.get("output", "")),
        "gt": str(row.get("gts", "")),
        "answer": str(row.get("pa_opd_parsed_answer", "")),
        "dataset_line": row.get("pa_opd_source_line"),
        "original_image_path": str(row.get("pa_opd_original_image_path", "")),
        "crop_image_path": str(row.get("pa_opd_crop_image_path", "")),
        "format_valid": format_valid,
        "format_source": format_source,
        "accuracy": accuracy,
        "rollout_n": row.get("pa_opd_rollout_n"),
        "score": score,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for step, values in sorted(by_step.items()):
        rollout_ns = sorted({str(value["rollout_n"]) for value in values if value["rollout_n"] is not None})
        summary.append(
            {
                "step": step,
                "rows": len(values),
                "format_rate": _mean([value["format_valid"] for value in values]),
                "accuracy_rate": _mean([value["accuracy"] for value in values]),
                "score_mean": _mean([value["score"] for value in values]),
                "rollout_n": ", ".join(rollout_ns) if rollout_ns else "—",
            }
        )
    return summary


def _safe_json(value: Any) -> str:
    # Do not allow a rollout string such as </script> to terminate the data tag.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")


def _page_html(rows: list[dict[str, Any]], summary: list[dict[str, Any]], source: Path) -> str:
    payload = _safe_json({"rows": rows, "summary": summary})
    escaped_source = escape(str(source))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PA-OPD rollout dashboard</title>
<style>
  :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --muted:#8b949e; --text:#e6edf3; --good:#3fb950; --bad:#f85149; --accent:#58a6ff; --warn:#d29922; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; color:var(--text); background:var(--bg); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  header {{ padding:24px max(20px, calc((100vw - 1500px)/2)); border-bottom:1px solid var(--line); background:linear-gradient(120deg,#101820,#161b22); }}
  h1 {{ margin:0 0 6px; font-size:26px; }} h2 {{ margin:26px 0 12px; font-size:18px; }}
  .muted {{ color:var(--muted); }} .container {{ width:min(1500px, calc(100% - 40px)); margin:0 auto 64px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
  .stat,.controls,.card,.table-wrap {{ border:1px solid var(--line); background:var(--panel); border-radius:10px; }}
  .stat {{ padding:14px; }} .stat .value {{ display:block; margin-top:4px; font-size:22px; font-weight:700; }}
  .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ position:sticky; top:0; background:#21262d; font-size:12px; color:var(--muted); }}
  tr:last-child td {{ border-bottom:0; }} .bar {{ width:110px; height:8px; border-radius:20px; background:#30363d; overflow:hidden; display:inline-block; vertical-align:middle; margin-right:6px; }} .bar > i {{ display:block; height:100%; background:var(--good); }}
  .controls {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; padding:14px; position:sticky; top:0; z-index:2; margin-bottom:14px; }}
  label {{ display:grid; gap:5px; color:var(--muted); font-size:12px; }} input,select,button {{ font:inherit; color:var(--text); border:1px solid var(--line); background:#0d1117; border-radius:6px; padding:8px; }} button {{ cursor:pointer; background:#21262d; }} button:hover {{ border-color:var(--accent); }}
  .result-note {{ color:var(--muted); margin:10px 2px; }} .cards {{ display:grid; gap:12px; }} .card {{ padding:14px; }}
  .card-top {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }} .tag {{ padding:2px 8px; border-radius:999px; font-size:12px; font-weight:650; background:#30363d; }} .good {{ background:color-mix(in srgb,var(--good) 25%,transparent); color:#7ee787; }} .bad {{ background:color-mix(in srgb,var(--bad) 25%,transparent); color:#ff7b72; }} .unknown {{ color:#d29922; }}
  details {{ border-top:1px solid var(--line); margin-top:10px; padding-top:9px; }} summary {{ cursor:pointer; color:var(--accent); }} pre {{ margin:10px 0 0; padding:12px; overflow:auto; white-space:pre-wrap; word-break:break-word; border-radius:7px; background:#0d1117; border:1px solid #21262d; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .pager {{ display:flex; gap:8px; align-items:center; justify-content:center; margin:18px 0; }}
  .media {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:12px 0; }}
  .media-item {{ min-width:0; padding:9px; border:1px solid var(--line); border-radius:8px; background:#0d1117; }} .media-item strong {{ display:block; margin-bottom:7px; }} .media-item a {{ color:var(--accent); }} .media-item img {{ display:block; width:100%; max-height:440px; object-fit:contain; background:#010409; border-radius:5px; }} .media-note {{ margin:12px 0; color:var(--warn); }}
  @media (max-width:800px) {{ .media {{ grid-template-columns:1fr; }} }}
  @media (max-width:650px) {{ .container {{ width:calc(100% - 24px); }} header {{ padding:20px 12px; }} th,td {{ padding:8px; }} }}
</style>
</head>
<body>
<header><h1>PA-OPD rollout dashboard</h1><div class="muted">Source: <code>{escaped_source}</code> · self-contained offline HTML</div></header>
<main class="container">
  <section id="stats" class="stats"></section>
  <h2>Step summary</h2><div class="table-wrap"><table><thead><tr><th>Step</th><th>Rows</th><th>rollout_n</th><th>Format-valid rate</th><th>Answer accuracy</th><th>Mean score</th></tr></thead><tbody id="summary"></tbody></table></div>
  <h2>Samples</h2>
  <section class="controls">
    <label>Step<select id="stepFilter"><option value="all">All steps</option></select></label>
    <label>Format<select id="formatFilter"><option value="all">All</option><option value="valid">Valid only</option><option value="invalid">Invalid only</option></select></label>
    <label>Answer<select id="accuracyFilter"><option value="all">All</option><option value="correct">Correct only</option><option value="incorrect">Incorrect only</option></select></label>
    <label>rollout_n<select id="rolloutFilter"><option value="all">All</option></select></label>
    <label>Search prompt / response<input id="query" placeholder="e.g. tower, <answer>, C"></label>
    <label>Rows per page<select id="pageSize"><option>20</option><option selected>50</option><option>100</option></select></label>
  </section>
  <div id="resultNote" class="result-note"></div><section id="cards" class="cards"></section><div id="pager" class="pager"></div>
</main>
<script id="rollout-data" type="application/json">{payload}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById('rollout-data').textContent);
  const $ = id => document.getElementById(id);
  const state = {{ page: 0 }};
  const percent = value => value == null ? '—' : (100 * value).toFixed(2) + '%';
  const metric = (label, value) => {{ const box=document.createElement('div'); box.className='stat'; box.innerHTML='<span class="muted"></span><strong class="value"></strong>'; box.firstChild.textContent=label; box.lastChild.textContent=value; return box; }};
  const formatClass = value => value == null ? 'unknown' : value >= .5 ? 'good' : 'bad';
  const formatLabel = (name, value) => value == null ? name + ': —' : name + ': ' + (value >= .5 ? '✓' : '✗');
  const createTag = (label, value) => {{ const span=document.createElement('span'); span.className='tag '+formatClass(value); span.textContent=formatLabel(label,value); return span; }};
  const appendPre = (details, title, text) => {{ const summary=document.createElement('summary'); summary.textContent=title; const pre=document.createElement('pre'); pre.textContent=text || '—'; details.append(summary,pre); }};
  const totalFormat = data.rows.filter(x => x.format_valid != null).map(x => x.format_valid);
  const totalAccuracy = data.rows.filter(x => x.accuracy != null).map(x => x.accuracy);
  const mean = xs => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : null;
  const imageMatched = data.rows.filter(row => row.image_match === 'matched' || row.image_match === 'logged').length;
  $('stats').append(metric('Steps', String(data.summary.length)), metric('Samples', String(data.rows.length)), metric('Format-valid', percent(mean(totalFormat))), metric('Answer accuracy', percent(mean(totalAccuracy))), metric('Image pairs', imageMatched ? `${{imageMatched}} / ${{data.rows.length}}` : 'not indexed'));
  const summaryBody = $('summary');
  for (const item of data.summary) {{ const tr=document.createElement('tr'); const rate = item.format_rate == null ? '—' : `<span class="bar"><i style="width:${{Math.max(0,Math.min(100,item.format_rate*100))}}%"></i></span>${{percent(item.format_rate)}}`; tr.innerHTML=`<td>${{item.step}}</td><td>${{item.rows}}</td><td>${{item.rollout_n}}</td><td>${{rate}}</td><td>${{percent(item.accuracy_rate)}}</td><td>${{item.score_mean == null ? '—' : item.score_mean.toFixed(4)}}</td>`; summaryBody.appendChild(tr); }}
  const steps=[...new Set(data.rows.map(x=>x.step))].sort((a,b)=>a-b); for(const step of steps) {{ const o=document.createElement('option'); o.value=step; o.textContent='Step '+step; $('stepFilter').appendChild(o); }}
  const ns=[...new Set(data.rows.map(x=>x.rollout_n).filter(x=>x!=null))].sort((a,b)=>a-b); for(const n of ns) {{ const o=document.createElement('option'); o.value=n; o.textContent=n; $('rolloutFilter').appendChild(o); }}
  function filtered() {{ const step=$('stepFilter').value, fmt=$('formatFilter').value, acc=$('accuracyFilter').value, n=$('rolloutFilter').value, q=$('query').value.trim().toLowerCase(); return data.rows.filter(row => {{ if(step!=='all' && String(row.step)!==step) return false; if(n!=='all' && String(row.rollout_n)!==n) return false; if(fmt==='valid' && !(row.format_valid>=.5)) return false; if(fmt==='invalid' && !(row.format_valid<.5)) return false; if(acc==='correct' && !(row.accuracy>=.5)) return false; if(acc==='incorrect' && !(row.accuracy<.5)) return false; return !q || (row.input+'\\n'+row.output+'\\n'+row.gt+' '+row.answer).toLowerCase().includes(q); }}); }}
  function render() {{ const rows=filtered(), size=Number($('pageSize').value), pages=Math.max(1,Math.ceil(rows.length/size)); state.page=Math.min(state.page,pages-1); const begin=state.page*size, selected=rows.slice(begin,begin+size); $('resultNote').textContent=`Showing ${{rows.length ? begin+1 : 0}}–${{Math.min(begin+size,rows.length)}} of ${{rows.length}} matching samples`; const cards=$('cards'); cards.replaceChildren(); for(const row of selected) {{ const card=document.createElement('article'); card.className='card'; const top=document.createElement('div'); top.className='card-top'; const title=document.createElement('strong'); title.textContent=`Step ${{row.step}} · sample ${{row.index+1}}`; const nTag=document.createElement('span'); nTag.className='tag'; nTag.textContent='rollout_n: '+(row.rollout_n ?? '—'); const answer=document.createElement('span'); answer.className='tag'; answer.textContent=`GT: ${{row.gt || '—'}} · answer: ${{row.answer || '—'}}`; top.append(title,nTag,createTag('Format',row.format_valid),createTag('Answer',row.accuracy),answer); card.appendChild(top); if ((row.image_match === 'matched' || row.image_match === 'logged') && row.image_files_exist) {{ const media=document.createElement('section'); media.className='media'; for (const item of [['Original image',row.original_image_uri,row.original_image_path],['Evidence crop',row.crop_image_uri,row.crop_image_path]]) {{ const panel=document.createElement('div'); panel.className='media-item'; const caption=document.createElement('strong'); caption.textContent=item[0]; const link=document.createElement('a'); link.href=item[1]; link.target='_blank'; link.rel='noopener'; link.textContent='Open image'; link.title=item[2]; const image=document.createElement('img'); image.loading='lazy'; image.src=item[1]; image.alt=item[0]; image.onerror=()=>{{ image.style.display='none'; link.textContent='Image file is unavailable'; }}; panel.append(caption,link,image); media.appendChild(panel); }} card.appendChild(media); }} else if (row.image_match) {{ const note=document.createElement('div'); note.className='media-note'; note.textContent=row.image_match === 'ambiguous' ? `Image mapping is ambiguous (${{row.image_candidate_count}} dataset candidates); no image is shown.` : 'No matching Vision-OPD image record was found.'; card.appendChild(note); }} const response=document.createElement('details'); response.open=true; appendPre(response,'Student output',row.output); card.appendChild(response); const prompt=document.createElement('details'); appendPre(prompt,'Prompt',row.input); card.appendChild(prompt); const raw=document.createElement('details'); appendPre(raw,`Metadata · format source: ${{row.format_source}} · score: ${{row.score ?? '—'}}`,JSON.stringify({{step:row.step,index:row.index,gts:row.gt,parsed_answer:row.answer,format_valid:row.format_valid,accuracy:row.accuracy,rollout_n:row.rollout_n,score:row.score}},null,2)); card.appendChild(raw); cards.appendChild(card); }} const pager=$('pager'); pager.replaceChildren(); const prev=document.createElement('button'); prev.textContent='← Previous'; prev.disabled=state.page===0; prev.onclick=()=>{{state.page--;render()}}; const label=document.createElement('span'); label.textContent=`Page ${{state.page+1}} / ${{pages}}`; const next=document.createElement('button'); next.textContent='Next →'; next.disabled=state.page>=pages-1; next.onclick=()=>{{state.page++;render()}}; pager.append(prev,label,next); }}
  for(const id of ['stepFilter','formatFilter','accuracyFilter','rolloutFilter','query','pageSize']) $(''+id).addEventListener(id==='query'?'input':'change',()=>{{state.page=0;render()}}); render();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path, help="Directory containing numeric *.jsonl rollout dumps.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <rollout_dir>/rollouts_dashboard.html).")
    parser.add_argument("--start-step", type=int, default=None, help="Include steps greater than or equal to this value.")
    parser.add_argument("--end-step", type=int, default=None, help="Include steps less than or equal to this value.")
    parser.add_argument("--max-rows-per-step", type=int, default=None, help="Optional cap per step, useful for a small preview file.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Vision-OPD-6K directory; attaches original and crop images by exact prompt matching.")
    parser.add_argument(
        "--image-url-mode",
        choices=("relative", "file"),
        default="relative",
        help="Image URL style: relative (default; use with an HTTP server) or file (only for direct local file access).",
    )
    args = parser.parse_args()

    rollout_dir = args.rollout_dir.expanduser().resolve()
    if not rollout_dir.is_dir():
        parser.error(f"rollout directory does not exist: {rollout_dir}")
    if args.max_rows_per_step is not None and args.max_rows_per_step < 1:
        parser.error("--max-rows-per-step must be positive")
    output = (args.output or rollout_dir / "rollouts_dashboard.html").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    malformed = 0
    for item in _find_rollout_files(rollout_dir, args.start_step, args.end_step):
        with item.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if args.max_rows_per_step is not None and index >= args.max_rows_per_step:
                    break
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(raw, dict):
                    malformed += 1
                    continue
                rows.append(_row_payload(raw, item.step, index))

    if not rows:
        parser.error("no valid JSONL rollout rows matched the selection")

    image_match_counts: dict[str, int] = {}
    for row in rows:
        status = _attach_logged_image_paths(row)
        if status is not None:
            image_match_counts[status] = image_match_counts.get(status, 0) + 1
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        image_index = _dataset_image_index(dataset_dir)
        for row in rows:
            if row.get("image_match") is not None:
                continue
            status = _attach_image_record(row, image_index)
            image_match_counts[status] = image_match_counts.get(status, 0) + 1

    _set_image_urls(rows, output.parent, args.image_url_mode)
    summary = _summarize(rows)
    output.write_text(_page_html(rows, summary, rollout_dir), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"steps={len(summary)} rows={len(rows)} malformed_rows_skipped={malformed}")
    if image_match_counts:
        print("image_mapping=" + ", ".join(f"{key}={value}" for key, value in sorted(image_match_counts.items())))
    if args.image_url_mode == "relative":
        print("image_urls=relative (serve the dashboard from a common parent directory over HTTP)")


if __name__ == "__main__":
    main()
