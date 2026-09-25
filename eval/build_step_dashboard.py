"""Build a self-contained offline dashboard; audit summaries against JSONL first."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

BENCHES = {'lrs-vqa': 'LRS-VQA', 'xlrs-bench': 'XLRS-Bench', 'mme-realworld-rs': 'MME-RealWorld-RS'}
COLORS = {'lrs-vqa': '#2563eb', 'xlrs-bench': '#d97706', 'mme-realworld-rs': '#0f766e'}
STEPS = list(range(30, 211, 30))
PREFIX = 'dp4-gpu32-2k-kl-sbbox-tthree-bs96-ep3-gtm-r2'
METRICS = ['accuracy', 'tolerant_accuracy', 'macro_category_accuracy', 'macro_category_tolerant_accuracy']
esc = lambda x: html.escape(str(x), quote=True)


def load(root):
    rows, identities, settings = [], {}, []
    for step in STEPS:
        for bench in BENCHES:
            directory = root / f'{PREFIX}-step{step}-{bench}'
            source = directory / 'summary.json'
            data = json.loads(source.read_text())
            summary = data['datasets'][bench]
            records = [json.loads(line) for line in (directory / f'{bench}.jsonl').read_text().splitlines() if line.strip()]
            ids = {r['sample_id'] for r in records}
            assert len(ids) == len(records) == summary['samples'], f'Duplicate or mismatched samples: {source}'
            assert len(records) == data['datasets']['inspections'][bench]['eligible'], f'Incomplete run: {source}'
            assert sum(bool(r['correct']) for r in records) == summary['correct'], source
            assert sum(r['status'] != 'ok' for r in records) == summary['errors'], source
            assert math.isclose(summary['accuracy'], summary['correct'] / len(records)), source
            identity = {(r['sample_id'], json.dumps(r['ground_truth'], sort_keys=True)) for r in records}
            assert bench not in identities or identities[bench] == identity, f'Sample set changed: {bench}'
            identities[bench] = identity
            if bench == 'lrs-vqa':
                assert all('tolerant_correct' in r for r in records), source
                assert sum(bool(r['tolerant_correct']) for r in records) == summary['tolerant_correct'], source
                assert math.isclose(summary['tolerant_accuracy'], summary['tolerant_correct'] / len(records)), source
                settings.append(data['inference']['lrs_semantic'])
            for metric in ['accuracy'] + (['tolerant_accuracy'] if bench == 'lrs-vqa' else []):
                categories = summary['by_category']
                assert math.isclose(summary['macro_category_' + metric], sum(c[metric] for c in categories.values()) / len(categories)), source
            rows.append(dict(step=step, benchmark=bench, source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **summary))
    assert all(s == settings[0] for s in settings), 'BGE scoring settings changed between checkpoints'
    return rows, settings[0]


def chart(series, title, full=False):
    # SVG is pre-rendered: charts remain usable with JavaScript disabled.
    values = [v * 100 for _, vals, _, _ in series for v in vals]
    lo = 0 if full else max(0, math.floor(min(values) - 1))
    hi = 100 if full else min(100, math.ceil(max(values) + 1))
    width, height, left, right, top, bottom = 1000, 330, 62, 25, 24, 54
    x = lambda step: left + (step - 30) / 180 * (width-left-right)
    y = lambda v: top + (hi-v*100) / (hi-lo) * (height-top-bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>']
    for n in range(6):
        v = lo + (hi-lo)*n/5
        yy = y(v/100)
        parts.append(f'<path d="M{left} {yy}H{width-right}" stroke="#e4eaf1"/><text x="{left-10}" y="{yy+4}" text-anchor="end">{v:.1f}%</text>')
    for step in STEPS:
        parts.append(f'<text x="{x(step)}" y="{height-28}" text-anchor="middle">{step}</text>')
    parts.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle">Training step</text>')
    for name, vals, color, dashed in series:
        points = ' '.join(f'{x(s)},{y(v)}' for s, v in zip(STEPS, vals))
        dash = 'stroke-dasharray="7 5"' if dashed else ''
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" {dash}/>')
        for step, v in zip(STEPS, vals):
            parts.append(f'<circle tabindex="0" cx="{x(step)}" cy="{y(v)}" r="4.5" fill="white" stroke="{color}" stroke-width="2"><title>{esc(name)} · Step {step} · {v*100:.4f}%</title></circle>')
    parts.append('</svg><div class="legend">')
    for name, vals, color, dashed in series:
        parts.append(f'<span><i style="background:{color}"></i>{esc(name)}{" · 虚线" if dashed else ""}</span>')
    return ''.join(parts) + '</div>'


def build(rows, semantic, root, output):
    by = {b: sorted([r for r in rows if r['benchmark'] == b], key=lambda r:r['step']) for b in BENCHES}
    cards = []
    for b, name in BENCHES.items():
        for metric, suffix in [('accuracy', 'Strict' if b == 'lrs-vqa' else 'Accuracy')] + ([('tolerant_accuracy', 'Tolerant')] if b == 'lrs-vqa' else []):
            best = max(r[metric] for r in by[b]); steps = ', '.join(str(r['step']) for r in by[b] if r[metric] == best)
            delta = (by[b][-1][metric]-by[b][0][metric])*100
            cards.append(f'<article class="stat"><div>{name} · {suffix}</div><strong>{best*100:.2f}<small>%</small></strong><p>最佳 step {steps} <span>210 vs 30：{delta:+.2f} pp</span></p></article>')
    overview = []
    for macro in [False, True]:
        series = []
        for b, name in BENCHES.items():
            metric = 'macro_category_accuracy' if macro else 'accuracy'
            series.append((name + (' · strict' if b == 'lrs-vqa' else ''), [r[metric] for r in by[b]], COLORS[b], False))
            if b == 'lrs-vqa':
                metric = 'macro_category_tolerant_accuracy' if macro else 'tolerant_accuracy'
                series.append(('LRS-VQA · tolerant', [r[metric] for r in by[b]], '#7c3aed', True))
        for full in [False, True]:
            overview.append(f'<div class="variant" data-macro="{int(macro)}" data-full="{int(full)}">{chart(series,"Accuracy vs training step",full)}</div>')
    detail = []
    for b, name in BENCHES.items():
        series = [(name, [r['accuracy'] for r in by[b]], COLORS[b], False)]
        if b == 'lrs-vqa': series.append(('Tolerant', [r['tolerant_accuracy'] for r in by[b]], '#7c3aed', True))
        detail.append(f'<section class="panel"><h2>{name} <small>n = {by[b][0]["samples"]:,} / step</small></h2>{chart(series,name)}<details><summary>展开类别 accuracy 曲线（{len(by[b][0]["by_category"])} 类）</summary>')
        for cat in sorted(by[b][0]['by_category']):
            seq = [(cat, [r['by_category'][cat]['accuracy'] for r in by[b]], COLORS[b], False)]
            if b == 'lrs-vqa': seq.append(('Tolerant', [r['by_category'][cat]['tolerant_accuracy'] for r in by[b]], '#7c3aed', True))
            detail.append(f'<h3>{esc(cat)}</h3>{chart(seq,cat)}')
        detail.append('</details></section>')
    table = ['<table><thead><tr><th>Step</th><th>Benchmark</th><th>Accuracy</th><th>Tolerant</th><th>Macro accuracy</th><th>Macro tolerant</th><th>Samples</th><th>Errors</th></tr></thead><tbody>']
    for r in rows:
        table.append(f'<tr><td>{r["step"]}</td><td>{BENCHES[r["benchmark"]]}</td>'+''.join(f'<td>{r[m]*100:.3f}%</td>' if m in r else '<td>—</td>' for m in METRICS)+f'<td>{r["samples"]:,}</td><td>{r["errors"]}</td></tr>')
    table.append('</tbody></table>')
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    errors = sum(r['errors'] for r in rows)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RS-OPSD · Checkpoint dashboard</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:#162339;font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1220px;margin:auto;padding:44px 24px}header{margin-bottom:28px}h1{font-size:34px;letter-spacing:-1px;margin:6px 0}h2{font-size:19px;margin:0 0 16px}h3{font-size:16px}p{margin:8px 0;color:#607089}.eyebrow{color:#0f766e;font-size:12px;letter-spacing:2px;font-weight:700}.badge{display:inline-block;background:#e3f4ed;color:#176449;border-radius:20px;padding:5px 12px;font-size:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.stat,.panel{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;box-shadow:0 6px 20px #15243b04}.stat>div{font-size:13px;color:#607089}.stat strong{font-size:36px;letter-spacing:-1px}.stat small{font-size:18px;margin-left:4px}.stat p{font-size:12px}.stat span{display:block}.panel{margin:20px 0}small{color:#607089;font-size:12px;font-weight:400}svg{width:100%;height:auto;display:block}svg text{font-size:12px;fill:#607089}circle:hover,circle:focus{r:7;outline:none}.legend{display:flex;gap:22px;justify-content:center;flex-wrap:wrap;font-size:12px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:18px}select,button{background:white;color:#162339;border:1px solid #d7e0ed;border-radius:8px;padding:8px 12px;font:inherit;cursor:pointer}button{background:#142f50;color:white;border:0}.variant{display:none}.variant[data-macro="0"][data-full="0"]{display:block}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #eef1f5}th{color:#607089;background:#f8fafc}th:nth-child(2),td:nth-child(2){text-align:left}details{margin-top:20px;border-top:1px solid #e2e8f0;padding-top:14px}summary{cursor:pointer;color:#2563eb}code{word-break:break-all;font-size:12px}.note{font-size:13px}footer{font-size:12px;color:#607089;margin-top:24px}@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:24px 12px}.panel{padding:16px}h1{font-size:27px}.stat{padding:16px}}@media print{button,.toolbar{display:none}.panel{break-inside:avoid}body{background:white}}
</style><main><header><div class="eyebrow">RS-OPSD / EVALUATION OBSERVATORY</div><h1>Accuracy 随 checkpoint 的变化</h1><p>Qwen3-VL-8B-Instruct · GPU32 · 2K KL · student bbox / teacher bbox + derived + crop<br>BS96 · EP3 · global-token-mean · r2</p>'''
    document += f'<span class="badge">7 checkpoints · 21 runs · {sum(r["samples"] for r in rows):,} 条评测记录 · {errors} errors</span></header><div class="cards">'+''.join(cards)+'</div>'
    document += '<section class="panel"><h2>总览</h2><div class="toolbar"><label>聚合方式 <select id="aggregate"><option value="0">样本平均 accuracy</option><option value="1">类别宏平均 accuracy</option></select></label><label>Y 轴 <select id="scale"><option value="0">自适应范围</option><option value="1">完整 0–100%</option></select></label><button id="download">下载指标 CSV</button></div>'+''.join(overview)+'<p class="note">悬停数据点查看精确数值。实线为原始 accuracy，紫色虚线为 LRS tolerant。默认 Y 轴为局部范围；各 benchmark 独立计分，不合并成总分。上方最佳值卡片按样本平均计分。</p></section>'
    document += ''.join(detail)+'<section class="panel"><h2>完整指标表</h2><div class="scroll">'+''.join(table)+'</div></section>'
    document += f'<section class="panel"><h2>指标说明与数据核验</h2><p class="note">Accuracy = correct / samples，错误请求计为错误。Macro accuracy 是各类别 accuracy 的等权平均。LRS tolerant 保留 strict 判断，增加形态词/同义词匹配和 BGE 余弦相似度判断；布尔与数字答案保留严格判断。阈值 {semantic["threshold"]}，模型 <code>{esc(semantic["model_path"])}</code>。</p><p class="note">已核对 JSONL 条数、唯一 sample ID、跨 step 的样本与 GT 一致性、正确数、错误数及宏平均。每组样本数与 eligible 一致。源文件路径与 SHA256 记录在 dashboard-data.json。</p><details><summary>数据来源</summary><code>{esc(root)}</code><ul>'+''.join(f'<li><code>{esc(r["source"])}</code></li>' for r in rows)+f'</ul></details></section><footer>生成时间 {generated} · 离线静态快照，无 CDN 或网络依赖 · 分数均以百分比显示，变化以百分点 pp 表示</footer></main>'
    fields = ['step','benchmark']+METRICS+['samples','correct','tolerant_correct','errors','source']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    csv_text=(output/'metrics.csv').read_text(encoding='utf-8-sig')
    document += '<script>const csvData='+json.dumps(csv_text).replace('<','\\u003c')+''';
function refresh(){document.querySelectorAll('.variant').forEach(p=>{p.style.display=p.dataset.macro===document.getElementById('aggregate').value&&p.dataset.full===document.getElementById('scale').value?'block':'none';});}
document.getElementById('aggregate').addEventListener('change',refresh);document.getElementById('scale').addEventListener('change',refresh);
document.getElementById('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob(['\\ufeff',csvData],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></html>'''
    (output/'index.html').write_text(document,encoding='utf-8')
    (output/'dashboard-data.json').write_text(json.dumps(dict(generated_at=generated,semantic=semantic,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Dashboard:', output/'index.html')
    for b in BENCHES:
        for m in ['accuracy']+(['tolerant_accuracy'] if b=='lrs-vqa' else []):
            best=max(by[b],key=lambda r:r[m]);print(b,m,'best step',best['step'],f'{best[m]*100:.3f}%')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Root containing evaluation results.')
    parser.add_argument('--output',type=Path,default=Path('eval/dashboards/gpu32-gtm-r2'))
    args=parser.parse_args()
    rows,semantic=load(args.root)
    build(rows,semantic,args.root,args.output)
