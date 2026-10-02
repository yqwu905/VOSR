"""Bounded sampling, standalone galleries and measured diagnostics for SR runs."""

import base64
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .augmentation import degrade, sample_seed
from .backends import read_rgb
from .geometry import iou
from .pipeline import atomic_json
from .quality import normalized_text


IMAGE_VIEWER_STYLE = """
img[src]{cursor:zoom-in}
.image-viewer{width:calc(100vw - 48px);max-width:none;max-height:calc(100vh - 48px);box-sizing:border-box;border:0;border-radius:8px;padding:16px;background:white;color:#172231}
.image-viewer::backdrop{background:rgba(0,0,0,.72)}
.image-viewer header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:12px}
.image-viewer header strong{overflow-wrap:anywhere}
.image-viewer button{flex-shrink:0;padding:8px 16px;cursor:pointer}
.image-viewer-body{overflow:auto;max-height:calc(100vh - 140px)}
.image-viewer-body img{display:block;width:auto;height:auto;max-width:none;cursor:default}
"""

IMAGE_VIEWER_HTML = """<dialog id="image-viewer" class="image-viewer" aria-labelledby="image-viewer-title">
<header><strong id="image-viewer-title">原尺寸图片</strong><button id="image-viewer-close" type="button">关闭</button></header>
<div class="image-viewer-body"><img id="image-viewer-image" alt="原尺寸图片"></div>
</dialog>"""

IMAGE_VIEWER_SCRIPT = """
const viewer=document.getElementById('image-viewer');
const viewerImage=document.getElementById('image-viewer-image');
const viewerTitle=document.getElementById('image-viewer-title');
document.querySelectorAll('img[src]').forEach(preview=>{
  const caption=preview.closest('figure')?.querySelector('figcaption')?.textContent.trim();
  const title=caption||preview.alt||'原尺寸图片';
  preview.tabIndex=0;
  preview.setAttribute('role','button');
  preview.setAttribute('aria-label','查看原尺寸：'+title);
  preview.title='点击查看原尺寸，按 Esc 关闭';
  const open=()=>{
    viewerImage.src=preview.src;
    viewerImage.alt=title;
    viewerTitle.textContent=title;
    viewer.showModal();
  };
  preview.onclick=open;
  preview.onkeydown=event=>{
    if(event.key==='Enter'||event.key===' '){event.preventDefault();open();}
  };
});
document.getElementById('image-viewer-close').onclick=()=>viewer.close();
viewer.addEventListener('click',event=>{if(event.target===viewer)viewer.close();});
viewer.addEventListener('close',()=>viewerImage.removeAttribute('src'));
"""


def png_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def annotation_errors(predictions, annotations):
    """CER against provided transcripts is a diagnostic, not a glyph verifier."""
    errors, characters, exact, matches = 0, 0, 0, 0
    used = set()
    for annotation in annotations:
        reference = normalized_text(annotation.get("text", ""))
        if not reference:
            continue
        candidates = [(iou(annotation["bbox"], p["bbox"]), i, p)
                      for i, p in enumerate(predictions) if i not in used]
        overlap, index, prediction = max(candidates, default=(0, -1, {}), key=lambda value: value[0])
        text = normalized_text(prediction.get("text", "")) if overlap >= 0.25 else ""
        if overlap >= 0.25:
            used.add(index)
        errors += edit_distance(reference, text)
        characters += len(reference)
        exact += reference == text
        matches += 1
    return {"edit_errors": errors, "reference_characters": characters, "exact_matches": exact,
            "transcripts": matches, "cer": errors / characters if characters else None}


def image_diagnostics(image):
    import cv2
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {"laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "luminance_std": float(gray.std())}


def outlined(image, predictions, threshold):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for region in predictions:
        color = "#00b86b" if region.get("confidence", 0) >= threshold else "#ee5533"
        draw.line([tuple(point) for point in region["polygon"]] + [tuple(region["polygon"][0])],
                  fill=color, width=max(2, round(image.width / 256)))
    return image


def panel(image, size=(384, 288)):
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, "#eceff3")
    result.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return result


def build_report(run, output, count=12, seed=42):
    if count < 1:
        raise ValueError("visualization count must be positive")
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    rng = random.Random(seed)
    pools, seen = defaultdict(list), Counter()
    statuses, reasons = defaultdict(Counter), defaultdict(Counter)
    confidence = defaultdict(lambda: {"count": 0, "sum": 0.0, "histogram": [0] * 20})
    transcript_totals = defaultdict(Counter)
    examples = 0
    for name in ("accepted", "review", "rejected", "errors"):
        with (run / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                source, status = record["source"], record["status"]
                statuses[source][status] += 1
                reasons[source].update(record.get("reasons", []))
                group = (source, status)
                if record.get("hr_path"):
                    seen[group] += 1
                    pool = pools[group]
                    if len(pool) < count:
                        pool.append(record)
                    else:
                        slot = rng.randrange(seen[group])
                        if slot < count:
                            pool[slot] = record
                for prefix, predictions, annotations in (
                    ("hr", record.get("ocr", []), record.get("annotations", [])),
                    ("original", record.get("original_ocr", []),
                     # Transform the crop annotations back into original image coordinates.
                     [])):
                    if prefix == "original" and record.get("annotations"):
                        from .geometry import transform
                        annotations = transform(record["annotations"], record["crop_to_original"])
                    errors = annotation_errors(predictions, annotations)
                    for key in ("edit_errors", "reference_characters", "exact_matches", "transcripts"):
                        transcript_totals[source][f"{prefix}_{key}"] += errors[key]
                # Reservoirs keep quantitative plot memory bounded too.
                for prediction in record.get("ocr", []):
                    value = float(prediction["confidence"])
                    confidence[source]["count"] += 1
                    confidence[source]["sum"] += value
                    confidence[source]["histogram"][min(19, int(value * 20))] += 1
                examples += 1
    chosen = []
    # Round-robin across source/status strata, preserving both successes and failures.
    groups = sorted(pools)
    for pool in pools.values():
        rng.shuffle(pool)
    while len(chosen) < count:
        changed = False
        for group in groups:
            if pools[group] and len(chosen) < count:
                chosen.append(pools[group].pop())
                changed = True
        if not changed:
            break
    summary = {"run": str(run), "run_input_sha256": metadata["input_sha256"], "seed": seed,
               "total_records": examples, "visualized": len(chosen),
               "statuses": {s: dict(v) for s, v in statuses.items()},
               "reasons": {s: dict(v) for s, v in reasons.items()},
               "transcript_diagnostics": {},
               "limitations": ["Teacher outputs are pseudo GT.", "CER uses dataset transcripts where available; confidence is not accuracy.",
                               "No original-HR paired ground truth is supplied; no PSNR/SSIM accuracy claim.",
                               "Laplacian variance measures edge energy, not naturalness or glyph correctness.",
                               "Audit outputs remain ineligible for training."]}
    for source, totals in transcript_totals.items():
        entry = dict(totals)
        for prefix in ("original", "hr"):
            denominator = totals[f"{prefix}_reference_characters"]
            entry[f"{prefix}_cer"] = totals[f"{prefix}_edit_errors"] / denominator if denominator else None
        summary["transcript_diagnostics"][source] = entry

    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    canvas = Image.new("RGB", (4 * 384, max(1, len(chosen)) * 340), "white")
    draw = ImageDraw.Draw(canvas)
    cards, measurements = [], []
    for index, record in enumerate(chosen):
        original = read_rgb(record["original_crop_path"])
        hr = read_rgb(record["hr_path"])
        variants = record.get("variants", [])
        if variants:
            variant = max(variants, key=lambda value: value["degradation"]["scale"])
            lr = read_rgb(variant["lr_path"])
            parameters = variant["degradation"]
            lr_preview_only = False
        else:
            lr, _, parameters = degrade(hr, record.get("ocr", []), config["augmentation"],
                                        sample_seed(seed, record["sample_key"], "visualization"))
            lr_preview_only = True
        lr_up = lr.resize(hr.size, Image.Resampling.NEAREST)
        boxes = outlined(hr, record.get("ocr", []), config["quality"]["ocr_min_confidence"])
        images = (original, hr, lr_up, boxes)
        labels = ("Original (aligned bicubic)", "VOSR2 pseudo GT", f"Degraded LR x{parameters['scale']} (nearest)", "OCR: green=pass / red=low")
        image_sources = []
        for column, (im, label) in enumerate(zip(images, labels)):
            path = previews / f"{index:03d}_{column}.png"
            im.save(path)
            image_sources.append(png_data_url(path))
            draw.text((column * 384 + 8, index * 340 + 25), label, fill="black")
            canvas.paste(panel(im), (column * 384, index * 340 + 46))
        title = f"{record['source']} | {record['status']} | {record['sample_id']} | {record['crop_id']}"
        draw.text((8, index * 340 + 4), title[:170], fill="black")
        diagnostics = {"sample_key": record["sample_key"], "source": record["source"],
                       "crop_id": record["crop_id"], "original": image_diagnostics(original),
                       "hr": image_diagnostics(hr), "lr": image_diagnostics(lr),
                       "degradation": parameters, "lr_preview_only": lr_preview_only}
        measurements.append(diagnostics)
        text = " | ".join(f"{p['text']} ({p['confidence']:.2f})" for p in record.get("ocr", []))
        figures = "".join(f'<figure><img src="{src}" alt="{html.escape(label)}" loading="lazy">'
                          f'<figcaption>{html.escape(label)}</figcaption></figure>'
                          for src, label in zip(image_sources, labels))
        details = {"reasons": record["reasons"], "annotations": record.get("annotations", []),
                   "ocr": record.get("ocr", []), "original_ocr": record.get("original_ocr", []),
                   "diagnostics": diagnostics, "teacher": record["teacher"],
                   "timings_seconds": record.get("timings_seconds", {})}
        cards.append(f'<article data-source="{html.escape(record["source"])}" data-status="{record["status"]}">'
                     f'<h3>{html.escape(title)}</h3><div class="images">{figures}</div>'
                     f'<p>{html.escape(text)}</p><p>{html.escape(", ".join(record["reasons"]))}</p>'
                     f'<details><summary>OCR / annotations / degradation / timing</summary>'
                     f'<pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details></article>')
    canvas.save(output / "comparison.png")
    summary["measurements"] = measurements
    summary["ocr_confidence"] = {source: {"count": values["count"],
                                         "mean": values["sum"] / values["count"] if values["count"] else None,
                                         "histogram": values["histogram"]}
                                 for source, values in confidence.items()}
    atomic_json(output / "analysis.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    sources = sorted(statuses)
    bottom = np.zeros(len(sources))
    for status, color in (("accepted", "#00a878"), ("review", "#f4b942"), ("rejected", "#d9534f"), ("error", "#555555")):
        values = np.asarray([statuses[source][status] for source in sources])
        axes[0].bar(sources, values, bottom=bottom, label=status, color=color)
        bottom += values
    axes[0].set_title("Crop decisions"); axes[0].set_ylabel("Crops"); axes[0].legend()
    for source, values in confidence.items():
        if values["count"]:
            axes[1].stairs(values["histogram"], np.linspace(0, 1, 21), fill=True, alpha=0.45, label=source)
    axes[1].axvline(config["quality"]["ocr_min_confidence"], color="black", linestyle="--", label="audit OCR threshold")
    axes[1].set_title("OCR confidence (not accuracy)"); axes[1].set_xlabel("Confidence"); axes[1].legend()
    fig.savefig(output / "statistics.png", dpi=160)
    plt.close(fig)
    options = "".join(f'<option>{html.escape(source)}</option>' for source in sources)
    mode = config["quality"].get("mode", "strict")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>VOSR2 数据管线抽样分析</title>
<style>body{{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172231}}article{{background:white;padding:16px;margin:20px 0;border-radius:8px}}.images{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}figure{{margin:0}}img{{width:100%;object-fit:contain}}figcaption{{font-size:12px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}select{{padding:8px}}.stats{{max-width:1100px}}@media(max-width:800px){{.images{{grid-template-columns:repeat(2,1fr)}}}}{IMAGE_VIEWER_STYLE}</style>
<h1>VOSR2：数据合成管线抽样分析</h1><p>原图 → VOSR2 伪 GT → 退化 LR → OCR。点击图片可查看原尺寸，可滚动查看细节；按 Esc 或“关闭”返回。</p>
<p>本实验为 {mode} 模式。OCR 置信度、边缘能量和伪 GT 均不能证明文字正确；无可靠文字依据或字形核验的样本不能用于训练。</p>
<p>来源 <select id="source"><option value="all">全部</option>{options}</select> 状态 <select id="status"><option value="all">全部</option><option>review</option><option>rejected</option><option>accepted</option><option>error</option></select></p>
<img class="stats" src="{png_data_url(output / 'statistics.png')}" alt="裁剪决策与 OCR 置信度统计"><details><summary>汇总指标与限制</summary><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k != "measurements"}, ensure_ascii=False, indent=2))}</pre></details>
{''.join(cards)}{IMAGE_VIEWER_HTML}<script>const sourceFilter=document.getElementById('source'),statusFilter=document.getElementById('status');function filter(){{document.querySelectorAll('article[data-source]').forEach(a=>a.hidden=!((sourceFilter.value==='all'||a.dataset.source===sourceFilter.value)&&(statusFilter.value==='all'||a.dataset.status===statusFilter.value)))}}sourceFilter.onchange=statusFilter.onchange=filter;{IMAGE_VIEWER_SCRIPT}</script></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")
    return {"report": str(output / "report.html"), "comparison": str(output / "comparison.png"),
            "analysis": str(output / "analysis.json"), "visualized": len(chosen)}
