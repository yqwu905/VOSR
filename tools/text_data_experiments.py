"""Experiments behind docs/text_data_strategy.md.

  python tools/text_data_experiments.py legibility --output legibility.json
  python tools/text_data_experiments.py screensr --screensr /data/ScreenSR --output screensr.json

``legibility`` renders clean Chinese/English text lines with exact
transcripts, degrades them with every profile and severity (x4), restores
them by bicubic interpolation and measures RapidOCR CER. ``screensr``
compares synthetic LR with ScreenSR's real re-photographed LR on spectral
transfer, tone, colour and noise features; pairs 0-39 are for diagnosis and
only the rest are reported. Needs ``rapidocr_onnxruntime`` and CJK fonts.
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.augmentation import degrade  # noqa: E402
from data_pipeline.config import validate_config  # noqa: E402
from data_pipeline.degradations import PROFILES, synthesize  # noqa: E402
from data_pipeline.probes import noise_sigma, to_gray  # noqa: E402

FONTS = {"sans_zh": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "serif_zh": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
         "kai_zh": "/usr/share/fonts/truetype/arphic/ukai.ttc",
         "sans_en": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "serif_en": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "mono_en": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"}
ZH = ["数据清洗与增强方案", "超分辨率文字恢复", "营业时间上午九点", "禁止停车违者拖车", "北京市海淀区学院路", "限时特价全场五折",
      "请勿触摸展品", "出口在左侧通道", "会议纪要第三章", "温馨提示小心地滑", "人民币壹佰贰拾元", "高铁站换乘指南"]
EN = ["Quick brown fox 42", "OPEN 24 HOURS", "Invoice No 4471-B", "Total due $1,284.50", "Platform 9 Exit B",
      "Super Resolution", "Keep Clear At All Times", "Fresh Coffee Daily", "Section 3.2 Results", "Call 555-0198 now"]
LEGACY = validate_config({
    "backends": {"teacher": {"type": "identity"}, "ocr": {"type": "sidecar_ocr"},
                 "iqa": {"type": "sidecar_iqa"}, "glyph": {"type": "identity_source"}},
    "quality": {"ocr_min_confidence": 0.9, "original_min_confidence": 0.95, "mode": "audit"},
    "augmentation": {"scales": [4]}})["augmentation"]


def text_line(text, font, height, fg=(20, 20, 30), bg=(245, 242, 235)):
    face = ImageFont.truetype(FONTS[font], height)
    box = face.getbbox(text)
    pad = max(6, height // 3)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    image = Image.new("RGB", (w + (-w) % 8, h + (-h) % 8), bg)
    ImageDraw.Draw(image).text((pad - box[0], pad - box[1]), text, font=face, fill=fg)
    return image


def cer(prediction, truth):
    a, b = "".join(prediction.split()), "".join(truth.split())
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
    return min(1.0, row[len(b)] / max(1, len(b)))


def legibility(args):
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()

    def read(image):  # a PIL input is converted RGB->BGR by RapidOCR
        result, _ = engine(image, use_det=False, use_cls=False, use_rec=True)
        return result[0][0] if result else ""

    rng = random.Random(0)
    lines = []
    for height in (16, 24, 32, 48):
        for i in range(8):
            lines.append(dict(lang="zh", text=ZH[i % len(ZH)], height=height,
                              font=rng.choice(["sans_zh", "serif_zh", "kai_zh"])))
            lines.append(dict(lang="en", text=EN[i % len(EN)], height=height,
                              font=rng.choice(["sans_en", "serif_en", "mono_en"])))
    rows = []
    for k, spec in enumerate(lines):
        fg = (rng.randint(0, 90), rng.randint(0, 90), rng.randint(0, 120))
        bg = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(190, 250))
        hr = text_line(spec["text"], spec["font"], spec["height"], fg, bg)
        rows.append(dict(spec, profile="hr", severity=None, cer=cer(read(hr), spec["text"])))
        lr, _, _ = degrade(hr, [], LEGACY, seed=k)
        rows.append(dict(spec, profile="legacy", severity=None,
                         cer=cer(read(lr.resize(hr.size, Image.Resampling.BICUBIC)), spec["text"])))
        for profile in PROFILES:
            for severity in (0.25, 0.5, 0.75):
                for seed in range(2):
                    lr, _ = synthesize(hr, 1000 * k + seed, 4, profile, severity=severity)
                    rows.append(dict(spec, profile=profile, severity=severity,
                                     cer=cer(read(lr.resize(hr.size, Image.Resampling.BICUBIC)), spec["text"])))
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False))
    return rows


BANDS = [(0.02, 0.05), (0.05, 0.1), (0.1, 0.18), (0.18, 0.3), (0.3, 0.5)]


def screensr_features(hr, lr):
    """Log power transfer per radial band, tone/colour drift, noise and chroma detail."""
    h, l = np.asarray(hr, dtype=np.float32), np.asarray(lr, dtype=np.float32)
    radius = np.hypot(*np.meshgrid(np.fft.fftfreq(h.shape[1]), np.fft.fftfreq(h.shape[0])))
    window = np.outer(np.hanning(h.shape[0]), np.hanning(h.shape[1]))
    power = lambda g: np.abs(np.fft.fft2((g - g.mean()) * window)) ** 2
    gh, gl = to_gray(h.astype(np.uint8)), to_gray(l.astype(np.uint8))
    ph, pl = power(gh), power(gl)
    out = {f"transfer_{a}_{b}": float(np.log10(pl[(radius >= a) & (radius < b)].sum() /
                                                ph[(radius >= a) & (radius < b)].sum())) for a, b in BANDS}
    lh, ll = cv2.GaussianBlur(h, (0, 0), 8), cv2.GaussianBlur(l, (0, 0), 8)
    gray = lambda x: to_gray(np.clip(x, 0, 255).astype(np.uint8))
    out["luma_shift"] = float((gray(ll) - gray(lh)).mean())
    out["contrast_ratio"] = float(gray(ll).std() / max(1e-3, gray(lh).std()))
    centred = lambda x: x - x.mean(axis=2, keepdims=True)
    out["color_shift"] = float(np.abs(centred(ll) - centred(lh)).mean())
    out["noise_lr"] = noise_sigma(gl)
    high = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2)
    out["chroma_hf"] = float(np.abs(high(centred(l)) - high(centred(h))).mean())
    return out


def screensr(args):
    hr_paths = sorted(glob.glob(str(Path(args.screensr) / "HR_512" / "*.png")))
    lr_paths = sorted(glob.glob(str(Path(args.screensr) / "LR_512" / "*.png")))
    conditions = {"real": None, "bicubic_x2": "bicubic@2", "bicubic_x3": "bicubic@3", "bicubic_x4": "bicubic@4",
                  "legacy": "legacy@4", "mixture": "mixture@4", **{p: f"{p}@4" for p in PROFILES},
                  "screen_photo_x3": "screen_photo@3", "mixture_x3": "mixture@3", "realesrgan_x3": "realesrgan@3"}
    results = {name: [] for name in conditions}
    for i, (hr_path, lr_path) in enumerate(zip(hr_paths, lr_paths)):
        hr = Image.open(hr_path).convert("RGB")
        for name, kind in conditions.items():
            if kind is None:
                restored = Image.open(lr_path).convert("RGB")
            else:
                method, scale = kind.split("@")
                size = (round(hr.width / int(scale)), round(hr.height / int(scale)))
                if method == "bicubic":
                    lr = hr.resize(size, Image.Resampling.BICUBIC)
                elif method == "legacy":
                    lr = degrade(hr, [], LEGACY, seed=i)[0]
                else:
                    lr = synthesize(hr, i, int(scale), None if method == "mixture" else method)[0]
                restored = lr.resize(hr.size, Image.Resampling.BICUBIC)
            results[name].append(screensr_features(hr, restored))
    held = {name: rows[40:] for name, rows in results.items()}  # 0-39: diagnosis only
    keys = list(held["real"][0])
    matrix = lambda rows: np.array([[r[k] for k in keys] for r in rows])
    mean, std = matrix(held["real"]).mean(0), matrix(held["real"]).std(0) + 1e-9
    real = (matrix(held["real"]) - mean) / std
    summary = {}
    for name, rows in held.items():
        synthetic = (matrix(rows) - mean) / std
        distance = np.linalg.norm(real[:, None] - synthetic[None], axis=2)
        summary[name] = {"coverage_median_nn": float(np.median(distance.min(axis=1))),
                         "precision_median_nn": float(np.median(distance.min(axis=0))),
                         "feature_means": {k: float(np.mean([r[k] for r in rows])) for k in keys}}
    Path(args.output).write_text(json.dumps({"summary": summary, "features": results}, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("legibility")
    one.add_argument("--output", required=True)
    two = commands.add_parser("screensr")
    two.add_argument("--screensr", required=True, help="folder with HR_512/ and LR_512/")
    two.add_argument("--output", required=True)
    args = parser.parse_args()
    result = legibility(args) if args.command == "legibility" else screensr(args)
    print(json.dumps(result if args.command == "screensr" else {"rows": len(result)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
