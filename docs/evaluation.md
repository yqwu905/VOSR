# Evaluation: PSNR / SSIM / LPIPS / DISTS and OCR-A / CER / 1-NED

`evaluate.py` scores SR outputs (`--pred`, required) against an optional GT folder
(`--gt`), an optional LQ folder (`--lq`) and optional text-box annotations (`--ann`).

| Metric | Computed when | Images | Better |
| --- | --- | --- | --- |
| PSNR, SSIM | `--gt` is given | pred vs gt | higher |
| LPIPS, DISTS | `--gt` is given | pred vs gt | lower |
| OCR-A, 1-NED | `--ann` is given | pred, plus lq / gt when given | higher |
| CER | `--ann` is given | pred, plus lq / gt when given | lower |

## Installation

PSNR/SSIM/LPIPS/DISTS use `pyiqa`, already listed in `requirements.txt`. LPIPS and
DISTS download their weights (torchvision AlexNet/VGG16 and the pyiqa heads) into the
torch hub cache on first use.

OCR metrics need PaddleOCR 3.x, which is only imported when `--ann` is given:

```bash
pip install paddleocr          # PaddleOCR >= 3.0
pip install paddlepaddle       # CPU; or paddlepaddle-gpu for a GPU build
```

The recognition model (`PP-OCRv5_server_rec` by default) is downloaded to
`~/.paddlex/official_models` on first use.

## Usage

```bash
# Paired benchmark: FR-IQA + OCR of pred, lq and gt.
python evaluate.py \
    --pred preset/results \
    --gt   path/to/HR \
    --lq   path/to/LR \
    --ann  path/to/Label.txt \
    --ann-ref gt \
    --output preset/results_eval

# No GT: OCR metrics of pred (and lq) only.
python evaluate.py --pred preset/results --lq path/to/LR --ann path/to/Label.txt --ann-ref lq

# FR-IQA only, RealSR naming (Nikon_024_LR4.png vs Nikon_024_HR.png).
python evaluate.py --pred preset/results --gt path/to/HR --strip-suffix _LR4 _HR
```

Images are matched by file stem, so `x_AIGC_INPUT.jpg` (LQ), `x_AIGC_INPUT.png`
(pred, as written by the inference scripts) and the annotation line of
`x_AIGC_INPUT.jpg` belong together. Directory prefixes in annotation names are
ignored. When file names differ by a suffix, list the suffixes with `--strip-suffix`.
Every prediction needs a GT/LQ image when `--gt`/`--lq` is given; predictions without
an annotation line are only skipped for OCR. pred and gt must have the same size.

Main options:

- `--ann-ref {pred,lq,gt}`: the image whose pixel coordinates the annotation uses. Boxes
  are scaled to the other images by their size ratio, so LQ boxes shrink with the LQ.
  It is required when pred/lq/gt sizes differ, and the script stops instead of guessing.
- `--crop-border N`: pixels removed from each border before PSNR/SSIM/LPIPS/DISTS (default 0).
- `--rgb`: PSNR/SSIM on RGB instead of the Y channel.
- `--include-difficult`, `--ignore-case`: see text comparison below.
- `--ocr-model`: PaddleOCR recognition model (default `PP-OCRv5_server_rec`;
  `PP-OCRv5_mobile_rec` is faster on CPU). Only compare numbers from the same model.
- `--ocr-cls-model PP-LCNet_x1_0_textline_ori`: optional 0/180-degree text-line
  orientation classifier applied before recognition (disabled by default).
- `--ocr-device`: e.g. `gpu:0` or `cpu`; by default PaddleOCR uses GPU 0 if its build supports it.
- `--ocr-batch-size`: default 1. PaddleOCR pads all crops of a batch to the widest one,
  so larger batches can change individual results (observed for 4 of 42 test crops).

## Annotation format

The PaddleOCR / PPOCRLabel `Label.txt` format, one image per line:

```text
<image name>\t[{"transcription": "手機報在线", "points": [[1179, 551], [1652, 544], [1654, 666], [1181, 673]], "difficult": false}, ...]
```

Boxes marked `"difficult": true` (unless `--include-difficult`) and the "don't care"
transcriptions `###` and `*` are not evaluated. Polygons need at least four points.

## Metric definitions

**FR-IQA** (pyiqa, averaged over images). PSNR and SSIM are computed on the Y channel of
YCbCr (ITU-R BT.601, as in BasicSR), set explicitly because pyiqa's own Y-channel default
is YIQ. LPIPS uses AlexNet (v0.1). All four run at full resolution; a ~12 MP pair needs
roughly 15 GB for DISTS or SSIM, so use `--device cpu` if the GPU runs out of memory.

**OCR.** Each box is cropped the way PPOCRLabel exports recognition crops: a quadrilateral
is perspective-rectified in annotation order (made clockwise if drawn counter-clockwise),
other polygons use their minimum-area rectangle, and crops at least 1.5x taller than wide
are rotated 90 degrees counter-clockwise. The crop is recognized with PaddleOCR.
Recognized text and transcription are both NFKC-normalized (full-width to half-width
characters), stripped of all whitespace, and lowercased with `--ignore-case`.

For text boxes *i* with normalized label *y_i*, recognition *p_i* and Levenshtein
distance *d_i* = ED(*p_i*, *y_i*):

- OCR-A = mean over boxes of [*p_i* == *y_i*] (exact-match accuracy per text line)
- CER = sum *d_i* / sum |*y_i*| (character error rate; can exceed 1)
- 1-NED = 1 - mean over boxes of *d_i* / max(|*p_i*|, |*y_i*|)

The summary aggregates over all evaluated text boxes of all images; `per_image.csv`
reports the same metrics per image. OCR-A and 1-NED follow PaddleOCR's `RecMetric`
(`acc` and `norm_edit_dis`).

## Outputs

The summary is printed; with `--output DIR` the script also writes:

- `summary.json`: averaged metrics, text-box counts and the command-line settings.
- `per_image.csv`: FR-IQA per image, the number of evaluated boxes, and
  `pred_/lq_/gt_` OCR-A, CER and 1-NED per image.
- `ocr_details.jsonl`: one line per text box with its transcription and, for each
  image, the recognized text, confidence and edit distance.

Images are read with PIL like the inference scripts (EXIF orientation is not applied),
so annotations must use the stored pixel orientation.

## Validation

```bash
python -m pytest -q tests/test_evaluate.py
```

The tests stub PaddleOCR and pyiqa, so no weights are needed. They cover annotation
parsing and errors, name matching, text normalization, edit distance and the metric
formulas, crop geometry (box order, vertical text, polygons, degenerate boxes), box
scaling, size checks and the output files. The script was additionally run end to end
with pyiqa 0.1.15 and PaddleOCR 3.7 (PP-OCRv5_server_rec, CPU) on synthetic Chinese/English
text images. Y-channel PSNR matched a BasicSR-style implementation exactly and SSIM to
within 1e-4; edit distance and 1-NED matched rapidfuzz, which PaddleOCR's `RecMetric` uses.
