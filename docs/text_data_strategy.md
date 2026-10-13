# 文字图像超分数据方案：来源、清洗与增强

本文给出 VOSR 文字超分训练数据的完整处理方案，并记录支撑每条规则的实测证据。它是
[数据增强与清洗 pipeline](data_pipeline.md) 的上游：先按来源收集候选图，用无模型探针清洗并分流，
再进入已有的教师伪 GT / OCR / IQA / 字形六阶段流程，最后用文字感知的退化 profile 生成 LR。

## 结论速览

- **来源**：核实 32 个 HF 文字图像仓库，按 8 类选 27 个配置 + ScreenSR，各抽 24 张**原始字节**共 672 张做实测。
  HF viewer 返回的图片一律是 JPEG 重编码（FUNSD 的 PNG 被转成 JPEG，最大误差 46/255），HR 必须从 parquet 原始字节取。
- **主要数据问题**：AnyWord 镜像统一缩放补边到 512²，OCR_* 子集大量 null 标注（COCO-Text 子集 23/24），LAION/Wukong 是
  PP-OCR 机器标注；PNG 不等于无损（Wukong 96% 内藏 JPEG）；小字普遍（FUNSD 92%、ChartQA 79%、OCR-VQA 62%、COCO-Text 50%
  的图行高 <16 px）；FUNSD 框是 0–1000 归一化坐标；SynthDoG 合成时已内置模糊与 q75 压缩；CORD 有隐私打码模糊块；
  Rosetta OCR 的 `ocr_info` 与像素完全一致率仅 7–42%。
- **清洗**：新增无模型探针（JPEG 历史、上采样倍数、文字区边缘宽度/噪声、半色调、边框、透明通道、近重复）并逐项标定，
  按"去除该退化所需的最小面积缩小倍数"分流为 direct / downscale / teacher / reject。调查样本中 55% 无需生成式教师即可得到 HR。
  JPEG 历史检测在 q30–q90 上质量误差 ≤1、178 张干净图零误报。
- **增强**：6 个场景 profile（camera、screenshot、web_chain、scan、screen_photo、realesrgan），严格像素对齐、可复现、强度可控，
  接入六阶段 pipeline 的 `augmentation.profiles`。测试中发现并修正了 5 处不真实的退化（LR 分辨率双线性去马赛克、
  逐通道彩噪、固定阈值二值化、点采样画出屏幕栅格、翻拍的强色偏与只压暗的渐晕）。
- **验证**：新增 76 项单元测试（数据相关测试共 129 项全部通过）；OCR 实验证实强度单调控制难度、HR 16 px 是可读性下限、
  中文需要更大字高；ScreenSR 实验显示真实翻拍 LR 更接近 ×3 而非 ×4，推荐训练尺度 [2, 3, 4]。

## 1. 目标与约束

文字超分对 HR 的要求比自然图更苛刻：模型会逐笔学习 HR 里的笔画，HR 里的任何缺陷都会被当成"正确答案"。

1. **HR 必须真清晰**：不能是上采样得来的、不能有 JPEG 块和振铃、不能有明显噪声或半色调网点。
2. **文字必须够大**：×4 退化后 LR 中的字已完全不可辨时，模型只能凭空"编"字，这类样本教的是幻觉。
3. **字形必须真实**：生成式教师（VOSR2/TADiSR）的伪 GT、生成模型合成的图都可能改字，需要独立核验。
4. **LR 要像真实输入**：截图缩放、网络转发压缩、打印扫描、翻拍屏幕、手机拍摄各有特有伪影，通用的
   Real-ESRGAN 链只覆盖其中一部分。
5. **可追溯、无泄漏**：每张图记录来源、版本、许可与抽样位置；训练集与基准（ScreenSR、Real-CE 等）去重。

## 2. 数据来源调研

### 2.1 方法

- 由一个调研子任务核实了 32 个 HF 数据集仓库的可用性（25 个 viewer 可用且带图像），从中选出 27 个配置覆盖
  场景文字、网络/电商图、文档、原生数字（截图/幻灯片/论文/封面/图表）、合成、生成式、手写、词裁剪八类，
  外加本仓库的 ScreenSR 真实成对基准。
- **抽样读取 parquet 原始字节**（`python -m data_pipeline sample-source`）：每源随机 shard、row group 和行，
  每源 24 张、种子 42，记录 parquet 转换 commit、row group、行号与字节 SHA256，保留原容器格式。
- 原因是实测发现 **HF datasets-server viewer 返回的所有图片都是 JPEG 重编码**：FUNSD 原始为 PNG，
  viewer 返回 JPEG，像素最大误差 46/255；EasyText 原始 362 KB JPEG 被重编码为 169 KB，平均误差 2.27/255。
  仓库已有的 `sample` 命令走 viewer 路径（`asset_transport: hf_viewer_rgb`），只适合预览，
  不应作为 HR 来源；需要 HR 时用 `sample-source` 或 `sample-parquet`。
- 每张图跑一遍无模型清洗探针（第 3 节）；没有文字框的来源用 RapidOCR（PP-OCRv4 检测，CPU）补框，
  有转写的来源再用 RapidOCR 识别标注区域，测量标注与像素的一致性。

```bash
python -m data_pipeline sources                       # 目录：27 个来源及类别
python -m data_pipeline sample-source --source anyword_lsvt --count 24 --output /data/survey
python -m data_pipeline probe --input /data/survey/anyword_lsvt.jsonl \
  --output /data/survey_probe/anyword_lsvt --detector rapidocr
```

### 2.2 来源总表（实测）

每源 24 张（ScreenSR 取前 24 张 HR），672 张。"含 JPEG"含容器为 JPEG 或像素中检出 JPEG 历史，括号内是有效质量（容器质量与像素历史中较差者，用于识别二次压缩）的中位数；
"文字边缘宽度"与"亮度噪声σ"优先在文字框掩膜内测量；无文字框的来源用 RapidOCR 检测补框（398 张）；
"大边框"指恒色边框超过 30% 面积；"行高"为每张图文字行高中位数的中位数。

| 来源 | 类别 | 短边/长边中位 | 尺寸种类 | 格式 | 含JPEG(中位q) | PNG内藏JPEG | 上采样 | 文字边缘宽度 | 亮度噪声σ | 半色调 | 大边框 | 检出文字 | 行高中位 | 行高<16px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `anyword_lsvt` | 场景 | 512/512 | 1 | png | 0% (–) | 0% | 0% | 2.00 | 1.47 | 0% | 38% | 100% | 30 | 4% |
| `anyword_rects` | 场景 | 512/512 | 1 | png | 0% (–) | 0% | 0% | 2.01 | 0.79 | 0% | 88% | 100% | 61 | 0% |
| `anyword_mtwi` | 场景 | 512/512 | 1 | png | 0% (–) | 0% | 4% | 1.92 | 0.71 | 0% | 0% | 100% | 38 | 8% |
| `anyword_cocotext` | 场景 | 512/512 | 1 | png | 0% (–) | 0% | 0% | 2.26 | 2.01 | 0% | 50% | 83% | 15 | 50% |
| `anyword_mlt` | 场景 | 512/512 | 1 | png | 0% (–) | 0% | 0% | 1.92 | 0.80 | 0% | 42% | 96% | 39 | 8% |
| `cocotext` | 场景 | 427/640 | 18 | jpeg | 100% (96) | 0% | 0% | 2.17 | 2.45 | 0% | 0% | 100% | 16 | 50% |
| `icdar2015` | 场景 | 720/1280 | 1 | jpeg | 100% (75) | 0% | 0% | 2.28 | 0.30 | 4% | 0% | 100% | 37 | 0% |
| `totaltext` | 场景 | 500/776 | 24 | jpeg | 100% (75) | 0% | 8% | 2.19 | 0.71 | 0% | 0% | 88% | 31 | 4% |
| `textvqa` | 场景 | 737/1024 | 13 | jpeg | 100% (96) | 0% | 12% | 2.16 | 0.91 | 0% | 0% | 96% | 33 | 0% |
| `textatlas_scenes` | 场景 | 418/694 | 23 | jpeg/png | 92% (76) | 4% | 0% | 1.84 | 0.55 | 4% | 8% | 100% | 18 | 25% |
| `anyword_laion` | 网络/电商 | 512/512 | 1 | png | 12% (85) | 12% | 12% | 2.20 | 0.26 | 0% | 8% | 100% | 57 | 0% |
| `anyword_wukong` | 网络/电商 | 512/512 | 1 | png | 96% (74) | 96% | 4% | 2.04 | 0.38 | 0% | 21% | 100% | 37 | 0% |
| `cord` | 文档 | 576/864 | 9 | png | 8% (89) | 8% | 0% | 2.19 | 2.23 | 0% | 0% | 100% | 26 | 4% |
| `sroie` | 文档 | 744/1523 | 20 | jpeg | 100% (84) | 0% | 4% | 1.81 | 0.19 | 0% | 4% | 100% | 29 | 4% |
| `funsd` | 文档 | 762/1000 | 10 | png | 0% (–) | 0% | 0% | 1.54 | 0.05 | 0% | 0% | 100% | 14 | 92% |
| `docvqa` | 文档 | 1702/2212 | 24 | png | 0% (–) | 0% | 4% | 1.83 | 0.22 | 12% | 4% | 100% | 33 | 0% |
| `websight` | 原生数字 | 1440/2560 | 5 | jpeg | 100% (75) | 0% | 0% | 1.68 | 0.00 | 0% | 54% | 100% | 39 | 0% |
| `textatlas_ppt` | 原生数字 | 2700/3600 | 7 | jpeg | 100% (95) | 0% | 0% | 1.69 | 0.00 | 0% | 12% | 100% | 95 | 0% |
| `textatlas_paper` | 原生数字 | 3060/3960 | 3 | png | 0% (–) | 0% | 4% | 1.50 | 0.00 | 0% | 8% | 100% | 63 | 0% |
| `textatlas_cover` | 原生数字 | 348/500 | 21 | jpeg | 100% (85) | 0% | 4% | 1.84 | 1.20 | 0% | 0% | 100% | 16 | 46% |
| `ocrvqa` | 原生数字 | 333/500 | 20 | jpeg | 100% (85) | 0% | 0% | 1.83 | 0.81 | 4% | 0% | 100% | 14 | 62% |
| `chartqa` | 原生数字 | 557/800 | 5 | png | 0% (–) | 0% | 0% | 1.59 | 0.02 | 0% | 0% | 100% | 14 | 79% |
| `synthdog_zh` | 合成 | 858/1109 | 24 | jpeg | 100% (75) | 0% | 50% | 2.62 | 0.17 | 0% | 0% | 100% | 34 | 0% |
| `synthdog_en` | 合成 | 879/1311 | 24 | jpeg | 100% (75) | 0% | 46% | 2.58 | 0.07 | 8% | 0% | 100% | 36 | 0% |
| `easytext` | 生成式 | 1024/1024 | 1 | jpeg | 100% (95) | 0% | 17% | 2.13 | 0.40 | 0% | 4% | 100% | 87 | 4% |
| `iam_line` | 手写 | 128/1670 | 24 | jpeg | 100% (75) | 0% | 8% | 2.40 | 0.37 | 54% | 0% | 100% | 91 | 0% |
| `mjsynth` | 词裁剪 | 31/103 | 21 | jpeg | 100% (78) | 0% | 0% | 2.19 | 0.40 | 0% | 0% | 42% | 14 | 25% |
| `screensr_hr` | 成对SR | 512/512 | 1 | png | 0% (–) | 0% | 0% | 2.11 | 1.22 | 0% | 4% | 50% | 19 | 4% |

### 2.3 各类来源的问题

以下结论结合了上表的测量与逐图目视检查（每源前 3 张叠框缩略图）。

**场景文字**（AnyWord OCR_*、COCO-Text、ICDAR2015、Total-Text、TextVQA、TextAtlas TextScenesHQ）
- AnyWord 镜像把所有图缩放并补边到 512×512 PNG（尺寸种类为 1）：ReCTS 88%、COCO-Text 50%、MLT 42%、LSVT 38% 的图
  有超过 30% 面积的恒色白边/黑边，原始分辨率已丢失，PNG 只是容器。需要高分辨率时应回到各原始数据集。
- AnyWord OCR_* 子集大量行的 `annotations` 是 null 而非空列表，仓库原有的 `convert_annotations` 遇到会崩溃
  （默认 `sample` 就抽 OCR_LSVT），本次已修复并记录为 `annotations_null`。
- 字太小：两个 COCO-Text 版本行高中位约 15–16 px，一半图像行高不足 16 px；COCO 原图宽度只有 640 px。
- 噪声与压缩：COCO-Text 文字区噪声 σ 中位 2.45；ICDAR2015（Google Glass 抓拍）与 Total-Text 全部是 q75 JPEG，
  ICDAR2015 文字边缘宽度 2.28，只能缩小清洗或交给教师；TextVQA 是 OpenImages 缩到长边 1024 的 q96 JPEG，12% 测出等效上采样。

**网络/电商图**（AnyWord LAION、Wukong）
- PNG 不等于无损：Wukong 96% 的 PNG 内藏 JPEG 量化痕迹（中位 q74），LAION 12%（q85）。这正是 `jpeg_history` 的用途。
- 标注是机器 OCR；原始标注里 LAION 有 27 条、Wukong 有 42 条被数据集标为无效或越界，被适配器记录而不是悄悄裁剪。
- 电商图常有后期叠加的促销大字和大量未标注的小字，商品图常带白底补边；水印关键词规则需在全量数据上启用。

**文档**（CORD、SROIE、FUNSD、DocVQA）
- FUNSD 的 `bboxes` 是 0–1000 归一化坐标（已用叠框核对），直接当像素会整体错位；且 92% 的图行高不足 16 px。
- CORD 是手机拍摄的小票：文字区噪声 σ 中位 2.23，71% 超过阈值；小票上的隐私信息被打码成模糊块，
  这些人为模糊若进入 HR，会教模型"生成模糊"——裁剪必须只取有标注的区域。
- SROIE 为热敏纸扫描，存在褪色低对比；容器多为 q100，但 24 张中 15 张的像素里检出更早的 q62–q86 压缩（二次压缩，q100
  重存并不能去掉先前的块效应），分流规则因此取容器与历史质量中较差者。EasyText 同样有 11 张 q95 文件内藏 q74–q85 的历史。
- DocVQA 为 1700×2200 灰度扫描，12% 检出周期纹理（网点/扫描网纹）；一题一行导致同图重复，抽样时已按 docId 去重，
  仍检出 1 对近重复。

**原生数字**（WebSight、TextAtlas PPT/Paper/Cover、OCR-VQA、ChartQA）
- 字形最干净的一类：Paper2Text（3060×3960 PNG，边缘宽度 1.50，零噪声）24 张中 23 张直接可用；PPT2Structured（JPEG q95）全部直接可用。
- WebSight 是 2560×1440 的 q75 JPEG（约 134 KB），文字清晰但被压缩，全部需要 1.5× 缩小清洗；54% 含大面积留白或占位块。
- 封面类分辨率低（短边约 340 px），OCR-VQA 62%、CoverBook 46% 行高不足 16 px；CoverBook 实际多为带相机日期戳和反光的翻拍照片，并非原生数字。
- ChartQA 图很小，79% 行高不足 16 px，基本不可用作 HR。

**合成与生成式**
- SynthDoG 在渲染时已内置模糊、透视和 q75 JPEG：边缘宽度 2.6、约一半测出等效上采样。合成不等于干净，不能直接当 HR；
  需要合成 HR 时应自建渲染并关闭后处理。
- EasyText 是生成模型输出（1024² JPEG q95），17% 测出等效上采样；字形可能是模型"编"出来的，`text` 字段是含占位符的描述
  而非转写，只能作为风格/背景多样性来源，不能作为字形真值。

**手写与词裁剪**：IAM-line 是 128 px 高的行裁剪（长宽比超过 8，q75 JPEG，54% 有周期纹理），MJSynth 是 31 px 高的词图，
都被尺寸规则拒绝——它们服务于识别模型，不适合做 SR 的 HR。

**成对 SR（ScreenSR）**：512² 裁剪中一半不含可检测文字；HR 是照片类内容，铜版画排线、植被、毛发会抬高噪声估计
（按 50% 平坦像素口径测得 σ 6–14），因此噪声探针改为只用最平坦 10% 像素并优先在文字区测量。与本次 27 个来源的
672 张抽样无近重复（pHash ≤6 bit）；正式训练前需对全量训练数据做同样的基准排除。

**通用**：大量 HF 镜像没有许可标签（`python -m data_pipeline sources --json` 列出了每个来源记录的许可信息），每条抽样记录都保留来源仓库、parquet 版本与许可字段，入库前需人工核对上游条款。

### 2.4 未在 HF 上直接抽样、但值得获取的来源

| 来源 | 价值 | 已知问题 / 处理 |
|---|---|---|
| [Real-CE](https://github.com/mjq11302010044/Real-CE)（ICCV 2023） | 中英真实 LR/HR 成对（13 mm vs 52 mm 焦距），1,935 训练 / 783 测试对 | [TADiSR](https://arxiv.org/abs/2506.04641) 报告需人工剔除错位对后仅剩 337 训练对；规模小，只宜做验证/微调，测试集必须加入去重排除表 |
| TextZoom | 英文真实成对文字行 | 只有英文、字符结构简单（[Real-CE 论文](https://arxiv.org/abs/2308.03262)），且分辨率很低，不宜做 HR |
| [Real-Texts](https://arxiv.org/abs/2601.17340)（TEXTS-Diff） | 3.4 万张中英真实文字 512² 裁剪 | 作者称将发布；其构建流程（PP-OCRv5 过滤 → 文字框裁剪 → VisualQuality-R1 打分 4.25 阈值 → 人工核验）与本方案一致，可直接接入 |
| LSDIR / DIV2K 等自然图 | 提供无字背景与纹理多样性 | TADiSR 用 OCR 排除含字图做背景；本方案在自然图上同样先过探针，再按需与文字数据混合 |
| 自有 PDF / 网页渲染 | 字形 100% 正确、可任意高分辨率、自带精确转写 | 需控制字体与版式多样性；渲染时禁用次像素抗锯齿；许可需逐源确认 |
| 扫描文档 SR（[Zyrek et al. 2025](https://arxiv.org/abs/2506.06953)） | 75–600 DPI 多分辨率真实扫描 | 作者称将公开；可作为 scan profile 的标定参照 |

## 3. 清洗设计

### 3.1 漏斗

```
L0 容器：解码、多帧、位深、透明通道、EXIF 方向、JPEG 量化表        ──┐
L1 像素探针：JPEG 历史、上采样倍数、边缘宽度、噪声、半色调、边框、对比度 │ data_pipeline/probes.py
L2 文字几何：文字框（标注或检测）、行高、覆盖率、水印关键词           ──┘
L3 去重：pHash 近重复（8×8 bit 分带 LSH）、基准测试集排除            ── data_pipeline/cleaning.py
L4 分流：direct / downscale(k) / teacher / reject，并物化清洗后的图像与标注
L5 六阶段：文字中心裁剪 → 正式 OCR → 参考核对 → 字形核验 → IQA     ── data_pipeline/pipeline.py（已有）
L6 退化：文字感知 profile 生成可复现 LR                             ── data_pipeline/degradations.py
```

关键思路是**先测量再决定用不用教师**：已经清晰的原生数字图直接当 HR（恒等教师，零幻觉风险）；
轻度退化的图用面积缩小"洗干净"（缩小 k 倍会平均掉噪声、模糊、上采样和 JPEG 残留，但字也小 k 倍，
所以只在字够大时采用）；其余才交给 VOSR2 生成伪 GT，再由已有的字形核验把关。

### 3.2 探针及其标定

所有探针在原始存储像素上计算（任何预缩放都会掩盖它要测的东西），超过 2048 像素的图取中心裁剪；
文字清晰度只在文字框掩膜内测量。数字均来自本次标定（脚本见第 5 节），测试见 `tests/test_data_cleaning.py`。

| 探针 | 测量对象 | 标定结果 |
|---|---|---|
| `jpeg_quality` | JPEG 文件的亮度量化表 → IJG 等效质量 | 读 Pillow 解析出的自然顺序表（已核对 `JpegImagePlugin` 源码），q30/60/90 精确还原 |
| `jpeg_history` | **PNG 等无损容器里的 JPEG 前科**：在 64 个网格相位上做 8×8 DCT，检查 9 个低频系数是否落在某量化步长的整数倍上；取"格点得分相对相邻步长的尖峰"而非绝对值 | 文字图 q30–q90 全部检出，质量误差 ≤1，裁剪偏移网格正确还原；178 张干净图（48 张渲染 + 130 张 ScreenSR 照片）零误报。q≥95 量化步长为 1，不可检也无害。二值/调色板图灰度级过少会伪造格点，直接跳过 |
| `effective_upscale` | 面积缩小再三次插值放大的能量损失，与 2 倍尺度的损失之比 | 原生 =1；双三次/Lanczos/双线性放大 ×2、×3、×4 分别还原为 2、3、4；σ=1 高斯模糊被视为 ×2（等效分辨率减半，本就应当如此） |
| `edge_width` / `text_edge_width` | 最陡 1% 梯度处"局部对比度/斜率" → 台阶宽度（像素） | 清晰渲染 1.53，σ=0.6/1/2 高斯模糊后 2.06/2.64/4.10，与 2.5σ 的理论值一致；不随背景面积变化；文字区与全图口径一致（渲染图上相差 ≤0.1） |
| `noise_sigma` / `text_noise_sigma` | Immerkær 噪声估计；平坦区在平滑图上选取以免噪声本身影响选择，且只用最平坦的 10% | 测的是亮度噪声：灰度噪声 σ=1/2/4 测得 0.99/1.99/3.98；逐通道 σ=2/5/10 测得 1.37/3.32/6.11，即 σ×0.67（RGB→Y 系数）。用 50% 平坦像素时，铜版画排线、植被、毛发会被测成 σ 6–14，干净 ScreenSR 照片中位 3.27；改为 10% 后中位 0.90 |
| `periodic_peak` | 离轴高频能量中最强峰的占比 | 干净照片 ≤0.020，各角度/周期的半色调 ≥0.047，阈值 0.03 |
| `padding_fraction` | 恒色边框（信箱黑边、方形补白） | 用于裁边修复，不作为拒绝理由 |
| `alpha_fraction` | 透明通道 | Pillow `convert("RGB")` 保留 α=0 处的底色（通常为黑），透明底黑字会整片变黑；分流为 `composite_alpha` 修复 |
| `phash` | 64 bit DCT 感知哈希 | 近重复阈值 6 bit；8 段×8 bit 分带保证 ≤7 bit 的对必然同桶，测试与暴力搜索结果一致 |
| `text_geometry` | 行高（最小外接矩形短边）、覆盖率 | 行高中位数用于"字是否够大"与"能缩小多少" |

### 3.3 分流规则

默认值在 `configs/data_pipeline/cleaning_rules.json`。每种退化换算成"去除它所需的最小面积缩小倍数" k，
取最大值并向上取到阶梯 {1, 1.5, 2, 3, 4}：

| 退化 | 换算 | 标定依据 |
|---|---|---|
| JPEG（容器与历史中较差者） | q≥90→1，75–90→1.5，50–75→2，40–50→3，<40→不可洗 | 文字图上缩小后 JPEG 残差 PSNR ≥40 dB 所需倍数（q90@1×=40.0 dB，q75@1.5×=41.2，q50@2×=39.5，q40@3×=40.6，q30@4×=38.3） |
| 上采样 | k = effective_upscale | 缩回原生分辨率 |
| 模糊 | edge_width > 2.2 时 k = edge_width / 1.6 | 模糊文字缩小后的边缘宽度：w=2.39→1.5× 后 2.07；w=3.3→3× 后 2.11 |
| 噪声 | σ > 1.5 时 k = σ / 1.5（优先用文字区噪声） | 亮度噪声随面积缩小近似按 1/k 下降（6.11→2× 后 3.12→3× 后 1.92） |
| 半色调/摩尔 | periodic_peak > 0.03 时 k = 2 | 网点周期多为 2–6 像素 |

- `direct`：k=1。`downscale`：k ≤ 4 且行高中位数 / k ≥ 16 px。`teacher`：需要的 k 太大或缩小后字太小。
- `reject`：短边 < 256、长宽比 > 8、亮度动态范围 < 24、无文字、行高中位数 < 16 px、水印关键词、近重复、与基准重叠。
- 修复（不拒绝）：透明合成白底、裁掉恒色边框（框必须完整落在保留区域内）、16 位转 8 位。
  EXIF 方向不自动旋转（标注采用存储像素坐标），带 EXIF 旋转的图在物化时拒绝。
- `min_text_height=16` 的依据见 4.4 节实验 A：HR 行高 16 px（×4 后 LR 4 px）时退化后的 OCR 已基本失效。

### 3.4 调查数据上的分流结果

按默认规则（`cleaning_rules.json`）对 672 张调查样本分流，"触发的退化"统计每张图需要缩小的原因（可多选）：

| 来源 | direct | downscale | teacher | reject | 主要拒绝原因 | 触发的退化（需缩小的原因） |
|---|---|---|---|---|---|---|
| `anyword_lsvt` | 10 | 6 | 7 | 1 | text_too_small 1 | noise 12, blur 6 |
| `anyword_rects` | 16 | 7 | 1 | 0 | – | noise 4, blur 4 |
| `anyword_mtwi` | 17 | 3 | 2 | 2 | text_too_small 2 | noise 4, blur 3, upscaled 1 |
| `anyword_cocotext` | 3 | 1 | 4 | 16 | text_too_small 12, no_text 4 | blur 14, noise 13 |
| `anyword_mlt` | 15 | 5 | 1 | 3 | text_too_small 2, no_text 1 | blur 6, noise 3 |
| `cocotext` | 2 | 2 | 8 | 12 | text_too_small 12 | noise 16, blur 11, jpeg 1 |
| `icdar2015` | 0 | 20 | 4 | 0 | – | jpeg 24, blur 16, periodic_pattern 1 |
| `totaltext` | 1 | 10 | 9 | 4 | no_text 3, text_too_small 1 | jpeg 22, blur 12, noise 5, upscaled 2 |
| `textvqa` | 9 | 9 | 5 | 1 | no_text 1 | blur 10, noise 4, upscaled 3, jpeg 3 |
| `textatlas_scenes` | 5 | 6 | 7 | 6 | text_too_small 6 | jpeg 17, noise 7, blur 3, periodic_pattern 1 |
| `anyword_laion` | 9 | 14 | 1 | 0 | – | blur 12, upscaled 3, jpeg 2, noise 1 |
| `anyword_wukong` | 1 | 14 | 9 | 0 | – | jpeg 23, blur 7, noise 2, upscaled 1 |
| `cord` | 2 | 9 | 12 | 1 | text_too_small 1 | noise 17, blur 11, jpeg 1 |
| `sroie` | 6 | 8 | 9 | 1 | text_too_small 1 | jpeg 16, blur 2, upscaled 1, noise 1 |
| `funsd` | 2 | 0 | 0 | 22 | text_too_small 22 | – |
| `docvqa` | 14 | 8 | 1 | 1 | near_duplicate 1 | blur 7, periodic_pattern 3, noise 1, upscaled 1 |
| `websight` | 0 | 24 | 0 | 0 | – | jpeg 24 |
| `textatlas_ppt` | 24 | 0 | 0 | 0 | – | – |
| `textatlas_paper` | 23 | 0 | 0 | 1 | blank_or_low_contrast 1 | upscaled 1 |
| `textatlas_cover` | 0 | 1 | 12 | 11 | text_too_small 11, too_small 3 | jpeg 24, noise 9, blur 3, upscaled 1 |
| `ocrvqa` | 0 | 3 | 6 | 15 | text_too_small 15 | jpeg 24, noise 4, blur 1, periodic_pattern 1 |
| `chartqa` | 5 | 0 | 0 | 19 | text_too_small 19 | – |
| `synthdog_zh` | 0 | 14 | 10 | 0 | – | jpeg 24, blur 17, upscaled 12, noise 1 |
| `synthdog_en` | 0 | 17 | 7 | 0 | – | jpeg 24, blur 16, upscaled 11, periodic_pattern 2 |
| `easytext` | 5 | 16 | 2 | 1 | text_too_small 1 | jpeg 11, blur 9, noise 5, upscaled 4 |
| `iam_line` | 0 | 0 | 0 | 24 | too_small 24, extreme_aspect 24 | jpeg 24, blur 19, periodic_pattern 13, upscaled 2 |
| `mjsynth` | 0 | 0 | 0 | 24 | too_small 24, no_text 14, text_too_small 6 | jpeg 20, blur 12, noise 3 |
| `screensr_hr` | 2 | 3 | 6 | 13 | no_text 12, text_too_small 1 | noise 11, blur 6 |
| **合计** | 171 | 200 | 123 | 178 | | |

direct+downscale 共 371 张（55%）可以不经生成式教师得到 HR；需要教师的 123 张主要来自 SynthDoG（17）、小票照片
CORD（12）、封面翻拍（12）、Total-Text、SROIE 与 Wukong（各 9）和 COCO-Text（8）；被拒绝的 178 张中绝大多数是"字太小"（FUNSD、ChartQA、OCR-VQA、COCO-Text）与尺寸不合格的裁剪类数据。
每源 24 张只能反映量级：比例的 95% 置信区间约为 ±20 个百分点。

### 3.5 标注与像素的一致性

对带转写的来源，把每个标注区域透视校正后用 RapidOCR 识别，与标注文本比较（忽略空白、区分大小写），
每图最多 12 个区域。它测的是"标注 vs 像素"的不一致率上界，包含 OCR 自身的错误：

| 来源 | 标注来源 | 识别区域数 | 完全一致 | 平均 CER | CER>0.5 |
|---|---|---|---|---|---|
| `anyword_lsvt` | 人工/数据集标注 | 37 | 78% | 0.10 | 5% |
| `anyword_rects` | 人工/数据集标注 | 38 | 82% | 0.09 | 5% |
| `anyword_mtwi` | 人工/数据集标注 | 108 | 70% | 0.18 | 16% |
| `anyword_cocotext` | 人工/数据集标注 | 1 | 0% | 1.00 | 100% |
| `anyword_mlt` | 人工/数据集标注 | 20 | 35% | 0.65 | 65% |
| `cocotext` | Rosetta OCR（数据集自带） | 72 | 42% | 0.30 | 26% |
| `anyword_laion` | 人工/数据集标注 | 86 | 85% | 0.05 | 1% |
| `anyword_wukong` | 人工/数据集标注 | 77 | 88% | 0.04 | 3% |
| `cord` | 人工/数据集标注 | 285 | 82% | 0.04 | 1% |
| `sroie` | 人工/数据集标注 | 288 | 54% | 0.26 | 27% |
| `funsd` | 人工/数据集标注 | 288 | 82% | 0.07 | 3% |
| `ocrvqa` | Rosetta OCR（数据集自带） | 263 | 7% | 0.64 | 57% |

- AnyWord 的 LAION/Wukong 标注全部带 `rec_score`，是 PP-OCR 机器标注；与同源的 RapidOCR 一致率高（85–88%）是相关误差，
  不是独立验证，因此这类文本只能作为"定位 + 弱参考"，不能标 `trusted`。
- AnyWord OCR_* 子集的人工标注在中文场景上与像素基本一致（LSVT/ReCTS 一致率约 80%），但 MLT 的韩文、阿拉伯文等
  超出 RapidOCR 的识别范围，0.65 的 CER 主要是识别器能力而非标注错误。AnyWord 的 COCO-Text 子集 24 行中 23 行没有标注。
- COCO-Text（howard-hou 镜像）与 OCR-VQA 的 `ocr_info` 是 Rosetta OCR 结果（小写化、按词切分），完全一致率仅 42% 与 7%，
  不能当作转写真值。
- SROIE 的行级标注存在错字（例如样本中印刷为 "SDN BHD" 的行被标为 "SDN BND"），与像素完全一致的只有 54%。

### 3.6 使用方式

```bash
# 探针 + 去重 + 分流 + 物化；--exclude 传入基准测试集清单，近重复即拒绝
python -m data_pipeline probe --input /data/all_sources.jsonl --output /data/clean_v1 \
  --rules configs/data_pipeline/cleaning_rules.json --detector rapidocr \
  --exclude /data/screensr.jsonl /data/realce_test.jsonl --materialize

# clean.jsonl：direct/downscale 路线，图已修复并缩小，标注同步变换 → 恒等教师
python -m data_pipeline run --input /data/clean_v1/clean.jsonl \
  --config my_clean_route.json --output /data/text_sr_clean_v1
# teacher.jsonl：原始记录 → VOSR2 教师（configs/data_pipeline/dataset_audit.json 或 production.json）
```

`configs/data_pipeline/clean_route.json` 是恒等教师路线的模板：`teacher: identity`、
`glyph: identity_source`（恒等教师没有生成步骤，字形不可能被改，配置校验禁止它与其他教师搭配）、
`upscale_to_long_edge: false`（长边 1024 只做上限，绝不把小图插值放大当 HR）。OCR 与 IQA 阈值同
`production.json` 一样留空，校准前拒绝运行。

## 4. 增强（退化）设计

### 4.1 原则

- **像素对齐**：HR 固定后不做任何几何形变，最终尺寸严格为 round(HR/scale)，文字框按实际缩放矩阵变换。
- **可复现**：一个种子决定场景、每个算子是否启用及全部参数，完整记录在 `degradation.steps`。
- **强度可控**：每个参数写成 `[轻, 重]` 区间，`severity∈[0,1]` 决定取值中心（训练时为 None 即全区间均匀，
  诊断时固定），同一套代码既做训练数据也做难度分析。
- **HR 侧增强不做翻转**：水平/垂直翻转产生镜像字，汉字镜像后不是合法字形；如需 HR 侧增强，
  只做反色（深色模式）、灰度化、色相/通道置换、gamma/对比度。

### 4.2 场景 profile 与算子

| profile | 对应真实场景 | 算子顺序（HR 分辨率 → LR 分辨率） |
|---|---|---|
| `camera` | 手机拍摄的街景/文档 | 光学模糊（高斯/各向异性广义高斯/离焦圆盘/运动）→ 缩放 → 线性光域泊松-高斯传感器噪声（色度噪声只占一部分）→ RGGB 去马赛克 + 假色抑制 → ISP 降噪 → USM 锐化 → JPEG 4:2:0 |
| `screenshot` | 截图、网页图被缩放 | 缩放（含最近邻、无抗混叠的 OpenCV 线性/三次、带抗混叠的 Pillow 三次）→ 调色板量化 ± 抖动 → JPEG（4:2:0 为主）/ WebP |
| `web_chain` | 社交平台多次转发 | 随机缩小 → JPEG → 缩放到 LR → 轻锐化 → 再次 JPEG |
| `scan` | 打印后扫描、传真 | CMY 半色调网点（15/75/0°）→ 洇墨 → 纸张纹理与不均匀 → 扫描模糊 → 缩放 → 灰度噪声 → Otsu 二值化（含漂移与椒盐）→ JPEG |
| `screen_photo` | 翻拍屏幕（ScreenSR 场景） | 轻度 gamma/对比度/白平衡漂移 → RGB 子像素栅格（间距接近 LR 像素、微倾斜）→ 镜头模糊 → 渐晕 → 面积采样缩放（产生彩色摩尔纹）→ 传感器噪声 → 降噪 → 锐化 → JPEG |
| `realesrgan` | 通用二阶退化（与训练端 GPU 版同构的 CPU 实现） | 两轮 [模糊（含 sinc）→ 随机缩放 → 噪声 → JPEG] |

实现细节都经过实测核对，其中几处是在测试中发现问题后修正的：

- RGGB 马赛克对应 OpenCV 的 `COLOR_BayerBG2RGB`（同一平滑图上四种图案代码仅它误差 0.52，其余 23–37）。
- **去马赛克**：在 LR 分辨率上对 1–2 像素宽的笔画做原始双线性去马赛克，会让 8 px 高的文字行几乎不可读
  （OCR CER 0.05→0.68，在马赛克前加光学低通也只降到 0.56）。手机 ISP 会做假色抑制（色度低通）并使用比双线性
  更好的算法，因此算子 = 双线性 + 色度 σ∈[2,1] 低通 + 按强度 α∈[0.3,1] 与原图混合；camera profile 轻强度下
  去马赛克的额外 CER 从 +0.20 降到 +0.02。OpenCV 的边缘感知变体对文字更差（抑制后仍 0.52），VNG 输出整体错位
  （平滑图误差 9.2，文字图 30），都不使用。
- 屏幕栅格用点采样缩放会画出栅格本身而非摩尔纹，因此该 profile 只用面积采样。
- 二值化用固定阈值会把彩色细笔画打断，改为 Otsu 加漂移。
- 逐通道独立的传感器噪声会产生真实手机照片里少见的粉紫色彩噪，因此只有一部分噪声逐通道独立。
- 色调/白平衡漂移保持轻度：初版 screen_photo 在 ScreenSR 首次全量运行中平均亮度偏移为 −6.1（真实 LR 为 −0.2），
  成对数据通常已做颜色对齐、VOSR 推理时也做小波颜色对齐，训练端现有 Real-ESRGAN 配置的 `color_jitter_prob`
  同样为 0，强色偏只会让模型学"改颜色"。

### 4.3 推荐权重

默认混合（`DEFAULT_WEIGHTS`）：camera 0.25、screenshot 0.2、web_chain 0.15、scan 0.15、screen_photo 0.15、realesrgan 0.1。
按来源类别分别运行时建议：

| 来源类别 | camera | screenshot | web_chain | scan | screen_photo | realesrgan |
|---|---|---|---|---|---|---|
| 场景文字、网络图 | 0.4 | 0.05 | 0.2 | 0.05 | 0.15 | 0.15 |
| 文档、小票 | 0.25 | 0.1 | 0.15 | 0.45 | 0.05 | 0 |
| 原生数字（截图、幻灯片、论文、封面） | 0.1 | 0.4 | 0.25 | 0 | 0.25 | 0 |

### 4.4 测试与实验

单元测试 `tests/test_degradations.py`（42 项）覆盖：全部 profile 在奇数尺寸与 ×2/×3/×4 下严格落在 LR 网格；
同种子逐像素可复现；权重限制场景；每个 profile 的 severity 与保真度单调；以及**用清洗探针测量算子注入的退化**
（JPEG 算子 q55 被历史探针测回 55±3、4:2:0 在红字上的色度误差显著高于 4:4:4、高斯噪声 σ=6 被测回 6±0.6、
模糊 σ 越大边缘越宽、非整数倍最近邻缩放保留混叠而面积缩放平均掉、半色调与屏幕栅格留下可测的周期/彩色摩尔纹）。
清洗与退化用同一套度量互相校验。

**实验 A：退化后文字还剩多少可辨信息**（`tools/text_data_experiments.py legibility`）。渲染 64 条带精确转写的中英文文字行
（宋/黑/楷与衬线/无衬线/等宽字体，HR 行高 16/24/32/48 px，随机前景/背景色），每个 profile × 强度 {0.25, 0.5, 0.75} × 2 个种子做 ×4 退化，
双三次放大回原尺寸后用 RapidOCR 识别，统计 CER（共 2432 次识别；所有 HR 的 CER 均为 0）：

| 退化 | 强度 | HR 16px（LR 4px） | HR 24px（LR 6px） | HR 32px（LR 8px） | HR 48px（LR 12px） | 平均 |
|---|---|---|---|---|---|---|
| legacy | 默认 | 0.91 | 0.29 | 0.10 | 0.00 | 0.33 |
| camera | 0.25 | 0.96 | 0.49 | 0.19 | 0.03 | 0.42 |
| camera | 0.5 | 0.96 | 0.62 | 0.28 | 0.04 | 0.48 |
| camera | 0.75 | 0.99 | 0.81 | 0.50 | 0.07 | 0.59 |
| screenshot | 0.25 | 0.93 | 0.63 | 0.32 | 0.04 | 0.48 |
| screenshot | 0.5 | 0.94 | 0.68 | 0.39 | 0.07 | 0.52 |
| screenshot | 0.75 | 0.95 | 0.74 | 0.46 | 0.09 | 0.56 |
| web_chain | 0.25 | 0.97 | 0.61 | 0.25 | 0.05 | 0.47 |
| web_chain | 0.5 | 0.97 | 0.71 | 0.30 | 0.04 | 0.51 |
| web_chain | 0.75 | 0.98 | 0.78 | 0.41 | 0.08 | 0.56 |
| scan | 0.25 | 0.91 | 0.43 | 0.16 | 0.03 | 0.38 |
| scan | 0.5 | 0.95 | 0.47 | 0.19 | 0.01 | 0.40 |
| scan | 0.75 | 0.95 | 0.62 | 0.30 | 0.02 | 0.47 |
| screen_photo | 0.25 | 0.95 | 0.39 | 0.09 | 0.00 | 0.36 |
| screen_photo | 0.5 | 0.97 | 0.60 | 0.18 | 0.02 | 0.44 |
| screen_photo | 0.75 | 0.98 | 0.88 | 0.36 | 0.08 | 0.57 |
| realesrgan | 0.25 | 0.96 | 0.68 | 0.36 | 0.04 | 0.51 |
| realesrgan | 0.5 | 0.97 | 0.84 | 0.56 | 0.13 | 0.62 |
| realesrgan | 0.75 | 0.99 | 0.97 | 0.84 | 0.51 | 0.82 |

- 每个 profile 的平均 CER 都随强度单调上升，强度参数确实控制难度；新 profile 的难度范围（0.36–0.82）覆盖并超出了现有
  legacy 链（0.33），后者在 32 px 时几乎不损失可读性（CER 0.10），难度偏低。
- **字高是硬约束**：HR 16 px（LR 4 px）时所有退化（包括 legacy）CER ≥ 0.91，此时 LR 已不含字形信息，训练只能教模型
  "编字"——这就是清洗规则 `min_text_height=16` 的依据。
- **中文需要更大的字**：跨所有 profile 与强度，24 px 时中文 CER 0.80、英文 0.52；32 px 时 0.49 vs 0.19。建议在训练分布中
  让中文行高主要落在 32 px 以上（×4），16–32 px 的中文样本控制比例。

**实验 B：与真实 LR 的分布距离**（`tools/text_data_experiments.py screensr`）。ScreenSR 是 130 对真实翻拍的 HR/LR
（LR 已放大到 512²）。对每张 HR 用各条件合成 LR 并放大回 512²，提取 10 个特征：5 个径向频带的功率传递（log10 LR/HR，
单位 cycles/px）、低频亮度偏移、对比度比、低频色偏、平坦区噪声、高频色度误差。前 40 对只用于诊断（下述 screen_photo
的调整即来自这里），后 90 对用于报告：以真实特征的均值/标准差标准化，"覆盖度"= 每个真实样本到最近合成样本距离的中位数，
"精度"= 每个合成样本到最近真实样本距离的中位数，均越小越好。

| 条件 | 覆盖度↓ | 精度↓ | T 0.02–0.05 | T 0.05–0.1 | T 0.1–0.18 | T 0.18–0.3 | T 0.3–0.5 | 亮度偏移 | 对比度比 | 色偏 | 噪声σ | 色度细节 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.00 | 0.00 | -0.01 | -0.09 | -0.55 | -1.24 | -2.04 | -0.26 | 0.98 | 0.65 | 0.42 | 2.53 |
| realesrgan_x3 | 3.38 | 5.99 | -0.08 | -0.43 | -1.19 | -1.98 | -2.99 | 0.10 | 0.98 | 1.62 | 0.22 | 2.66 |
| screen_photo_x3 | 3.51 | 7.49 | 0.03 | 0.00 | -0.33 | -1.05 | -2.76 | -0.54 | 0.96 | 1.97 | 0.33 | 2.67 |
| bicubic_x2 | 4.31 | 5.22 | -0.00 | -0.02 | -0.14 | -0.66 | -1.42 | 0.03 | 1.00 | 0.03 | 0.19 | 1.62 |
| mixture_x3 | 4.47 | 5.65 | -0.00 | -0.07 | -0.44 | -1.23 | -2.85 | -2.25 | 0.99 | 1.76 | 0.19 | 2.65 |
| camera | 4.59 | 7.37 | 0.07 | -0.02 | -0.53 | -1.92 | -2.85 | 0.61 | 0.99 | 2.21 | 0.23 | 2.67 |
| realesrgan | 4.62 | 7.46 | -0.09 | -0.57 | -1.42 | -2.51 | -3.02 | 0.12 | 0.98 | 1.92 | 0.19 | 2.65 |
| bicubic_x3 | 4.64 | 5.44 | -0.00 | -0.06 | -0.44 | -1.31 | -2.92 | 0.01 | 1.00 | 0.04 | 0.13 | 2.09 |
| screen_photo | 4.83 | 9.68 | 0.08 | -0.01 | -0.50 | -1.87 | -2.86 | -0.63 | 0.97 | 2.12 | 0.25 | 2.66 |
| scan | 4.92 | 6.38 | -0.01 | -0.20 | -0.83 | -2.14 | -2.87 | -7.87 | 1.11 | 3.09 | 0.15 | 2.76 |
| legacy | 5.00 | 4.34 | -0.03 | -0.21 | -0.98 | -2.28 | -3.00 | 0.08 | 1.00 | 1.03 | 0.15 | 2.64 |
| bicubic_x4 | 5.39 | 5.32 | -0.01 | -0.16 | -0.86 | -2.20 | -3.00 | 0.02 | 1.00 | 0.05 | 0.11 | 2.35 |
| mixture | 5.42 | 6.57 | 0.00 | -0.13 | -0.69 | -2.07 | -2.92 | -2.24 | 1.00 | 2.12 | 0.16 | 2.66 |
| screenshot | 5.49 | 6.39 | 0.01 | -0.02 | -0.45 | -1.91 | -2.90 | 0.16 | 1.00 | 1.57 | 0.12 | 2.63 |
| web_chain | 5.81 | 6.23 | 0.03 | -0.02 | -0.46 | -1.91 | -2.88 | 0.22 | 1.00 | 1.68 | 0.13 | 2.64 |

- **尺度最关键**：真实 LR 的中频衰减（T 0.1–0.18 为 −0.55、T 0.18–0.3 为 −1.24）与 ×3 双三次（−0.44、−1.31）最接近，
  而非 ×4（−0.86、−2.20）。覆盖度最好的三个条件都是 ×3；`clean_route.json` 因此把 `scales` 设为 [2, 3, 4]。
- 同为 ×4 时，camera、realesrgan、screen_photo、scan（4.59–4.92）覆盖度好于 legacy（5.00）。legacy 的"精度"最好，是因为它
  只产生一类温和退化，样本都挤在真实分布附近——新 profile 有意覆盖更宽的难度范围，单一基准上的精度不是目标。
- **screen_photo 的修正**分三步：首次全量运行（130 对，尚未划分）发现初版色偏过强（亮度偏移 −6.1），先把色调漂移调温和；
  第二次运行的汇总（后 90 对）显示亮度偏移仍为 −4.0，原因是渐晕只会压暗，改为保持平均亮度（相机自动曝光）；
  最后只在前 40 对上做算子消融，去掉"色偏 + 屏幕栅格"后覆盖度从 5.8 降到 3.6（×3），说明 ScreenSR 这种已对齐的成对数据几乎没有摩尔纹与色偏。
  野外翻拍确实存在这两种伪影，因此保留并把概率降到 0.2/0.3。留出集上 ×4 覆盖度 5.90→4.83、精度 12.86→9.68。
- 真实 LR 在平坦区的噪声（σ 0.42）与最高频段能量（T 0.3–0.5 为 −2.04）高于所有合成条件：真实翻拍的颗粒/残余噪声仍未被完全
  建模，是下一步需要用独立翻拍数据标定的方向。

### 4.5 与训练的衔接

- 六阶段 pipeline：在配置的 `augmentation` 里写 `profiles`（及可选 `severity`）即切换到文字感知退化；
  不写时保持原有的高斯模糊 + 双三次 + 噪声 + JPEG 链，已有配置与续跑语义不变。
- 已有 HQ 列表（如 `train_hq.txt`）离线生成配对：
  `python -m data_pipeline synthesize --hq-list train_hq.txt --output /data/pairs --variants 2 --weights '{"scan":1,"camera":1}'`。
- 在线训练：`synthesize(hr, seed, scale, profile, weights, severity)` 只依赖 NumPy/OpenCV/Pillow，可在 DataLoader
  worker 中调用；现有训练脚本仍在 GPU 上做 Real-ESRGAN 退化，若要改为在线文字感知退化，需要让数据集返回 LQ
  并在 `prepare_training_batch` 中跳过 GPU 退化（本次未改训练代码）。

## 5. 复现

环境：`uv sync --group data-pipeline` 之外，调查与实验另需 `rapidocr_onnxruntime`（PP-OCRv4 ONNX，CPU）、
`tesseract-ocr`（含 `chi_sim`）与 Noto CJK / DejaVu / Liberation / AR PL UKai 字体。本文数据在 4 核 CPU、无 GPU 的环境中得到。

```bash
# 1. 抽样（每源 24 张原始字节，种子 42）与探针
for s in $(python -m data_pipeline sources | python -c "import json,sys; print(' '.join(json.load(sys.stdin)))"); do
  python -m data_pipeline sample-source --source $s --count 24 --output /data/survey
  python -m data_pipeline probe --input /data/survey/$s.jsonl --output /data/survey_probe/$s --detector rapidocr
done

# 2. 退化实验（tools/text_data_experiments.py；ScreenSR 目录含 HR_512/ 与 LR_512/）
python tools/text_data_experiments.py legibility --output legibility.json
python tools/text_data_experiments.py screensr --screensr /data/ScreenSR --output screensr.json

# 3. 测试（新增 tests/test_data_cleaning.py 与 tests/test_degradations.py）
python -m pytest tests/test_data_pipeline.py tests/test_data_pipeline_datasets.py \
  tests/test_data_cleaning.py tests/test_degradations.py -q
```

探针标定（3.2、3.3 节的数字）由渲染的中英文文字图、ScreenSR HR 照片及其受控退化版本得到；对应的断言写在
`tests/test_data_cleaning.py` 中（JPEG 质量与网格偏移还原、上采样倍数还原、噪声与边缘宽度跟踪、半色调分离等）。

## 6. 局限

- **抽样规模**：每源 24 张只反映量级（比例的 95% 置信区间约 ±20 个百分点），正式使用前应在每个来源上扩大抽样并人工抽检。
- **探针**：阈值在合成文字图和少量真实数据上标定。自然纹理仍会抬高噪声估计（ScreenSR HR 仍有 46% 被判有噪声）；
  JPEG 历史对 q≥95 不可检，压缩后再缩放会抹掉证据；半色调探针排除了坐标轴方向的周期（0°/90° 网屏会漏检）；
  水印检测只做关键词匹配，依赖 OCR 文本。
- **可读性代理**：实验 A 用 RapidOCR 读双三次复原的 LR，反映的是"LR 中还剩多少可辨信息"，不等于 SR 模型的可恢复性；
  RapidOCR 的识别范围也限制了多语种（韩文、阿拉伯文等）的结论。
- **ScreenSR 是评测基准**：前两轮调整看过全量/后 90 对的汇总特征，最后的消融只用前 40 对，报告用后 90 对；这仍接触了测试分布；更严谨的做法是用独立采集的
  翻拍数据标定 screen_photo。真实 LR 的高频细节与平坦区噪声目前没有任何合成条件完全覆盖。
- **训练端未改动**：本次只改数据侧；在线使用文字感知退化需要让数据集返回 LQ，并在 `prepare_training_batch` 中跳过 GPU 退化。
- **未覆盖**：Real-CE、TextZoom、Real-Texts 等不在 HF 上的数据只做了文献调研；自有 PDF/网页渲染管线未实现；许可需人工审核。
