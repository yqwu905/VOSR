# 文字超分数据增强与清洗

依据 [数据构造](https://chatgpt.com/space/page_6abcb03b0d108191b0cac50aab487148)，实现教师超分、1K 归一化和文字中心裁剪、正式 OCR、低置信过滤、SR 幻觉校验、IQA 六步流程。多源输入由逐行 JSONL 汇聚，`source` 可标记 magazine、anyword3m、easytext 或其他来源；不假定这些数据集已有下载或固定标注格式。

## 快速验证

在仓库根目录运行。轻量流程只依赖 Python ≥3.9、Pillow ≥10.1、NumPy，仓库已有这些依赖。

```bash
python -m data_pipeline demo --output /tmp/text-sr-demo
python -m data_pipeline run \
  --input /tmp/text-sr-demo/input.jsonl \
  --config /tmp/text-sr-demo/config.json \
  --output /tmp/text-sr-demo/output --resume
python -m pytest tests/test_data_pipeline.py -q
```

演示使用双三次插值、标注回放和固定测试分数，只验证流程。演示后端强制写入 `review.jsonl`，`eligible_for_training=false`；HQ 训练列表和训练配对列表为空。复核图片与 LR 演示变体可查看，但不能作为模型效果或数据真实性的证据。

## 生产配置和模型

复制 `configs/data_pipeline/production.json`，用校准数据填写 OCR 阈值、可靠原图 OCR 阈值和 clarity / naturalness / artifacts 三个维度的 IQA 阈值。模板保留 `null` 并在启动时拒绝运行，避免将演示分数口径用在真实数据上。每个 IQA 维度明确指定 `higher` 或 `lower`，分数不要求处于 0–1。

默认从 `hr_path` 读取预生成伪 GT，要求记录单个教师的 `model` 和 `checkpoint`。教师只能选 TADiSR 或 VOSR2；不会串联或投票。输入有镜像子目录时，HR 根目录需要保留相同目录结构，扩展名为 `.png`。

```bash
# 对单层来源目录使用已有 VOSR2 批量推理入口生成伪 GT。
python inference_vosr_onestep.py -c preset/ckpts/VOSR2 \
  -i /data/magazine -o /data/magazine_hr -u 4

python -m data_pipeline index --root /data/magazine --source magazine \
  --hr-root /data/magazine_hr --teacher vosr2 \
  --teacher-checkpoint /models/VOSR2 --output /data/magazine.jsonl
```

也可选择常驻 VOSR2 后端，整个运行只初始化一次模型，调用仓库已有的一步采样、VAE、DINO 和分块推理函数。在仓库根目录运行，需要完整 VOSR2 权重、Qwen VAE、DINO 缓存和对应运行依赖；模型下载与依赖准备遵循仓库原有说明。

```json
{
  "type": "vosr2",
  "kwargs": {
    "checkpoint": "/models/VOSR2",
    "upscale": 4,
    "device": "cuda",
    "overrides": {"tile_size": 512, "vae_tile_size": 1024}
  },
  "fingerprint_files": ["/models/VOSR2/args.json", "/models/VOSR2/model.safetensors"]
}
```

`fingerprint_files` 可用于任何后端，记录本地权重、配置或其他外部依赖的文件 SHA256，恢复运行时检查变化。在线模型和自定义后端的版本也应在配置中明确固定。当前环境没有教师权重，因此 VOSR2 适配器尚未完成真实模型推理验证。

OCR 内置 Tesseract TSV 适配器，将单词合并为完整文本行，置信度取行内最低值。系统需有 `tesseract` 和所选语言包；中文例如 `language: "chi_sim+eng"`。当前环境只有 `eng` / `osd` 语言包。它不会因置信度高而自动把原图 OCR 标为“可靠”。

IQA 可读取每个裁剪的离线分数，也可使用仓库已有 PyIQA：

```json
{
  "type": "pyiqa",
  "kwargs": {
    "device": "cuda",
    "metrics": {
      "clarity": "填写经校准的模型名称",
      "naturalness": "填写经校准的模型名称",
      "artifacts": "填写经校准的模型名称"
    },
    "metric_options": {}
  }
}
```

这里不指定 IQA 模型，也不把任意通用质量分数当作单个维度的已验证测量。应选择无参考指标，确认输入要求和分数方向，并通过业务标注校准。

## 输入与坐标

每行是一个对象。图片相对路径相对于 JSONL 所在目录解析；后端的模型路径相对于进程工作目录解析。图片不自动 EXIF 旋转；标注必须采用存储像素的坐标，或在入库前同步旋转图片与标注。多帧图像、坏图、越界框、非有限分数等进入错误清单。

```json
{
  "id": "issue01/page01.jpg",
  "source": "magazine",
  "group_id": "magazine/issue01",
  "image_path": "/data/magazine/page01.jpg",
  "hr_path": "/data/magazine_hr/page01.png",
  "teacher": {"model": "vosr2", "checkpoint": "/models/VOSR2"},
  "annotations": [
    {"bbox": [10, 20, 180, 50], "text": "可信文本行", "trusted": true}
  ]
}
```

`bbox` 是 `[x_min, y_min, x_max, y_max]`，使用连续像素边界坐标，也支持 `polygon: [[x,y], ...]`。定位框可以没有文本。可信参考需要 `trusted: true` 和文本；文本比较保留大小写和标点，仅做 NFC 与空白规范化。OCR 与可信标注最好采用一致的文本行粒度；无法可靠匹配时进入复核。裁剪优先使用已有标注，同时纳入原图前置检测框，正式 OCR 在裁剪后的 HR 上重新执行。

`long_edge: 1024` 明确采用长边 1K、保留宽高比，例如 2000×500 → 1024×256，允许统一时放大较小图片。不会强制输出 1024×1024。裁剪初始尺寸可配置，按文字框中心定位并保留上下文，遇到边缘相交文字框会递归扩展，保留完整字形和文本行；长行或密集文字可能产生大于初始尺寸的裁剪。每个原图最多输出 `max_crops` 个去重裁剪。

正式 OCR 保留框、文本和置信度，任一文字区域低于阈值时拒绝整张裁剪。原图可靠 OCR 作为参考时，需要显式 `reliable: true` 且超过原图阈值；Tesseract 本身不产生这个可靠性标签。

离线 OCR 后端 `sidecar_ocr` 的输入为：

```json
{
  "ocr": {
    "original": {
      "size": [320, 180],
      "regions": [{"bbox": [10,20,180,50], "text": "文本", "confidence": 0.98, "reliable": true}]
    },
    "final": {
      "crop_0000": {
        "size": [512, 256],
        "image_sha256": "处理后裁剪的像素摘要",
        "regions": [{"bbox": [10,20,180,50], "text": "文本", "confidence": 0.98}]
      }
    }
  }
}
```

原图框使用原图坐标，`final` 框使用该裁剪的局部 HR 坐标。`size` 必须匹配实际处理图像；建议提供像素摘要以绑定离线结果。正式 OCR artifact 缺失是输入错误；原图 OCR 可缺失，但仍需定位标注或检测结果。离线 IQA 对应每个裁剪：

```json
{
  "iqa": {
    "crop_0000": {
      "size": [512,256],
      "image_sha256": "处理后裁剪的像素摘要",
      "scores": {"clarity": 0.9, "naturalness": 0.8, "artifacts": 0.1}
    }
  }
}
```

上述数值只展示格式。缺少 IQA 维度或整体缺少离线 IQA 会进入复核；已提供但尺寸、摘要或分数无效时进入错误清单。

## 幻觉校验与复核

文字参考核对和字形校验分别记录。字符与可信文本冲突、可靠原图内容冲突、缺少参考、漏检参考文字、缺少字形校验都进入复核。原图无法辨认时，不强制超分前后 OCR 完全一致；OCR 高置信、前后一致和 IQA 高分都不能单独使样本通过。

默认 `human` 字形校验读取人工结果，只有明确核对原图与 HR 的字符、笔画和结构后，才应提供 `verified`。它要求审核人、证据、HR 摘要和原图裁剪摘要，避免将旧审核结果用于新图片。

```json
{
  "glyph_checks": {
    "crop_0000": {
      "size": [512,256],
      "image_sha256": "review.jsonl 中的 crop_sha256",
      "original_crop_sha256": "review.jsonl 中的 original_crop_sha256",
      "status": "verified",
      "reviewer": "审核人员标识",
      "evidence": "逐字核对原图和 HR，未发现伪笔画或结构变化"
    }
  }
}
```

状态还支持 `conflict`、`unverifiable`。审核人员不能辨认原图、又缺少其他可信依据时，应保留 `unverifiable`。人工结果是一项输入证据，pipeline 不会替人工判断本身作真实性保证。

没有审核或 IQA 结果时可先运行，查看 `review.jsonl` 中的 HR 和原图裁剪路径，取得实际尺寸与摘要。补齐审核或离线分数后生成新的输入 JSONL，并选择新的输出目录运行。`--resume` 用于恢复相同输入、配置和代码的任务，不用于修改既有审核结果。

TADiSR、其他 OCR 或自动字形验证器通过 `module:Class` 接入；接口定义在 `data_pipeline/backends.py`：

```python
class CustomTeacher:
    demo = False

    def __init__(self, checkpoint):
        ...  # 一次性加载模型

    def restore(self, image, context):
        ...  # 返回 RGB PIL 图像，保持像素对齐和宽高比

# "teacher": {"type": "my_backends:CustomTeacher", "kwargs": {"checkpoint": "..."}}
```

OCR 的 `recognize(image, context)` 返回区域列表；IQA 的 `score(image, context)` 返回维度分数；字形验证器的 `check(original_crop, hr_crop, context)` 返回 `{status, evidence}`。`context` 提供输入记录、当前阶段、裁剪编号、变换后的参考框与可复现的教师 seed。

## 运行与产出

```bash
python -m data_pipeline run --input /data/all_sources.jsonl \
  --config /data/calibrated_config.json --output /data/text_sr_v1
```

- `full_hr/`：长边 1K 的教师伪 GT；可用 `save_full_hr: false` 关闭。完整图并未通过每个裁剪的独立筛选，不直接进入训练列表。
- `hr/`、`original_crops/`：通过或待复核的文字裁剪及原图对照。
- `accepted.jsonl`、`review.jsonl`、`rejected.jsonl`、`errors.jsonl`：筛选状态、原因、OCR、IQA 标签和审核记录。
- `lr/`：通过样本的可复现退化 LR，记录模糊、降采样、255 像素尺度噪声和 JPEG 参数；HR 不变，LR 框同步按实际输出尺寸变换。复核变体默认不生成。
- `pairs_train.jsonl`、`pairs_validation.jsonl`：只有通过样本的 HR/LR 配对。
- `train_hq.txt`、`validation_hq.txt` 及对应 `*_dataset.txt`：VOSR 现有 TXT 加载器可使用的 HQ 列表。
- `run.json`、`summary.json`、`state.sqlite3`：运行配置及摘要、按来源计数、逐原图事务状态。

输出文件采用摘要命名，避免不同来源或目录的重名覆盖。原图以解码 RGB 像素精确去重，SQLite 保存状态，不把百万级 JSONL 全部加载到内存。近重复清理不在当前实现范围；同一物理页面、视频或其他相关内容建议提供共享的全局 `group_id`，保证其裁剪与增强版本落入同一数据划分。未提供分组时以原图像素摘要分组。

恢复时核对输入 JSONL、配置、实现代码、显式外部依赖摘要，以及每个源图和预生成 HR 的内容；成功的样本不会重复导出，丢失或损坏的裁剪/变体会重新生成。单写者文件锁避免并发写入相同输出目录。坏图和单样本模型错误不会阻止其他样本完成；最终退出码 `2` 表示数据错误，`1` 表示启动或完整性错误，`0` 表示运行完成且没有数据错误。不同任务使用独立输出目录。

VOSR 原有训练加载器会再次随机裁剪 HQ，并在 GPU 端执行在线退化；离线 LR 配对供配对训练或检查，不会自动替换原有退化逻辑。如果训练也要求严格保留完整文本行，需使用 manifest 的框约束训练裁剪，不能假定原有随机裁剪具有这个保证。
