# 中文文档数据：PPT、论文、Word 与 PDF 的来源与采集工程

开放的文字图像数据里中文很少，而且几乎都是渲染好的图片：CDLA、M6Doc、OmniDocBench、CC-OCR、XFUND 等中文文档集
只提供 150–200 DPI、JPEG q75 左右的页面图，许可多为非商用或未声明，只适合做评测和去重。原生数字文档（PDF、Word、PPT）
则不同：矢量页面可以按任意 DPI 渲染成无压缩 HR，文本层直接给出逐行四边形框和转写，字体、版式和场景（公文、财报、论文、
课件、报纸）也比合成数据丰富。本文记录这类文档的来源调研（逐个实测）、合规规则、采集工程
（`data_pipeline/documents`）和用法，以及需要人工决定的事项。

渲染产出的 `clean.jsonl` 是 VOSR manifest，可以直接交给 [文字数据方案](text_data_strategy.md) 的无模型探针
（`python -m data_pipeline probe`）和 [六阶段 pipeline](data_pipeline.md)。

## 结论速览

- **规模**：登记了 25 个来源，都在本环境里实测过（列表、抽样下载并渲染）。中文原生 PDF/Word 的可得量在千万页量级：
  FinePDFs 中文约 340 万份原生数字 PDF、巨潮每年约 63 万份公告、联合国中文文件约 40 万份、DOAJ 中文期刊约 40 万篇、
  ModelScope 年报镜像约 270 万页、台湾行政院公报约 50 万页。
- **PPT 是真正稀缺的**：开放数据集里几乎没有中文 PPTX（Zenodo10K 10,448 份中只有 13 份可用），Common Crawl 每期
  .cn 只有约 16 份。可用路线是定向爬取（高校课件、政府政策解读、上市公司路演原生 PPT、VALSE 报告），加上
  "真实模板版式 + 真实中文文本"的回填合成（`refill.py`）。
- **Word**：司法部 606 部法规 DOCX、深交所投资者关系记录约 5.7 万份原生 DOC/DOCX、政府附件、北京政府采购招标文件、
  联合国 DOC/DOCX，都经 LibreOffice 加固转换后渲染。
- **合规**：自写 RFC 9309 解析器，AI 爬虫退出同样遵守，每一跳重定向都检查；不绕过登录、验证码、WAF 和付费墙；
  按条款分为 7 类，默认只运行 `official` 和 `open`，其余要人工审阅后显式放行。
- **标注**：页面按文字行高（默认 32–64 px）选 DPI 直接从矢量渲染，行四边形和转写来自文本层；乱码、
  轮廓字、扫描页、隐藏文字、缺字、模板占位符都有专门的检测，不可信的行或页不会标成可信。
- **质量**：端到端实测见第 6 节（最终数字在渲染修复完成后更新）。
- **待决**：data.commoncrawl.org 的 robots 冲突、非默认条款类别的法律审阅、联合国文件的版权依据、
  是否申请 API 密钥或批量授权（第 7 节）。

## 1. 来源总表

规模一栏标明"实测"（数过、读过元数据或抽样下载渲染过）或"估计"（给出依据）。条款类别见第 3 节；
只有 `official` 和 `open` 默认运行，其余类别要在人工审阅条款后用 `--accept-terms <类别>` 显式放行。

### 1.1 政府、法律与国际组织文件

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `moj_xzfg` | 司法部国家行政法规库 | DOCX + PDF | 实测 606 部现行行政法规，各一份 DOCX 和 PDF（另有历史版本） | official | 20 份文件渲染 263 页，保留 261 页，全部 text 类 |
| `gov_attachments` | TRS/WebPlus 政府网站附件：北京、上海、科技部、财政部、全国人大报告、吉林自然资源厅政策解读（含 PPTX） | PDF、DOC、DOCX、PPT、PPTX | 抽样 146 个页面中 51 个带附件，共 78 个（PDF 43、DOCX 15、DOC 12、XLS/XLSX 8） | official | 34 份文件保留 301 页；吉林政策解读 PPT 多为整页图片，按扫描页丢弃 |
| `gov_gazette` | 国务院公报 2021 年起各期 PDF | PDF | 每期 13–216 MB | official | 2021 年后大量页面字形转为轮廓（vector 类，无标注）；文字页用方正 BookMaker 字体 |
| `gov_standards` | 生态环境部环境标准全文（强制性 GB 与推荐性 HJ、GB/T） | PDF | — | public_web（推荐性标准受著作权保护） | — |
| `tw_gazette` | 台湾行政院公报 2005 年起各条目及总说明/对照表附件，取自官方开放资料压缩包 | PDF | 估计 7–9 万条（过滤处分、公示送达和名单后），约 50 万页 | official（台湾著作权法第九条；开放资料按政府资料开放授权条款） | 15 份渲染 135 页，保留 117 页，全部 text 类，CJK 中位数 0.75，繁体 |
| `un_documents_zh` | 联合国正式文件中文版（1946–2022），取自 HF 上的 UPRPRC parquet 转储 | PDF、DOC、DOCX | 估计约 19 万份中文 PDF、22–30 万份中文 DOC 类文件 | official（ST/AI/189/Add.9/Rev.2） | 216 份全部下载成功，渲染 1,262 页，保留 1,086 页；每千份中文 PDF 传输 1.8–2.0 GB |

### 1.2 金融披露

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `cninfo_reports` | 巨潮资讯 A 股定期报告与公告 | PDF | 实测 2024 年 632,144 条公告，约 0.5 TB/年（估计） | disclosure | 20 份分层抽样 2,344 页，99.8% 有文本层，私用区/替换符 0.04% |
| `cninfo_ir_office` | 深交所投资者关系活动记录（2012–2020） | DOC、DOCX、PPT、PPTX | 65,680 条记录，约 87% 为原生 Word（约 5.7 万份）；至少 93 份原生 PPT/PPTX 路演材料 | disclosure | 18 份原生 Office 文件全部转换成功，无 VBA 宏 |
| `cninfo_slides` | 业绩说明会、投资者演示文稿 | PDF、PPT、PPTX | A 股每年约 20 份，港股每年 14–16 份（标题检索下限） | disclosure | 15 份 96.8% 为横版页，23% 页面只有图片 |
| `modelscope_finglm` | ModelScope 上的 FinGLM 年报原文（2019–2021） | PDF | 实测 11,588 份，74.4 GB，约 200 万文本页 | disclosure | 63 份每隔两页抽查 3,284 页，全部 text 类 |
| `modelscope_reports_2023` | ModelScope 上的 2023 年定期报告 | PDF | 实测 13,506 份（去掉摘要和英文版后 9,731 份），约 70 万页 | disclosure | 与巨潮重复的文件在存储层按 SHA-256 自动去重 |
| `ccgp_beijing` | 北京市政府采购网公告附件（招标文件、采购需求等），默认跳过合同和供应商材料 | PDF、DOC、DOCX | 列表模式约 2 万条公告、约 1 万个文件、约 30 万页；检索模式回溯到 2010 年，估计 20 万以上文件 | public_web | 48 份全部下载成功，渲染 584 页，保留 534 页，CJK 0.78–0.82 |

### 1.3 学术

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `doaj_zh` | DOAJ 收录的中文期刊（Magtech、仁和、OJS 等平台） | PDF | 331 种期刊、406,883 篇文章；约四分之一给出直接 PDF（另一半链接是勤云的 JS 单页应用） | open_nc（77% CC BY-NC-ND） | 抽样的中文论文全部为原生数字 |
| `ojs_zh` | 中文 OJS 期刊（OAI-PMH） | PDF | ojs.omniscient.sg 84,879 条记录 | open_nc（CC BY-NC 4.0） | CJK 比例 0.87–0.89 |
| `openalex_zh` | OpenAlex 快照里有 PDF 链接的中文开放获取作品 | PDF | 586.6 万中文作品，估计 13–22 万有直接 PDF 链接 | open_nc | 大部分链接指向拒绝爬虫的中科院机构库或已排除的站点 |
| `ntu_theses` | 台大博硕士论文（全球公开授权的部分） | PDF | 每小时约 10 个候选（Crawl-delay 10） | public_web | 新近条目中约 3% 可公开下载 |

学术 PDF 要按页过滤语言：英文摘要和参考文献会把整篇论文的 CJK 比例拉低到 0.28–0.43，所以按页而不是按文件判断。

### 1.4 幻灯片

开放数据里没有成规模的中文 PPTX：最大的许可明确的 PPTX 集 Zenodo10K 有 10,448 份，只有 20 份标题或文件名含中文
（13 份可用，其中 8 份是四川大学模板）；Common Crawl 每期 .cn 只有约 16 份 PPT/PPTX（对比 11.5 万份 PDF）。
因此只能定向爬取，并辅以合成路线（4.5 节）。

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `edu_courseware` | 高校 WebPlus/Sudy 网站上的课程课件（`/_upload/article/files/`） | PPT、PPTX、PDF | 单个课程页挂 11–14 章，每份 10–30 MB、26–112 页 | public_web | 约 90% CJK；转换 3–9 秒/份 |
| `gov_attachments`（`preset=jilin_zcjd`） | 政府政策解读 PPT 附件（TRS 的 `P0….pptx`） | PPTX、PDF | — | official | 占位符为 0；不少页面是整页图片 |
| `cninfo_ir_office` | 上市公司路演 PPT（见 1.2） | PPT、PPTX | ≥93 份 | disclosure | — |
| `valse_slides` | VALSE 讲者报告 PDF | PDF | 约 281 个目录 × 1.5 份 | public_web | 字体内嵌，约 25% 以中文为主 |
| `zenodo_cjk` | Zenodo10K 中的 CC BY 中文 PPTX，也可按记录 ID 追加 | PPTX | 13 份 | open | 12 份渲染：模板页 210 页、模板整份 35 页被丢弃，保留 125 页 |
| `ypppt_courseware` | 优品PPT 语文课件与模板（ZIP/RAR 直链） | PPTX、ZIP、RAR | 约 763 课，模板数千 | study_only | 模板 12–56% 是占位文字，适合作为 refill 合成路线的版式来源 |

### 1.5 报纸、图书与手册

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `rmrb_pages` | 人民日报电子版逐版 PDF（2023 年起；更早的返回 403，是付费数据库） | PDF | 1,386 期，约 2.2 万版，每天新增约 17 版 | study_only（站点声明只允许阅读和学习研究，禁止商用、建库和镜像） | 31 版保留 30 版，每版中位数 418 行，94.7% 行可信，约 298 DPI |
| `jjrb_pages` | 经济日报电子版逐版 PDF（2020 年起） | PDF | 约 2,400 期，约 2.9 万版 | public_web | 13 版保留 12 版，全部可信 |
| `ctan_zh` | CTAN 中文宏包文档（ctex、lshort-zh-cn、学位论文模板等） | PDF | 79 个宏包 109 份 PDF，约 8–9k 页 | open（只收 CTAN 自由许可列表里的许可） | 页面干净，代码和英文页按语言过滤 |
| `open_books_zh` | 逐份核实许可的开放图书与手册：d2l-zh、Debian 手册、FreeBSD 手册、Python 文档、SUSE 指南 | PDF、ZIP | 113 个候选，约 2 万页 | open（Apache、GPL、MIT、BSD、PSF、GFDL） | 前 30 页抽样保留 56%（前言、代码页被语言过滤）；整本渲染要把 `--max-pages` 调到 1000 以上 |

### 1.6 爬虫语料

| 来源 ID | 内容 | 格式 | 规模 | 条款 | 实测质量 |
|---|---|---|---|---|---|
| `finepdfs_zh` | HuggingFaceFW/finepdfs 的 cmn_Hani 子集（带 Common Crawl WARC 指针） | PDF | 4,913,567 条；68.9% 由 docling 从文本层抽取且未截断，约 340 万份原生数字候选；约 30% .tw、10% .hk | web_mixed | docling 标记的抽样全部为原生数字；回源站下载成功率 38–51% |
| `cc_office_zh` | Common Crawl 列式索引里中文主机下的文档 | PDF、DOC、DOCX、WPS、PPT、PPTX | 每期 .cn 约 11.5 万 PDF、1.6k DOC、950 DOCX、440 WPS、16 PPT/PPTX；繁中为主的 tw/hk 主机另有约 14.5 万份 | web_mixed | 13 份未截断 .cn 记录 12 份原生数字；2024 年前截断率 37% |

## 2. 不采集的来源

| 来源 | 原因 |
|---|---|
| 中国知网、万方、维普、国家哲学社会科学文献中心、中国科技论文在线 | 付费墙或登录；`BLOCKED_HOSTS` 直接丢弃 |
| 百度文库、道客巴巴、豆丁、原创力（book118）、人人文库 | 用户上传的付费文档分享站，权属不明 |
| 国家法律法规数据库 flk.npc.gov.cn | robots.txt `Disallow: /`，并有禁止自动化工具的书面声明；中国政府网两个检索后端同样被 robots 禁止 |
| 中国裁判文书网 | 需要登录且有反爬，文书含大量个人信息 |
| 港交所 HKEXnews | 使用条款明确禁止抓取和 AI 训练（调研时只发过 42 个元数据请求，未下载文件） |
| SciEngine（中国科学杂志社）、docx-corpus 的 CDN 主机 | robots.txt 对 AI 爬虫退出或 `Content-signal: ai-train=no`，`robots.py` 视同禁止 |
| 中国光学期刊网 | robots 拒绝，下载需登录 |
| ChinaXiv | robots.txt 返回 403，按 RFC 9309 视为全站禁止（约 4.8 万篇，CC 许可，建议直接联系获取批量授权） |
| 中科院机构知识库（CSpace/IR Grid） | robots 拒绝本爬虫 |
| 第一PPT 1ppt.com | 下载走阿里云 WAF 的 JS 挑战（acw_sc__v2），不绕过；条款仅限学习且禁止再分发 |
| 51pptmoban、叮当设计、包图 | 文件 CDN 或主站 robots.txt 返回 403 |
| 熊猫办公、WPS 稻壳、iSlide、千图等 | VIP/登录付费墙 |
| 中国大学 MOOC、学堂在线、CCF 数字图书馆 | 课件需要登录或会员 |
| 中小学教材 PDF（ChinaTextbook 系列镜像） | 出版社版权，风险高 |
| 国家标准全文公开系统 openstd.samr.gov.cn | 只有在线阅读器，没有可下载文件 |

## 3. 合规规则

### 3.1 robots.txt 与礼貌

- **自写 RFC 9309 解析器**（`robots.py`）。标准库 `urllib.robotparser` 按第一条匹配且不支持 `*`、`$`，
  会把 Zenodo 的 `Allow: /api/records/*/files` 判错。实现要点：按产品名选组（没有则用 `*` 组），
  最长匹配优先，长度相同时 Allow 优先，容忍全角空格。
- **AI 爬虫退出也算禁止**：robots.txt 对 GPTBot、ClaudeBot、CCBot、Google-Extended 等 AI 代理禁止的路径，
  或声明 `Content-signal: ai-train=no` 的站点，本爬虫同样不抓（`reason` 记为 `ai_optout`）。
- **状态码策略**：401/403 视为全站禁止（比 RFC 严格）；其他 4xx 视为没有规则；429、5xx 或连不上视为暂时全站禁止，
  600 秒后重查（一次网络抖动不会永久封掉一个站，续跑时这类候选会重试）。唯一的放宽：签名 URL 的 CDN
  对所有不带签名的路径（包括 robots.txt）返回 403，并在 `X-Tengine-Error` 里写明 `denied by req auth`，
  这种 403 按 RFC 9309 的"不可用"处理并记日志——ModelScope 的文件都经这种 CDN 下发，而跳转来源
  `www.modelscope.cn/api/` 的 robots.txt 明确允许。
- **robots.txt 本身**：按 UTF-8 读取（忽略 BOM），只解析前 500 KiB（RFC 9309 2.5），通配符用线性时间匹配，
  超过 600 秒的 `Crawl-delay` 视为本次不抓。
- **每一跳都检查**：重定向手动跟随，每一跳都按目标主机自己的 robots.txt 检查，并拒绝 `BLOCKED_HOSTS`、
  非 http(s) 协议、非 ASCII 主机名、有歧义的 authority（反斜杠、`%` 转义）和解析到非公网地址的主机
  （连接时再核对一次实际连上的地址，防 DNS rebinding）；HTTP Range 读取（parquet、WARC）走同一套检查。
  WARC 里存的网页还要再过一遍原网站当前的 robots.txt（含 AI 爬虫退出），结论与直接抓取一致。
- **节奏**：每个主机一个时间槽，默认间隔 2 秒，`Crawl-delay` 更大时取更大值；429/5xx 指数退避，
  `Retry-After` 和退避都不超过 600 秒；curl 通道不自行重试。
- **不绕过任何访问控制**：登录、验证码、WAF/JS 挑战、付费墙、下载配额一律不碰；遇到 JS 挑战页只记录为 HTML。
- **User-Agent** 是 `VOSR-research-crawler/0.1 (text super-resolution research)`，联系方式只能由运行者
  通过 `--contact` 自愿追加，代码里不含任何个人信息。

### 3.2 条款类别

| 类别 | 含义 | 默认运行 |
|---|---|---|
| `official` | 国家机关的法律、法规、决议、决定、命令和其他具有立法、行政、司法性质的文件（著作权法第五条）；台湾著作权法第九条的法律命令公文；联合国 ST/AI/189/Add.9/Rev.2 规定进入公有领域的正式文件 | 是 |
| `open` | 逐文件确认的 CC BY / BY-SA / CC0、Apache、LPPL 等开放许可 | 是 |
| `open_nc` | CC BY-NC(-SA/-ND) 等非商用许可 | 否 |
| `disclosure` | 法定信息披露（上市公司公告、投资者关系记录），版权仍归发布者 | 否 |
| `public_web` | 公开发布但保留所有权利（课件、报告幻灯片、学位论文、报纸、推荐性标准） | 否 |
| `study_only` | 站点条款只允许学习研究使用（模板站） | 否 |
| `web_mixed` | 爬虫语料（Common Crawl 及其衍生），逐文件权利不一 | 否 |

`doc-discover` 遇到未放行的类别直接报错；`doc-fetch` 也会拒绝候选列表里未放行类别的条目，
所以手工拼接的候选文件也绕不过去。适配器选项不能把一个来源变成另一个条款类别的来源
（例如 `gov_attachments` 拒绝 `preset=mee_standards`，推荐性标准只能走 `gov_standards`）。

### 3.3 个人信息

- 标题像人员名单的文件在发现阶段跳过（各适配器的 `PERSONAL` 正则：名单、名册、花名册、身份证，政府站另加
  拟录用、拟聘用、面试人员、考生，巨潮另加签到、登记表、承诺书、人员姓名、通讯录）。
- 台湾公报的公示送达、处分类条目，政府采购的合同公告（签字盖章扫描件、供应商个人信息）默认不采。
- 渲染结果保留来源 URL 和标题，便于事后按需删除。

## 4. 工程实现

```
doc-discover  来源适配器 → 候选 JSONL（key、url / warc / parquet 指针、标题、类型、许可、条款类别）
doc-fetch     Fetcher（robots、节奏、重定向、大小上限、重试）→ 内容寻址 Store + fetch_log（可续跑）
doc-render    解包 → LibreOffice 转 PDF（加固）→ 页面分类 → 按目标行高定 DPI 渲染 → 行框 + 转写 → clean.jsonl
probe         现有无模型探针（JPEG 历史、上采样、边缘宽度、近重复）→ direct / downscale / teacher / reject
```

### 4.1 发现：几类适配器

每个来源一个适配器函数（`catalog.SOURCES` 登记，`doc-sources` 列出）。按枚举方式可分为几类，各有需要注意的坑：

| 方式 | 来源 | 要点 |
|---|---|---|
| 静态 CMS 列表页 | `gov_attachments`、`edu_courseware`、`ccgp_beijing` | TRS 分页 `index_N.html`（上海从 `index_2` 开始），WebPlus 分页 `list2.htm`；附件 URL 有固定签名（TRS 的 `P0…/W0…`，WebPlus 的 `/_upload/article/files/`），真实标题藏在 `sudyfile-attr` 里；北京采购网的列表在 iframe 里，每类只保留最新 143 页 |
| 静态 JSON 索引 | `gov_gazette`（`gbgl.json`） | 中国政府网的检索后端被 robots 禁止，只能用静态 JSON |
| 未公开的检索 API | `cninfo_*`、`ccgp_beijing` 检索模式 | 巨潮每次查询最多 3,000 条，第 101 页起静默回到第 1 页：按日期二分、再按板块拆分，并检测翻页回绕 |
| 公开 API | `doaj_zh`（单查询 1,000 条上限，按年份拆分，2 次/秒）、`modelscope_*`（树形列表，每页最多 2,000 条） | |
| OAI-PMH | `ojs_zh` | 期刊声明的语言不可信（中文期刊标 `eng`），按标题里的 CJK 判断；resumptionToken 重复时停止 |
| 页面 meta | `doaj_zh`、`ntu_theses` | `citation_pdf_url` 只从页面头部读取 |
| 日期遍历 | `rmrb_pages`、`jjrb_pages` | 两套版式按日期切换，连续缺刊若干天或遇到 401/403 即停止 |
| 官方开放资料包 | `tw_gazette` | 月度/年度 ZIP 里同时有 XML 元数据和 PDF，下载阶段可以直接取包内成员，免去二次下载 |
| 远程 parquet 列投影 | `openalex_zh`、`finepdfs_zh`、`zenodo_cjk`（元数据）、`un_documents_zh` | 只用 HTTP Range 精确读需要的列（缓冲读取器的预读曾把一个文件读成 181 MB）；UN 转储的二进制列一个行组可达 2.6 GB，借助页索引只读包含目标行的数据页 |
| Common Crawl 列式索引 | `cc_office_zh` | 每期 300 个 parquet 分片按 SURT 随机编号，页脚读一次后缓存，再按 `url_surtkey` 统计裁剪行组（.cn 只落在 1,497 个行组中的 11 个）；文档的 CLD2 语言字段为空，用主机前缀和同主机 HTML 的 zho 比例判断中文 |
| 逐份核实许可的清单 | `open_books_zh`、`ctan_zh` | 发现阶段重新读取许可页面，找不到许可声明就跳过 |

### 4.2 下载

- `Fetcher.get` 手动跟随最多 5 次重定向并逐跳检查（见 3.1）；声明的 Content-Length 超过上限（默认 200 MB，
  `--max-mb`）直接拒绝，流式读取时超限也中止；连接中途断开会重试；每个请求有一个总截止时间（状态行和头部在
  `timeout` 内到齐，正文平均速率不低于 4 KiB/s），到点直接关闭 socket，解压或 trailer 的死循环也拖不住整个任务。
  少数主机的 TLS 拒绝 Python 客户端时（按异常类型判断）回退到 curl，同样检查 robots 和节奏，走与 requests
  相同的代理；直连时把 curl 钉在已检查过的地址上。
- 单个候选的任何异常（坏压缩包、写不进日志的字段等）只记在这个候选上，任务继续；磁盘写满这类本地错误才会结束任务，
  腾出空间后续跑即可。
- `Store`：文件按 SHA-256 存为 `files/<sha[:2]>/<sha>.<类型>`，类型由文件头嗅探（`sniff.py`：PDF、OLE2 里区分
  ppt/doc/xls、OOXML 里区分 pptx/docx/xlsx、OFD、ZIP 炸弹、RAR、7z、RTF、HTML）。`fetch_log.jsonl` 逐条记录结果，
  重跑时跳过已完成的 key。机构库常把上传时的 multipart 包装一起存进 PDF，入库时剥掉。
- 默认只把可渲染的文档和压缩包算作成功下载；登录墙、JS 挑战返回的 HTML 记为失败原因而不是文档。
- WARC 指针：按偏移和长度做 Range 读取并解 gzip；FinePDFs 只给偏移，就开放式读到 gzip 成员结束。被截断的记录
  （CC-MAIN-2025-13 之前 1 MiB、之后 5 MiB）默认判失败，`--refetch-truncated` 时回源站重新抓取。
- parquet 单元格：`{url, row_group, row, column, item}` 指针通过 HTTP Range 只读需要的列；列块带页索引时
  只读包含目标行的数据页（自带的小型页解码器，与 pyarrow 做过 400 组写入参数的对照，并对文件声明的所有长度设上限），
  否则整块读取；同一行组只读一次，读取失败记在该行组的所有行上。
- ZIP 成员：候选可以写成 `{"zip": {"url", "member"}}`，压缩包只下载一次（台湾公报的开放资料包里已含各条目 PDF，
  免去二次下载）。
- `fetch_log.jsonl` 只按换行符分行（标题里的 U+2028 等不会断行），崩溃留下的半行在续跑时丢弃。

### 4.3 Office 转 PDF

LibreOffice 无头转换，每个 worker 独立 profile，并做了以下加固（都有测试或实测佐证）：

- **宏与链接**：宏安全级别最高并禁止执行，Writer/Calc/Impress 不加载 VBA；Writer 链接更新设为 0（从不），
  Calc 设为 1（`LM_NEVER`，取值查自 LibreOffice 源码 `sw/inc/linkenum.hxx`、`sc/inc/global.hxx`），并开启
  `BlockUntrustedRefererLinks`。对照实验：默认 profile 打开带远程图片链接的 DOCX 会发出 OPTIONS + GET 请求，
  加固后为 0 个请求。
- **网络隔离**：可用时用 `unshare -rn` 在无网络的命名空间里运行 soffice；超时整组进程 kill。
- **字体**：中文文档引用的字体（微软雅黑、宋体、仿宋_GB2312、楷体、等线、方正/汉仪/字魂系列）在 Linux 上几乎都没有，
  默认 fontconfig 会把它们匹配到只含拉丁字母的 Inter。`convert.py` 生成的 fonts.conf：
  - 用 `<accept>` 别名把常见字体名映射到 Noto Sans/Serif CJK SC/TC 和 AR PL UKai（`<prefer>` 会连已安装的字体一起覆盖，
    实测不能用）；
  - 按名字片段追加规则（楷/魏/隶/行 → 楷体，宋/仿/明 → 衬线，黑/圆/雅/等线 → 无衬线）；
  - `<rejectfont>` 屏蔽 Unifont、文泉驿点阵、IPA、FreeFont、X11 位图等兜底字体，缺字时输出 .notdef 而不是一个
    风格错乱的点阵字，渲染阶段据此把该行标为不可信。
- 导出参数：无损图片、不降采样、不导出隐藏幻灯片和备注页。

### 4.4 渲染与标注

`render.py` 用 PyMuPDF 逐页处理（每份文件默认最多 200 页，多进程）：

1. **页面分类**（先分类，再决定标注是否可信）：

   | 类别 | 判据 | 处理 |
   |---|---|---|
   | `text` | 可用文本层 | 渲染，行框和转写为可信标注（`trusted: true`） |
   | `garbled` | 私用区/替换符/控制字符超过 2%；方正 BookMaker 等无 ToUnicode 字体且 CJK 比例低；CJK 中 GB2312∪Big5 常用字比例 < 0.9（编码映射错乱时会出现一堆生僻字） | 只渲染，字形框准确但转写不可信 |
   | `vector` | 不足 10 个字符但有矢量墨迹（字形转成了轮廓，如国务院公报 2021 年后各期） | 只渲染，无标注 |
   | `scanned` | 图片覆盖超过半页，且文字很少或大多是不可见的 OCR 层 | 丢弃 |
   | `blank` | 无可见内容 | 丢弃 |

   `text` 页的 CJK 比例低于 `--min-cjk`（默认 0.3）时按语言丢弃：英文摘要、参考文献页会把整份论文的 CJK 比例拉到
   0.28–0.43，所以按页过滤而不是按文件过滤。
2. **按行高定 DPI**：以字符数加权的可见行高中位数（斜排文字按垂直于行的方向量）为基准，把它缩放到
   `--text-height` 区间（默认 32–64 px，每页用确定性种子抽一个目标），DPI 限制在 72 到 600 之间，单页不超过
   2500 万像素（图纸、海报在 72 DPI 下已超限的，允许低于 72 DPI）。HR 直接由矢量渲染得到，从不重采样。
3. **逐字可见性**：文本层的每个字符都和绘制轨迹（`get_texttrace`）对上号，再按字符判断是否真的画出来了：
   渲染模式 3、透明度 0、批注和表单域里的文字（页面渲染时不画批注）直接去掉；渲染后字框里必须有墨迹
   （按 RGB 各通道比较，彩色字和同亮度底色也能区分）；之后又被图片、填充盖住的字，要和"去掉文字后重渲染的页面"
   比较像素才算可见（照片下面的文字、扫描页下面的 OCR 层；整页文字都在图片下面的页改判为扫描页）；
   渲染后低于 8 px、或低于"页面行高中位数与本行最大字高两者较小值"的 0.35 倍的字（夹在正文里几磅大小的防复制干扰字）
   算隐藏。一行只保留连续可见的片段，每段各有自己的四边形和转写。
4. **行框几何**：竖排文字和 LibreOffice 输出的负宽度字框改用实际绘制的字形框，竖向堆叠的单字合成一列；
   字体度量异常时改用标准高度（方正 BookMaker 的 ascender 只有 0.001，`recover_line_quad` 会算出零高度的斜线；
   超过 1.5 em 的字体限制在基线上方 1 em、下方 0.25 em）；仍然退化的框丢弃（`degenerate_lines`）；超出图像的字形
   被截掉（`offpage_lines`），其余夹到图像内，保证每个多边形都满足 `geometry.regions` 的约定。最后乘以
   `page.rotation_matrix` 和缩放矩阵，与渲染像素对齐（旋转页有专门测试）。
5. **重复图层**：阴影、伪粗体、叠印产生的重叠副本（IoU > 0.5）只保留最上层；相距更远的同文本（远阴影、表格下一行的
   相同单元格）各自保留标注。
6. **逐段可信度**：以下情况保留框但 `trusted: false`——含 .notdef 字形（所有允许字体都缺这个字）；含转写无法表达的字符
   （控制字符，例如方正字体把数字编码成控制字符，"（2021—2035年）"会变成"（—年）"；私用区；U+FFFD；
   MuPDF 找不到 Unicode 的字形）；画了字形却被文本层读成空格的字符；ToUnicode 给一个字形塞了多个字符而它又不是
   拉丁连字（fi、ffl 等，宽度足以容纳）或重复标点（一个字形画出的"——"）。去掉符号字体的项目符号和人民日报 NEU
   字体的边框碎片，NFKC 只用于康熙部首和兼容汉字，全角标点保持原样。页面大部分 CJK 用未嵌入字体绘制时标
   `style_substituted`（MuPDF 用自带的兜底字体画，字形正确但字体风格不是原样）。
7. **模板与去重**：演示文稿和横版 PDF 页上的占位符、所有页面上的模板站水印都会让该页被丢弃；全局去近重复要求 pHash
   相近且转写的公共子序列达到 90%（不足 20 字时要求完全相同），只看 pHash 会把同一版式、文字不同的页误删。
   同一个文件（不同 URL、压缩包内外）只渲染一次，其他来源记在 `also` 里。
8. **容错**：单个文件出错（转换失败、worker 崩溃或被 OOM 杀掉、超时）只记为该文件的错误，任务继续。
9. **可选检测器**（`--detector rapidocr`，约 0.7 秒/页）：检测文本层没覆盖到的文字（图片里的字、艺术字），
   未标注面积超过 `--max-unlabeled`（默认 5%）的页丢弃。

输出：`pages/` 下的 PNG、`pages.jsonl`（所有页面及丢弃原因）、`clean.jsonl`（保留页，VOSR manifest 格式：
`id`、`source`、`document{sha256, type, url, title, license, terms, page}`、`image_path`、`dpi`、`size`、
`median_text_px`、`annotations[{polygon, text, font, size_px, annotation_origin, trusted}]`、`flags`、`class`）、
`documents.jsonl`（每份文件的状态和用到的字体）和 `summary.json`。

### 4.5 PPT 专项

- 真实课件和政府解读 PPT 的占位符字符为 0，而模板站的模板有 12–56% 是"单击此处添加标题"类占位文字、34–72% 段落重复。
  渲染时含占位符或模板站水印的页面直接丢弃，占位页超过 30% 的整份文件丢弃。
- **合成路线 `refill.py`**：保留真实模板的版式、主题、字体和段落格式，把占位文字和英文填充文字替换成视觉宽度相近的
  真实中文句子（建议用第五条政府文件做语料），再走同样的转换渲染流程。在 ypppt 一份模板上实测：替换 116 段，
  占位符比例 0.555 → 0，CJK 比例 0.096 → 0.555，重复段落 0.723 → 0.023，26 页版式完好。
- 压缩包：模板站提供 ZIP/RAR，ZIP 成员名是没有 UTF-8 标志的 GBK；`unpack.py` 通过系统 libarchive 解 ZIP 和 RAR，
  在内存中读取，只放行 PDF/PPT/PPTX/DOC/DOCX/ODP，成员名压平成清洗过的文件名（从不按压缩包里的路径写盘），
  跳过链接和加密成员，并限制条目数（10,000）、返回成员数（64）、单个成员（200 MB）和总量（512 MB）以防压缩炸弹。

## 5. 用法

依赖：`pymupdf`、`pyarrow`、`requests`、`numpy`、`opencv-python`（`data-pipeline` 依赖组），系统里的
LibreOffice（`soffice`）、Noto Sans/Serif CJK 和 AR PL UKai 字体，以及 `libarchive.so.13`（解 RAR）。

```bash
# 来源目录：条款类别、文件类型、是否默认运行
python -m data_pipeline doc-sources

# 1) 发现：每个来源一个候选 JSONL；--opt 传适配器选项（值可以是 JSON）
python -m data_pipeline doc-discover --source moj_xzfg --output /data/docs/cand/moj.jsonl
python -m data_pipeline doc-discover --source gov_attachments --opt preset=beijing --limit 500 \
  --output /data/docs/cand/beijing.jsonl
python -m data_pipeline doc-discover --source cninfo_ir_office --accept-terms disclosure \
  --opt years=2012-2020 --output /data/docs/cand/cninfo_ir.jsonl

# 2) 下载：内容寻址存储，可中断续跑；默认只保留可渲染的文档和压缩包
python -m data_pipeline doc-fetch --candidates /data/docs/cand/moj.jsonl --store /data/docs/store
python -m data_pipeline doc-fetch --candidates /data/docs/cand/cninfo_ir.jsonl --store /data/docs/store \
  --accept-terms disclosure --types doc,docx,ppt,pptx

# 3) 转换渲染：每页 HR PNG + 行框；--text-height 控制渲染后的文字行高
python -m data_pipeline doc-render --store /data/docs/store --output /data/docs/render --workers 8

# 4) 接入现有清洗探针与六阶段 pipeline
python -m data_pipeline probe --input /data/docs/render/clean.jsonl --output /data/docs/probe
```

公共网络参数：`--interval`（同一主机两次请求的最小间隔，默认 2 秒）、`--max-mb`（单文件上限，默认 200）、
`--contact`（追加到 User-Agent 的联系方式，默认没有）、`--accept-terms`（放行审阅过的条款类别，可重复）、
`--cc-data-host`（允许读取 data.commoncrawl.org，见第 7 节）。发现阶段的日志写到 stderr，候选逐条落盘，
Ctrl-C 中断时保留已找到的部分。

各适配器的选项写在对应模块的函数文档里（`data_pipeline/documents/<模块>.py`）。

## 6. 端到端实测

（渲染修复完成后，在 131 份文件的实测存储上重新渲染并更新本节。）

## 7. 需要决定的事项

以下事项涉及法律或对外关系，代码里都做了保守的默认值，需要项目负责人决定：

1. **data.commoncrawl.org**：它的 robots.txt 写着 `User-agent: * / Disallow: /`，而 Common Crawl 的文档把它列为
   HTTPS 下载入口。默认不访问：只用 Hugging Face 桶镜像的 6 期（CC-MAIN-2026-17/21/25/30/34/39），FinePDFs 的 WARC
   指针（大多是更早的期次）改为回源站抓取，实测成功率只有 38–51%。加 `--cc-data-host` 才会从该主机读索引和 WARC。
2. **条款类别的法律审阅**：`open_nc`、`disclosure`、`public_web`、`study_only`、`web_mixed` 是否可用于内部训练。
   中国著作权法没有文本与数据挖掘例外（第二十四条未列），报纸、公报还有出版者的版式设计权（第三十七条，10 年）。
3. **联合国文件**：目前按 ST/AI/189/Add.9/Rev.2（正式记录和联合国文件原则上属于公有领域）归入 `official`。
   该文件的试行期曾通过增编延长，需要确认它仍然有效。
4. **AI 爬虫退出的范围**：现在只把对 AI 爬虫明确禁止的路径算作退出。北京市政府网站在 robots.txt 里列了 10 个
   AI 爬虫，但它们和其他爬虫共用 `Disallow: /sd`，没有单独全站禁止，所以没有当作全站退出。是否从严处理？
5. **API 密钥与批量授权**：OpenAlex API（免费，但要用个人邮箱注册密钥）、CORE API 可以替代 3.6 GB 的快照扫描；
   ChinaXiv（robots 拒绝，约 4.8 万篇 CC 论文）、国家法律法规数据库（robots 禁止）可以发函申请批量数据。
   这些都需要以项目名义联系，代码里不会自动使用任何个人信息。
6. **字体授权**：现在用 Noto CJK、AR PL UKai 替代微软雅黑、宋体、方正、汉仪等商用字体。HR 就是渲染结果本身，
   替代字体不影响标注正确性，但字体风格的多样性受限；是否购买字体授权由项目决定。

## 8. 已知局限

- **方正 BookMaker 等无 ToUnicode 的 PDF**（北京市政府公报全部、部分学报和财报）：渲染正确但转写是乱码，
  只能作为无标注 HR 使用；恢复标签需要 OCR 或字形哈希表，尚未实现。
- **字形转轮廓的页面**（国务院公报 2021 年后各期）没有文本层，同样只能作为无标注 HR，或配合检测器/OCR 补框。
- **OFD 和金山 WPS 格式**：LibreOffice 打不开 OFD；.wps 靠内容嗅探转换，只用改名的 doc/docx 测过。
- **字体风格**：Office 文件用替代字体渲染，楷/宋/黑三大类能保持，具体字体（方正、汉仪、字魂）的风格会丢失；
  PDF 内嵌字体则保持原样。
- **地域可达性**：部分省级门户（浙江、四川、广东文件库、多数省级政府采购网）从当前出口连接被重置或返回 503，
  按 robots 规则视为暂时禁止；从国内网络运行时可能可以访问。
- **标注的剩余边界情况**（对抗式验证用合成样例构造出来，在 131 份实测文件里没有出现或只影响本来就错的标注）：
  TeX 数学扩展字体（cmex）的 ToUnicode 常常是错的拉丁字母，个别仍被标为可信，且其悬挂到基线以下的定界符框被截短；
  斜排文字里"画了字形却被读成空格"的情况没有做墨迹检查；与可见行相距不到 1 em 的白色同文本副本可能借到邻行墨迹。
- **体量估计**：表中"估计"项按抽样比例外推，实际可用量还要乘以下载成功率和页面保留率（见第 6 节）。
