"""Chinese document sources: what each holds, its terms, and the adapter that lists candidates.

An adapter is ``discover(fetcher, options) -> iterator of candidate dicts`` where a
candidate has ``key`` (stable, unique), ``url`` and/or ``warc`` ({filename, offset,
length?, base?}), ``source``, ``title``, ``type_hint`` (pdf, docx, pptx, ...), ``license``
and free-form ``meta``; optional ``headers`` (e.g. a Referer) are sent with the download.
Every HTTP request goes through the given ``Fetcher`` (robots.txt, per-host pacing).

Terms classes, from least to most restrictive. Only ``official`` and ``open`` run by
default; the others need ``--accept-terms <class>`` after a human has read the terms:
  official        PRC Copyright Law Art. 5: laws, regulations and documents of state organs;
                  Taiwan Copyright Act Art. 9 (laws, orders, official documents)
  open            a free licence verified per document: CC BY / BY-SA / CC0, or the free-software
                  documentation licences (Apache, MIT, BSD, GPL, LGPL, LPPL, GFDL, PSF)
  open_nc         CC BY-NC(-SA/-ND) or similar non-commercial licences
  disclosure      statutory public disclosures; copyright stays with the issuer
  public_web      freely published, all rights reserved (courseware, talk slides, theses)
  study_only      the site's terms allow study/research use only (template sites)
  web_mixed       crawl corpora (Common Crawl and derivatives): per-document rights vary
"""

import importlib

DEFAULT_TERMS = ("official", "open")

SOURCES = {
    # --- government and legal (Art. 5) ---
    "moj_xzfg": dict(adapter="gov:moj_xzfg", terms="official", doc_types=["docx", "pdf"],
                     content="国家行政法规库: all ~606 current administrative regulations, each as DOCX and PDF"),
    "gov_gazette": dict(adapter="gov:gazette", terms="official", doc_types=["pdf"],
                        content="国务院公报 issue PDFs 2021-; vector pages, many with outlined glyphs (no text layer)"),
    "gov_attachments": dict(adapter="gov:trs_attachments", terms="official", doc_types=["pdf", "doc", "docx", "ppt", "pptx"],
                            content="Attachments of government document pages on TRS/WebPlus CMS sites (Beijing, Shanghai, "
                                    "MOST, MOF, NPC reports, 政策解读 slide decks); site presets or listing URLs",
                            refuse_options={"preset": ["mee_standards"]}),
    # Recommended standards (GB/T, HJ) are protected works (1999 NCAC reply), unlike mandatory ones.
    "gov_standards": dict(adapter="gov:trs_attachments", terms="public_web", doc_types=["pdf"],
                          content="生态环境部 environmental standards full texts (mandatory GB and recommended HJ/GB/T)",
                          defaults={"preset": "mee_standards"}),
    "tw_gazette": dict(adapter="taiwan:ey_gazette", terms="official", doc_types=["pdf"],
                       content="行政院公報 (Taiwan Executive Yuan Gazette) items 2005-: regulations, administrative rules, "
                               "draft-rule previews, plus 總說明/對照表 attachments, from the Open Data archives; "
                               "處分, service by publication and lists of people skipped (Traditional Chinese)"),
    # Notices under the Government Procurement Law, drafted by purchasers and agencies: not clearly Art. 5.
    "ccgp_beijing": dict(adapter="procurement:ccgp_beijing", terms="public_web", doc_types=["pdf", "doc", "docx"],
                         content="北京市政府采购网 notice attachments: tender notices, 采购需求 and full 招标文件 "
                                 "(DOCX/PDF/DOC); listings hold the newest 2,002 notices per type, --opt search=true "
                                 "since=YYYY-MM-DD reaches 2010-; contracts and supplier paperwork skipped"),
    # --- financial disclosure ---
    "cninfo_ir_office": dict(adapter="cninfo:ir_office", terms="disclosure", doc_types=["doc", "docx", "ppt", "pptx"],
                             content="巨潮 SZSE investor-relations records 2012-2020: ~57k native Word files and native "
                                     "PPT/PPTX roadshow decks"),
    "cninfo_slides": dict(adapter="cninfo:slides", terms="disclosure", doc_types=["pdf", "ppt", "pptx"],
                          content="巨潮 investor presentation decks (业绩说明会/演示文稿/推介材料), mostly PDF slides"),
    "cninfo_reports": dict(adapter="cninfo:reports", terms="disclosure", doc_types=["pdf"],
                           content="巨潮 A-share periodic reports and other announcements (~632k items in 2024)"),
    # --- academic ---
    "doaj_zh": dict(adapter="academic:doaj", terms="open_nc", doc_types=["pdf"],
                    content="Articles of DOAJ journals in Chinese (331 journals, ~407k articles), PDFs via "
                            "citation_pdf_url on Magtech/Renhe/OJS platforms; licence per journal"),
    "ojs_zh": dict(adapter="academic:ojs", terms="open_nc", doc_types=["pdf"],
                   content="Chinese-language OJS journals over OAI-PMH (e.g. ojs.omniscient.sg, CC BY-NC)"),
    "openalex_zh": dict(adapter="academic:openalex", terms="open_nc", doc_types=["pdf"],
                        content="Chinese OA works with a best_oa pdf_url from the OpenAlex snapshot on S3"),
    "ntu_theses": dict(adapter="academic:ntu_theses", terms="public_web", doc_types=["pdf"],
                       content="國立臺灣大學 theses authorised for worldwide public access (Traditional Chinese)"),
    # --- slides ---
    "edu_courseware": dict(adapter="slides:webplus", terms="public_web", doc_types=["ppt", "pptx", "pdf"],
                           content="University course slides linked from WebPlus/Sudy CMS pages (/_upload/article/files/)"),
    "valse_slides": dict(adapter="slides:valse", terms="public_web", doc_types=["pdf"],
                         content="VALSE webinar speaker slides 2014- (PDF, fonts embedded; ~25% mainly Chinese)"),
    "zenodo_cjk": dict(adapter="slides:zenodo_cjk", terms="open", doc_types=["pptx"],
                       content="CC-BY Chinese decks among Zenodo10K (13 usable of 20 CJK rows; 8 are 四川大学 "
                               "templates) and other Zenodo records named by id (--opt records=[...])"),
    "ypppt_courseware": dict(adapter="slides:ypppt", terms="study_only", doc_types=["pptx", "zip", "rar"],
                             content="优品PPT 语文 courseware and templates; direct ZIP/RAR downloads; study-only terms"),
    # --- openly licensed books and manuals ---
    "ctan_zh": dict(adapter="books:ctan_chinese", terms="open", doc_types=["pdf"],
                    content="Chinese documentation of CTAN packages (topics chinese-doc and chinese: ctex, lshort-zh-cn, "
                            "thesis templates, ...; 109 PDFs from 79 packages, licence ids on CTAN's free list)"),
    "open_books_zh": dict(adapter="books:open_books", terms="open", doc_types=["pdf", "zip"],
                          content="Curated openly licensed Chinese books and manuals, licence page read at discovery: "
                                  "d2l-zh (Apache-2.0), Debian manuals (GPL/MIT), FreeBSD handbooks, Python docs (PSF), "
                                  "SUSE guides (GFDL); ~20k pages"),
    # --- newspapers ---
    # The site notice allows reading and study/research use only (no commercial use, databases or mirrors).
    "rmrb_pages": dict(adapter="news:rmrb", terms="study_only", doc_types=["pdf"],
                       content="人民日报 e-paper page PDFs 2023- (Founder typesetting, embedded fonts, exact text layer; "
                               "~22k pages, ~17 more a day)"),
    "jjrb_pages": dict(adapter="news:jjrb", terms="public_web", doc_types=["pdf"],
                       content="经济日报 e-paper page PDFs 2020- (~29k pages)"),
    # --- crawl corpora ---
    "finepdfs_zh": dict(adapter="commoncrawl:finepdfs", terms="web_mixed", doc_types=["pdf"],
                        content="HuggingFaceFW/finepdfs cmn_Hani: 4.9M Chinese PDFs with Common Crawl WARC pointers"),
    "cc_office_zh": dict(adapter="commoncrawl:index", terms="web_mixed", doc_types=["pdf", "doc", "docx", "ppt", "pptx"],
                         content="Common Crawl columnar index: office/PDF captures under Chinese host prefixes "
                                 "(.cn, edu.cn, gov.cn, zho-dominant tw/hk hosts)"),
}

# Hosts that opt out of AI crawlers, block automated clients or sit behind paywalls; candidate
# URLs on them are dropped whatever the adapter found (robots.txt is still checked for the rest).
BLOCKED_HOSTS = ("cnki.net", "cnki.com.cn", "wanfangdata.com.cn", "cqvip.com", "wenku.baidu.com", "doc88.com",
                 "docin.com", "book118.com", "renrendoc.com", "sciengine.com", "engine.scichina.com",
                 "opticsjournal.net", "hkexnews.hk", "hkex.com.hk", "flk.npc.gov.cn", "wenshu.court.gov.cn",
                 "ncpssd.cn", "ncpssd.org", "paper.edu.cn", "1ppt.com")


def load(source_id):
    module, function = SOURCES[source_id]["adapter"].split(":")
    return getattr(importlib.import_module(f".{module}", __package__), function)


def blocked(url):
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in BLOCKED_HOSTS)


def discover(source_id, fetcher, options=None, accepted=()):
    spec = SOURCES[source_id]
    if spec["terms"] not in DEFAULT_TERMS and spec["terms"] not in accepted:
        raise PermissionError(f"{source_id}: terms class '{spec['terms']}' needs --accept-terms {spec['terms']} "
                              "after reviewing the source's terms (see docs/chinese_document_sources.md)")
    options = dict(spec.get("defaults", {}), **(options or {}))
    for key, refused in spec.get("refuse_options", {}).items():
        if options.get(key) in refused:
            raise PermissionError(f"{source_id}: {key}={options[key]} belongs to another source with other terms")
    for candidate in load(source_id)(fetcher, options):
        if candidate.get("url") and blocked(candidate["url"]):
            continue
        candidate.setdefault("source", source_id)
        candidate.setdefault("terms", spec["terms"])
        yield candidate
