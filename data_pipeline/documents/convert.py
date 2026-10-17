"""Office files to PDF with LibreOffice: font aliases, macro-free disposable profiles, timeouts.

Chinese decks and Word files name Windows or commercial fonts (微软雅黑, 宋体, 仿宋_GB2312,
楷体 ...) that are rarely installed and almost never embedded. Without aliases fontconfig
maps them to a Latin-only face and CJK falls back to one sans font, so Song, Kai and
FangSong styles are lost. The aliases below map each family to an installed CJK face
of the same style; the HR target is the rendered page, so substituted glyphs are
still exact ground truth, but layout may shift, which is why text boxes come from the
converted PDF and never from the source file's coordinates.

Profiles disable macros (security level 3, execution off) and the automatic update of
links: Writer Content/Update/Link 0 = NEVER (sw/inc/linkenum.hxx), Calc 1 = LM_NEVER
(sc/inc/global.hxx ScLkUpdMode). Files that carry VBA projects are refused before
conversion.
"""

import os
import signal
import subprocess
import zipfile
from pathlib import Path

FONT_ALIASES = {
    "Noto Sans CJK SC": ["微软雅黑", "Microsoft YaHei", "Microsoft YaHei UI", "Microsoft YaHei Light", "黑体", "SimHei",
                         "等线", "等线 Light", "DengXian", "DengXian Light", "思源黑体", "思源黑体 CN", "思源黑体 CN Normal",
                         "思源黑体 CN Bold", "思源黑体 CN Light", "Source Han Sans", "Source Han Sans CN",
                         "Source Han Sans SC", "阿里巴巴普惠体", "Alibaba PuHuiTi", "幼圆", "YouYuan", "华文细黑", "STXihei",
                         "华文黑体", "STHeiti", "苹方", "PingFang SC", "方正黑体简体", "方正兰亭黑简体", "汉仪旗黑",
                         "冬青黑体简体中文", "Hiragino Sans GB"],
    "Noto Serif CJK SC": ["宋体", "SimSun", "新宋体", "NSimSun", "华文中宋", "STZhongsong", "华文宋体", "STSong",
                          "思源宋体", "Source Han Serif", "Source Han Serif CN", "Source Han Serif SC", "方正小标宋简体",
                          "方正小标宋_GBK", "方正书宋简体", "方正书宋_GBK", "仿宋", "仿宋_GB2312", "FangSong",
                          "FangSong_GB2312", "华文仿宋", "STFangsong", "方正仿宋简体", "方正仿宋_GBK", "宋体-方正超大字符集"],
    "AR PL UKai CN": ["楷体", "楷体_GB2312", "KaiTi", "KaiTi_GB2312", "华文楷体", "STKaiti", "方正楷体简体", "方正楷体_GBK",
                      "隶书", "LiSu", "华文隶书", "STLiti", "华文行楷", "STXingkai"],
    "Noto Sans CJK TC": ["微軟正黑體", "Microsoft JhengHei", "Microsoft JhengHei UI", "蘋方-繁", "PingFang TC"],
    "Noto Serif CJK TC": ["新細明體", "細明體", "PMingLiU", "MingLiU", "MingLiU_HKSCS"],
    "AR PL UKai TW": ["標楷體", "DFKai-SB"],
}

# Unknown families are matched by name fragments first (later rules win, so Kai beats Song
# beats Hei); the explicit aliases above are applied after these and take precedence.
FONT_KEYWORDS = [
    ("Noto Sans CJK SC", ["黑", "圆", "雅", "等线", "Hei", "Sans GB", "Sans CJK", "Sans SC", "Sans CN", "PingFang"]),
    ("Noto Serif CJK SC", ["宋", "仿", "明", "Song", "Ming", "Serif CJK", "Serif SC", "Serif CN"]),
    ("AR PL UKai CN", ["楷", "魏", "隶", "行", "Kai"]),
]

PROFILE_SETTINGS = {
    ("org.openoffice.Office.Common/Security/Scripting", "MacroSecurityLevel"): "3",
    ("org.openoffice.Office.Common/Security/Scripting", "DisableMacrosExecution"): "true",
    ("org.openoffice.Office.Common/Security/Scripting", "BlockUntrustedRefererLinks"): "true",
    ("org.openoffice.Office.Writer/Content/Update", "Link"): "0",
    ("org.openoffice.Office.Calc/Content/Update", "Link"): "1",
}


class ConversionError(Exception):
    pass


def fonts_conf(directory):
    """Write the alias fontconfig file once and return its path (for FONTCONFIG_FILE)."""
    from xml.sax.saxutils import escape
    path = Path(directory) / "fonts.conf"
    lines = ['<?xml version="1.0"?>', '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">', "<fontconfig>",
             '  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>']
    for target, keywords in FONT_KEYWORDS:
        for keyword in keywords:
            lines.append(f'  <match target="pattern"><test qual="any" name="family" compare="contains">'
                         f'<string>{escape(keyword)}</string></test><edit name="family" mode="prepend" '
                         f'binding="same"><string>{escape(target)}</string></edit></match>')
    for target, families in FONT_ALIASES.items():
        for family in families:
            lines.append(f'  <alias binding="same"><family>{escape(family)}</family>'
                         f"<prefer><family>{escape(target)}</family></prefer></alias>")
    text = "\n".join(lines + ["</fontconfig>", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def prepare_profile(directory):
    """A LibreOffice user profile whose registry disables macros and link updates."""
    user = Path(directory) / "user"
    user.mkdir(parents=True, exist_ok=True)
    items = "\n".join(f'<item oor:path="/{path}"><prop oor:name="{name}" oor:op="fuse"><value>{value}</value>'
                      f"</prop></item>" for (path, name), value in PROFILE_SETTINGS.items())
    (user / "registrymodifications.xcu").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" '
        'xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f"{items}\n</oor:items>\n", encoding="utf-8")
    return Path(directory)


def has_macros(path):
    data = Path(path).read_bytes()
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                return any(name.lower().endswith("vbaproject.bin") for name in archive.namelist())
        except zipfile.BadZipFile:
            return False
    return "_VBA_PROJECT".encode("utf-16-le") in data or b"_VBA_PROJECT" in data


def to_pdf(source, out_dir, profile, fonts, timeout=180, soffice="soffice"):
    """Convert one file; raises ConversionError. ``source`` keeps its sniffed extension."""
    source, out_dir = Path(source), Path(out_dir)
    if has_macros(source):
        raise ConversionError("macros")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [soffice, f"-env:UserInstallation={Path(profile).resolve().as_uri()}", "--headless", "--invisible",
               "--norestore", "--nolockcheck", "--nodefault", "--nologo", "--convert-to", "pdf",
               "--outdir", str(out_dir), str(source)]
    env = dict(os.environ, FONTCONFIG_FILE=str(fonts))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                               start_new_session=True)
    try:
        _, error = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)  # soffice forks; kill the whole group
        process.communicate()
        raise ConversionError("timeout")
    pdf = out_dir / (source.stem + ".pdf")
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise ConversionError("failed: " + error.decode("utf-8", "replace").strip()[-200:])
    return pdf
