"""File type from content, not from the URL or Content-Type (portals serve PDFs as
application/octet-stream, error pages as .pdf, and .doc files that are really HTML)."""

import io
import zipfile

OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# Zip archives are inspected, never extracted; these bound a zip bomb's directory.
MAX_ZIP_ENTRIES = 20000
MAX_ZIP_UNCOMPRESSED = 2 << 30

# Kingsoft .wps/.dps are compound files with Word/PowerPoint streams and sniff as doc/ppt.
CONVERTIBLE = {"pptx", "ppt", "docx", "doc", "odp", "odt", "rtf"}
RENDERABLE = CONVERTIBLE | {"pdf"}


def _utf16(name):
    return name.encode("utf-16-le")


def sniff(data):
    """Return a short type name for the bytes: pdf, pptx, docx, ppt, doc, ofd, ..."""
    head = data[:2048]
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(OLE2):
        # Compound-file directory entries store stream names in UTF-16LE.
        if _utf16("PowerPoint Document") in data:
            return "ppt"
        if _utf16("WordDocument") in data:
            return "doc"
        if _utf16("Workbook") in data or _utf16("Book") in data:
            return "xls"
        return "ole2"
    if head.startswith(b"PK\x03\x04"):
        return _zip_type(data)
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if stripped.startswith(b"{\\rtf"):
        return "rtf"
    if stripped.startswith((b"<!doctype html", b"<html", b"<head", b"<script", b"<?xml")) or b"<html" in stripped[:512]:
        return "html"
    if head.startswith(b"Rar!\x1a\x07"):
        return "rar"
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if head[:4] in (b"\x89PNG", b"GIF8") or head[:3] == b"\xff\xd8\xff":
        return "image"
    return "unknown"


def _zip_type(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES or sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
                return "zip_bomb"
            names = {i.filename for i in infos}
            if "ppt/presentation.xml" in names:
                return "pptx"
            if "word/document.xml" in names:
                return "docx"
            if "xl/workbook.xml" in names:
                return "xlsx"
            if "OFD.xml" in names:
                return "ofd"
            if "mimetype" in names:
                kind = archive.read("mimetype")[:100]
                if b"opendocument.presentation" in kind:
                    return "odp"
                if b"opendocument.text" in kind:
                    return "odt"
            return "zip"
    except (zipfile.BadZipFile, ValueError, OSError):
        return "zip_broken"


def cjk_ratio(text):
    """Share of CJK ideographs among non-space characters."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if "一" <= c <= "鿿" or "㐀" <= c <= "䶿" or "豈" <= c <= "﫿")
    return cjk / len(chars)
