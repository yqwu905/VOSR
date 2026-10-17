"""Document members of the ZIP and RAR archives that template sites ship, read in memory.

ZIP goes through zipfile; Chinese tools store GBK names without the UTF-8 flag, which zipfile
decodes as cp437, so such names are re-encoded and decoded as gb18030. RAR (and 7z, tar) goes
through the system libarchive (libarchive.so.13) via ctypes, since unrar and 7z are not
installed. Only whitelisted document types come back; names are flattened to a sanitised base
name (nothing is written under an archive-controlled path), links and encrypted members are
skipped, and member count and sizes are capped against archive bombs.
"""

import ctypes
import ctypes.util
import io
import os
import re
import stat
import zipfile

EXTENSIONS = ("pdf", "ppt", "pptx", "doc", "docx", "odp")
MAX_ENTRIES = 10000      # archive entries looked at
MAX_MEMBERS = 64         # documents returned
MAX_MEMBER = 200 << 20   # bytes of one member
MAX_TOTAL = 512 << 20    # bytes of all members

ARCHIVE_EOF, ARCHIVE_WARN, AE_IFREG = 1, -20, 0o100000
_lib = None


class ArchiveError(Exception):
    pass


def members(path, extensions=EXTENSIONS, max_members=MAX_MEMBERS, max_member=MAX_MEMBER, max_total=MAX_TOTAL,
            log=None):
    """[(flat name, bytes)] of the archive's document members, in archive order.

    ``path`` is a file path or the archive bytes. A damaged or unsupported archive yields
    the members read before the damage (possibly none) instead of raising."""
    log = log or (lambda message: None)
    if isinstance(path, (bytes, bytearray)):
        head = bytes(path[:4])
    else:
        with open(path, "rb") as handle:
            head = handle.read(4)
    entries = _zip_entries(path) if head in (b"PK\x03\x04", b"PK\x05\x06") else _archive_entries(path)
    found, names, total = [], set(), 0
    try:
        for index, (name, size, read) in enumerate(entries):
            if index >= MAX_ENTRIES or len(found) >= max_members:
                log(f"archive entry cap reached at {index} entries")
                break
            base = _flat(name)
            if (_extension(base) not in extensions or base.startswith("._") or "__MACOSX" in name
                    or (size is not None and (size > max_member or total + size > max_total))):
                continue
            data = read(max_member + 1)
            if data is None or len(data) > max_member or total + len(data) > max_total:
                log(f"skip member {name!r}: unreadable or over the size cap")
                continue
            total += len(data)
            found.append((_unique(base, names), data))
    except (ArchiveError, zipfile.BadZipFile, OSError, EOFError) as error:
        log(f"archive damaged after {len(found)} members: {error}")
    finally:
        entries.close()
    return found


def _extension(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _flat(name):
    base = re.split(r"[\\/]", name)[-1]
    base = re.sub(r'[\x00-\x1f\x7f:*?"<>|]+', "_", base).strip(" .")
    return base[-120:] or "member"


def _unique(base, names):
    name, number = base, 1
    while name in names:
        stem, dot, extension = base.rpartition(".")
        name = f"{stem}_{number}.{extension}" if dot else f"{base}_{number}"
        number += 1
    names.add(name)
    return name


def _zip_entries(path):
    with zipfile.ZipFile(io.BytesIO(path) if isinstance(path, (bytes, bytearray)) else path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.flag_bits & 0x1 or stat.S_ISLNK(info.external_attr >> 16):
                continue
            name = info.filename
            if not info.flag_bits & 0x800:
                name = _recode(name)
            yield name, info.file_size, lambda limit, info=info: _read_zip(archive, info, limit)


def _recode(name):
    """The raw bytes of a name zipfile decoded as cp437: UTF-8 (macOS tools omit the flag) or GBK."""
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for codec in ("utf-8", "gb18030"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            pass
    return name


def _read_zip(archive, info, limit):
    try:
        with archive.open(info) as member:
            return member.read(limit)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError, ValueError):
        return None  # bad CRC, unsupported method (e.g. deflate64) or a truncated member


def _library():
    global _lib
    if _lib is None:
        name = ctypes.util.find_library("archive") or "libarchive.so.13"
        lib = ctypes.CDLL(name)
        void, text, size = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
        signatures = {"archive_read_new": ([], void), "archive_read_support_format_all": ([void], ctypes.c_int),
                      "archive_read_support_filter_all": ([void], ctypes.c_int),
                      "archive_read_open_filename": ([void, text, size], ctypes.c_int),
                      "archive_read_open_memory": ([void, void, size], ctypes.c_int),
                      "archive_read_next_header": ([void, ctypes.POINTER(void)], ctypes.c_int),
                      "archive_read_data": ([void, void, size], ctypes.c_ssize_t),
                      "archive_read_free": ([void], ctypes.c_int), "archive_error_string": ([void], text),
                      "archive_entry_pathname": ([void], text), "archive_entry_pathname_utf8": ([void], text),
                      "archive_entry_size": ([void], ctypes.c_int64),
                      "archive_entry_size_is_set": ([void], ctypes.c_int),
                      "archive_entry_filetype": ([void], ctypes.c_uint),
                      "archive_entry_is_encrypted": ([void], ctypes.c_int)}
        for function, (arguments, result) in signatures.items():
            getattr(lib, function).argtypes, getattr(lib, function).restype = arguments, result
        _lib = lib
    return _lib


def _archive_entries(path):
    try:
        lib = _library()
    except OSError as error:
        raise ArchiveError(f"libarchive unavailable: {error}") from error
    handle = lib.archive_read_new()
    try:
        lib.archive_read_support_format_all(handle)
        lib.archive_read_support_filter_all(handle)
        if isinstance(path, (bytes, bytearray)):
            buffer = ctypes.create_string_buffer(bytes(path), len(path))  # kept alive until the handle is freed
            status = lib.archive_read_open_memory(handle, buffer, len(path))
        else:
            status = lib.archive_read_open_filename(handle, os.fsencode(path), 1 << 16)
        if status != 0:
            raise ArchiveError((lib.archive_error_string(handle) or b"cannot open").decode("utf-8", "replace"))
        entry = ctypes.c_void_p()
        while True:
            status = lib.archive_read_next_header(handle, ctypes.byref(entry))
            if status == ARCHIVE_EOF:
                return
            if status < ARCHIVE_WARN:
                raise ArchiveError((lib.archive_error_string(handle) or b"bad header").decode("utf-8", "replace"))
            if lib.archive_entry_filetype(entry) != AE_IFREG or lib.archive_entry_is_encrypted(entry) > 0:
                continue
            size = lib.archive_entry_size(entry) if lib.archive_entry_size_is_set(entry) else None
            yield _entry_name(lib, entry), size, lambda limit: _read_data(lib, handle, limit)
    finally:
        lib.archive_read_free(handle)


def _entry_name(lib, entry):
    name = lib.archive_entry_pathname_utf8(entry)
    if name:
        try:
            return name.decode("utf-8")
        except UnicodeDecodeError:
            pass
    raw = lib.archive_entry_pathname(entry) or b""
    for codec in ("utf-8", "gb18030"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _read_data(lib, handle, limit):
    chunks, size, buffer = [], 0, ctypes.create_string_buffer(1 << 16)
    while size < limit:
        count = lib.archive_read_data(handle, buffer, len(buffer))
        if count < 0:
            return None  # damaged or encrypted data
        if count == 0:
            break
        chunks.append(buffer.raw[:count])
        size += count
    return b"".join(chunks)
