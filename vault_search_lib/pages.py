"""Safe folder walk, exclusions, scopes and page reading.

Rules that run before any note body is opened:
- the vault root must be a local path with no link or junction on it or above it;
- only `*.md` files are read;
- links, junctions and other reparse points inside the vault are never followed
  (cloud placeholder files such as OneDrive "files on demand" are allowed);
  a link to a folder whose name a folder rule matches is skipped like a folder,
  and a link to a file is a read problem whatever its name;
- a note file with more than one name (a hard link) is refused before it is
  opened, where the file system reports the link count;
- dot folders and dot files (when `skip_dot_dirs`), folders named in
  `skip_dir_names`, and folders matching `exclude_dirs` are skipped;
- a folder or note whose name cannot be written as UTF-8 is a read problem;
- every requested scope (--in) must resolve to a folder that holds Markdown.
The same link and link-count rules apply to the configuration, question and
report files that the CLI opens inside the vault, decided on the real path
(`checked_file`, `read_file`, `create_file`), also when a link outside the
vault leads into it; a report inside the vault is only ever a new file that
is not a note.
"""
from __future__ import annotations

import errno
import functools
import os
import re
import stat
from pathlib import Path, PurePosixPath

from . import markdown as md
from .errors import SearchError

folded = md.folded


def unsafe_reparse(info):
    """True for a symlink or a Windows reparse point other than a cloud placeholder."""
    if stat.S_ISLNK(info.st_mode):
        return True
    if not getattr(info, "st_file_attributes", 0) & 0x400:
        return False
    tag = getattr(info, "st_reparse_tag", None)
    return not (isinstance(tag, int) and (tag & 0xFFFF0FFF) == 0x9000001A and not tag & 0x20000000)


def normalize_extended(value):
    """Use one Windows spelling (no \\\\?\\ prefix) before containment comparisons."""
    text = os.fspath(value)
    if text.startswith("\\\\?\\") and len(text) >= 6 and text[5] == ":":
        return text[4:]
    return text


def reject_unc(value):
    """Search runs on local paths only. A UNC path (\\\\server\\share, \\\\?\\UNC\\...) stops
    before any folder is touched. A local extended path such as \\\\?\\C:\\ is allowed."""
    text = os.fspath(value).replace("/", "\\")
    if not text.startswith("\\\\"):
        return
    for prefix in ("\\\\?\\", "\\\\.\\"):
        if text.startswith(prefix):
            rest = text[len(prefix):]
            if len(rest) >= 2 and rest[0].isalpha() and rest[0].isascii() and rest[1] == ":":
                return
            break
    raise SearchError("SEARCH_ROOT_INVALID")


def checked_root(value):
    """The vault root as one real local path.

    A link or junction on the given folder or on any folder above it is refused
    before resolve() would hide it, so the private-folder and scope checks see
    the same path the files really have. A mapped network drive shows up as UNC
    only after resolve(), so the resolved path is checked again. Where resolve()
    keeps the spelling given (_SPELLING_KEPT), every folder name is then spelled
    as its parent folder lists it (_listed_spelling), so one folder has one
    spelling, also on a volume that ignores case; a folder whose parent cannot
    be listed keeps the spelling given, so there two spellings of one vault
    stay two paths (and two caches). A `~` that cannot be expanded
    (no home folder) is SEARCH_VAULT_NOT_FOUND.
    """
    reject_unc(value)
    try:
        given = Path(os.path.abspath(Path(normalize_extended(os.fspath(value))).expanduser()))
    except RuntimeError:  # Path.expanduser() when the home folder cannot be determined
        raise SearchError("SEARCH_VAULT_NOT_FOUND") from None
    reject_unc(given)
    for current in reversed((given, *given.parents)):
        try:
            info = current.lstat()
        except OSError:
            raise SearchError("SEARCH_VAULT_NOT_FOUND") from None
        if unsafe_reparse(info):
            raise SearchError("SEARCH_ROOT_INVALID")
    try:
        resolved = given.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SearchError("SEARCH_VAULT_NOT_FOUND") from None
    reject_unc(resolved)
    if not resolved.is_dir():
        raise SearchError("SEARCH_VAULT_NOT_FOUND")
    if _SPELLING_KEPT:
        resolved = _listed_spelling(resolved)
    return Path(normalize_extended(resolved))


def _listed_spelling(path):
    """path with each folder name spelled as its parent folder lists it.

    POSIX realpath() keeps the letter case and Unicode form given, so on a
    volume that ignores case (the macOS default) `/x/notes` can be the folder
    listed as `/x/Notes`. A name the parent lists exactly is kept; otherwise the
    one listed name that is equal after NFC and case folding and is the same
    folder (_same_folder: device and file number) is used. A name is kept as
    given when the parent cannot be listed or no listed name is that folder;
    then other letter case of that name gives another path (and another cache).
    """
    current = type(path)(path.anchor)
    for part in path.parts[1:] if path.anchor else path.parts:
        step = current / part
        try:
            names = os.listdir(current)
        except OSError:
            names = None
        if names is not None and part not in names:
            same = [name for name in names if folded(name) == folded(part) and _same_folder(current / name, step)]
            if len(same) == 1:
                step = current / same[0]
        current = step
    return current


# ---------- patterns ----------

def _pattern_parts(pattern):
    text = pattern.replace("\\", "/").strip()
    anchored = text.startswith("/")
    parts = tuple(folded(p) for p in PurePosixPath(text.strip("/")).parts if p not in ("", "."))
    return anchored, parts


@functools.lru_cache(maxsize=1024)
def _component_re(wanted):
    """`*` (any run) and `?` (one character) are the only wildcards; `[`, `]` and every other
    character match themselves, so a folder named `[Old] Notes` is matched by its own name."""
    return re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in wanted), re.S)


def run_matches(parts, pattern):
    """A folder pattern matches a contiguous run of path components (wildcards per component).

    A leading `/` anchors the run at the first component.
    """
    anchored, wanted = _pattern_parts(pattern)
    if not wanted or len(wanted) > len(parts):
        return False
    starts = (0,) if anchored else range(len(parts) - len(wanted) + 1)
    return any(all(_component_re(wanted[j]).fullmatch(parts[i + j]) for j in range(len(wanted)))
               for i in starts)


def excluded(relative, cfg, is_directory=False):
    """Path-only exclusion, decided before any Markdown body or embedding is read."""
    parts = tuple(folded(p) for p in PurePosixPath(relative).parts)
    if not parts:
        return False
    if cfg.skip_dot_dirs and any(part.startswith(".") for part in parts):
        return True
    folders = parts if is_directory else parts[:-1]  # a note's own name is never a folder name
    skip = {folded(name) for name in cfg.skip_dir_names}
    if any(part in skip for part in folders):
        return True
    return any(run_matches(folders, pattern) for pattern in cfg.exclude_dirs)


def private_path(path, cfg, vault, is_directory=False):
    """True for a path under a `private_dirs` folder (or, with is_directory, that folder itself).

    Only folders are compared: a file's own name never matches a folder pattern.
    A pattern without a leading `/` is tested on the whole absolute path, so a
    private folder still counts when the vault root is that folder or a folder
    inside it. A pattern with a leading `/` is anchored at the vault root.
    """
    if not cfg.private_dirs:
        return False
    target = Path(path)
    # A UNC anchor (\\server\share\) is one item of Path.parts; split it so `share` is a component too.
    anchor = [folded(p) for p in target.anchor.replace("\\", "/").strip("/").split("/") if p]
    rest = target.parts[1:] if target.anchor else target.parts
    absolute = tuple(anchor + [folded(p) for p in rest])
    try:
        relative = tuple(folded(p) for p in target.relative_to(vault).parts)
    except ValueError:
        relative = ()
    if not is_directory:
        absolute, relative = absolute[:-1], relative[:-1]
    for pattern in cfg.private_dirs:
        anchored = _pattern_parts(pattern)[0]
        if run_matches(relative if anchored else absolute, pattern):
            return True
    return False


# ---------- walking ----------

def _safe_components(vault, path):
    vault, path = Path(normalize_extended(vault)), Path(normalize_extended(path))
    try:
        parts = path.relative_to(vault).parts
    except ValueError:
        raise SearchError("SEARCH_PATH_REJECTED") from None
    current = vault
    info = None
    for part in (None, *parts):
        if part is not None:
            current /= part
            info = current.lstat()
            if unsafe_reparse(info):
                raise SearchError("SEARCH_PATH_REJECTED")
        else:
            info = current.lstat()
    return info


# ---------- files the CLI opens (configuration, questions, recall result) ----------

# Fixed reasons for a refused file, never with its path.
REFUSED_REASONS = {
    "SEARCH_PATH_REJECTED": "link or special file inside the vault",
    "SEARCH_MULTIPLE_NAMES": "file with more than one name inside the vault",
    "SEARCH_OUTPUT_REFUSED": "existing file or note inside the vault",
    "SEARCH_SOURCE_CHANGED": "file inside the vault changed while it was opened",
}


def refused_reason(code):
    return REFUSED_REASONS.get(code, REFUSED_REASONS["SEARCH_PATH_REJECTED"])


def _single_regular(info):
    """The rule every file opened inside the vault follows: a regular file with one name.

    A second name could sit outside a private or excluded folder (or outside the
    vault), and which name is the original cannot be told, so every name is
    refused. A count of 0 means "not reported".
    """
    if not stat.S_ISREG(info.st_mode):
        raise SearchError("SEARCH_PATH_REJECTED")
    if info.st_nlink > 1:
        raise SearchError("SEARCH_MULTIPLE_NAMES")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read_regular(vault, path, before):
    """Bytes of the file `before` (its lstat) describes; SEARCH_SOURCE_CHANGED when another file is opened."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != _identity(before):
            raise SearchError("SEARCH_SOURCE_CHANGED")
        _single_regular(opened)
        raw = stream.read()
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise SearchError("SEARCH_SOURCE_CHANGED")
    if _identity(_safe_components(vault, path)) != _identity(before) or len(raw) != before.st_size:
        raise SearchError("SEARCH_SOURCE_CHANGED")
    return raw


def _user_path(value):
    text = os.fspath(value)
    if not text.strip() or "\x00" in text:
        raise OSError(errno.EINVAL, "invalid file path")
    try:
        return Path(os.path.abspath(Path(normalize_extended(text)).expanduser()))
    except RuntimeError:  # `~` without a home folder: a file that cannot be opened, like any other
        raise OSError(errno.ENOENT, "home folder not found") from None


_MAX_LINK_HOPS = 40


def _followed(given):
    """The path with every link and junction on it followed, for a path whose end does not exist.

    The parts from the first missing one on are kept as given, so a link whose
    target does not exist yet still says where a new file would be made. None
    when not even the drive exists. A link that loops or cannot be read raises
    OSError: where it leads cannot be told.
    """
    current, rest, hops = Path(given.parts[0]), list(given.parts[1:]), 0
    while rest:
        part = rest.pop(0)
        if part == "..":
            current = current.parent  # current holds no link, so its parent is the real parent
            continue
        step = current / part
        try:
            info = step.lstat()
        except (FileNotFoundError, NotADirectoryError):
            rest.insert(0, part)
            break
        if not unsafe_reparse(info):
            current = step
            continue
        hops += 1
        if hops > _MAX_LINK_HOPS:
            raise OSError(errno.ELOOP, "too many links")
        target = Path(normalize_extended(os.readlink(step)))
        if not target.is_absolute():
            target = current / target
        current, rest = Path(target.parts[0]), list(target.parts[1:]) + rest
    try:
        base = Path(normalize_extended(current.resolve(strict=True)))
    except (OSError, RuntimeError):
        return None
    return base.joinpath(*rest)


def _spellings(given):
    """(lexical, real, spelled, linked) spellings of a path the user gave.

    lexical: the path up to its first link, junction or missing part in the
    spelling the file system keeps (a Windows 8.3 short name or other case
    becomes the real name), then the rest as given; no link is followed.
    real: the path with every link followed (for a missing file or a link to a
    missing file, the existing part followed and the rest as given), or None
    when not even the drive exists. A link that loops or cannot be read raises
    OSError.
    spelled: lexical is in the file system's spelling (False when it could not
    be resolved and is the path as given).
    linked: lexical stopped at a link or junction.
    """
    parts = given.parts
    prefix, rest = Path(parts[0]), list(parts[1:])
    linked = False
    while rest:
        step = prefix / rest[0]
        try:
            info = step.lstat()
        except OSError:
            break
        if unsafe_reparse(info):
            linked = True
            break
        prefix = step
        rest.pop(0)
    try:
        lexical, spelled = Path(normalize_extended(prefix.resolve(strict=True))).joinpath(*rest), True
    except (OSError, RuntimeError):
        lexical, spelled = given, False
    try:
        real = Path(normalize_extended(given.resolve(strict=True)))
    except (OSError, RuntimeError):
        real = _followed(given)
    return lexical, real, spelled, linked


# Windows resolve() gives the spelling the file system keeps. POSIX realpath() keeps the letter case
# that was given, so on a volume that ignores case (the macOS default) `/x/notes` can be the vault
# `/x/Notes`; there the folder at the vault's depth is compared by identity as well.
_SPELLING_KEPT = os.name != "nt"


def _same_folder(path, vault):
    """True when path exists and is the vault folder (same device and file number)."""
    try:
        first, second = os.stat(path), os.stat(vault)
    except (OSError, ValueError):
        return False
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _under(path, vault, exact):
    """path when it is under vault, else None.

    exact: both are in the file system's spelling, so their components compare
    as written (Windows pathlib folds case, which would put a folder `vault` of
    a case-sensitive parent inside a vault `Vault`). Where the spelling given is
    kept (_SPELLING_KEPT), a path whose folder at the vault's depth is the vault
    folder itself is under it too, and comes back spelled from the vault down;
    a file that does not exist yet is decided by that existing folder.
    """
    parts, root = path.parts, vault.parts
    if exact:
        if len(parts) > len(root) and parts[:len(root)] == root:
            return path
    elif path != vault and path.is_relative_to(vault):
        return path
    if _SPELLING_KEPT and len(parts) > len(root) and _same_folder(type(path)(*parts[:len(root)]), vault):
        return vault.joinpath(*parts[len(root):])
    return None


def _inside_info(vault, path, must_exist):
    """lstat of a path inside the vault after the vault rules, or None for a file that does not exist yet."""
    try:
        info = _safe_components(vault, path)
    except FileNotFoundError:
        if must_exist:
            raise
        if path.parent != vault:
            try:
                parent = _safe_components(vault, path.parent)
            except FileNotFoundError:
                return None  # writing will fail with an OSError
            if not stat.S_ISDIR(parent.st_mode):
                raise NotADirectoryError(path.parent.name) from None
        return None
    _single_regular(info)
    return info


def _locate(vault, value, must_exist):
    """(path to open, inside the vault, lstat or None).

    A path is inside the vault when its lexical or its real spelling is, so
    neither a short name nor a link that leads out of the vault skips the vault
    rules. A path inside the vault by either spelling that goes through a link
    or junction anywhere (also outside the vault, also a link to a file that
    does not exist yet) is refused before it is opened. Inside the vault an
    existing file must be a regular file with one name.
    """
    vault = Path(normalize_extended(vault))
    given = _user_path(value)
    lexical, real, spelled, linked = _spellings(given)
    candidates = [(lexical, spelled)]
    if real is not None and os.fspath(real) != os.fspath(lexical):
        candidates.append((real, True))
    inside = [found for found in (_under(path, vault, exact) for path, exact in candidates) if found is not None]
    if not inside:
        return given, False, None
    if linked:
        raise SearchError("SEARCH_PATH_REJECTED")
    info = None
    for path in inside:
        info = _inside_info(vault, path, must_exist)
    chosen, exact = (real, True) if real is not None else (lexical, spelled)
    return _under(chosen, vault, exact) or chosen, True, info


def _refuse_existing(path, info):
    """A result file inside the vault must be new and must not be a note (or a stream of another file)."""
    name = path.name.rstrip(". ").casefold()
    if info is not None or name.endswith(".md") or (os.name == "nt" and ":" in path.name):
        raise SearchError("SEARCH_OUTPUT_REFUSED")


def checked_file(vault, value, must_exist=True, new_file=False):
    """A file path the user named for the CLI to open (configuration, questions, report).

    Outside the vault the path is the user's own choice and is used as given.
    Inside the vault (by the given or the real spelling, see _locate) no
    component may be a link, junction or other reparse point (the same rule as
    for notes; SEARCH_PATH_REJECTED), and an existing file must be a regular
    file with one name (SEARCH_PATH_REJECTED, SEARCH_MULTIPLE_NAMES). With
    new_file, a path inside the vault must not exist yet and must not end in
    `.md` (SEARCH_OUTPUT_REFUSED). A missing file raises FileNotFoundError when
    must_exist; without must_exist the folder it would go in is checked instead
    (NotADirectoryError when that is a file). An empty path, one with NUL and
    a `~` that cannot be expanded (no home folder) are an OSError.
    """
    path, inside, info = _locate(vault, value, must_exist)
    if inside and new_file:
        _refuse_existing(path, info)
    return path


def read_file(vault, value):
    """Bytes of a configuration or question file, by the checked_file rules.

    Inside the vault the opened file must be the one that was checked.
    """
    path, inside, info = _locate(vault, value, must_exist=True)
    if not inside:
        return Path(path).read_bytes()
    return _read_regular(vault, path, info)


def create_file(vault, value, data):
    """Write data to the recall result file, by the checked_file rules with new_file.

    Inside the vault the file is created exclusively, so a file that appeared
    after the check is not replaced. Outside the vault an existing file is replaced.
    """
    path, inside, info = _locate(vault, value, must_exist=False)
    if not inside:
        Path(path).write_bytes(data)
        return
    _refuse_existing(path, info)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o666)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _name_ok(name):
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: the name cannot become a cache key or a link
        return False
    return True


def _link_to_directory(path, info):
    """True when the link or junction at path (info: its lstat) is known to lead to a folder.

    Windows keeps the kind on the link itself (the directory attribute of a
    directory link or a junction); elsewhere only the target's metadata says it.
    The target is never opened. False for a link to a file and for one whose
    target cannot be read.
    """
    attributes = getattr(info, "st_file_attributes", None)
    if attributes is not None:
        return bool(attributes & stat.FILE_ATTRIBUTE_DIRECTORY)
    try:
        return stat.S_ISDIR(os.stat(path).st_mode)
    except OSError:
        return False


def _walk(vault, start, cfg, directories=False, problem_paths=None):
    """Markdown files (or folders) under start. Problems are counted, never printed with paths."""
    candidates, problems = [], []
    pending = [start]
    while pending:
        directory = pending.pop()
        if directories and private_path(directory, cfg, vault, is_directory=True):
            continue
        try:
            _safe_components(vault, directory)
            with os.scandir(directory) as entries:
                items = sorted(entries, key=lambda item: folded(item.name))
        except OSError:
            problems.append("DIRECTORY_READ_FAILED")
            if problem_paths is not None:
                problem_paths.append(directory)
            continue
        except SearchError:
            problems.append("PATH_REJECTED")
            if problem_paths is not None:
                problem_paths.append(directory)
            continue
        def folder_rule(path, relative):
            """A folder at path is skipped (excluded, or private while looking for a folder name)."""
            return (excluded(relative, cfg, is_directory=True)
                    or (directories and private_path(path, cfg, vault, is_directory=True)))

        for item in items:
            path = Path(item.path)
            relative = path.relative_to(vault).as_posix()
            try:
                looks_dir = item.is_dir(follow_symlinks=False)
            except OSError:
                looks_dir = False
            if folder_rule(path, relative) if looks_dir else excluded(relative, cfg):
                continue
            try:
                info = item.stat(follow_symlinks=False)
            except OSError:
                problems.append("METADATA_READ_FAILED")
                if problem_paths is not None:
                    problem_paths.append(path)
                continue
            is_dir = stat.S_ISDIR(info.st_mode)
            if is_dir and not looks_dir and folder_rule(path, relative):
                continue
            # Never descend through a link or junction, and never open its target.
            if unsafe_reparse(info):
                # Folder rules apply to a link only when it is known to lead to a folder; a link to a
                # file, or of a kind that cannot be told, is a read problem like any other link.
                if folder_rule(path, relative) and _link_to_directory(path, info):
                    continue
                problems.append("PATH_REJECTED")
                if problem_paths is not None:
                    problem_paths.append(path)
                continue
            is_note = stat.S_ISREG(info.st_mode) and path.suffix.casefold() == ".md"
            # In folder mode a note is not a candidate, so its name cannot affect which folder is found.
            if (is_dir or (is_note and not directories)) and not _name_ok(item.name):
                problems.append("NAME_UNENCODABLE")
                if problem_paths is not None:
                    problem_paths.append(path)
                continue
            if is_dir:
                pending.append(path)
                if directories:
                    candidates.append(path)
            elif is_note:
                if not directories:
                    candidates.append(path)
    return candidates, problems


def scope_prefix(vault, value):
    """One spelling of a --in value: vault-relative, folded, ending in '/'."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise SearchError("SEARCH_SCOPE_INVALID")
    vault = Path(normalize_extended(vault))
    reject_unc(value.strip())
    requested = Path(normalize_extended(value.strip()).replace("\\", "/"))
    if requested.root and not requested.drive and os.name == "nt":
        raise SearchError("SEARCH_SCOPE_INVALID")
    if requested.is_absolute():
        if ".." in requested.parts:
            raise SearchError("SEARCH_SCOPE_INVALID")
        # Compared in the file system's spelling, component by component, as for the CLI's files:
        # Windows pathlib folds case, which would put a folder `vault` of a case-sensitive parent
        # inside a vault `Vault`. A link on the path is kept as a name and refused later.
        try:
            lexical, _, spelled, _ = _spellings(requested)
        except OSError:
            raise SearchError("SEARCH_SCOPE_INVALID") from None
        if (lexical.parts == vault.parts) if spelled else (lexical == vault):
            return ""  # the vault root; refused by the caller
        inside = _under(lexical, vault, spelled)
        if inside is None:
            raise SearchError("SEARCH_SCOPE_INVALID")
        requested = PurePosixPath(*inside.parts[len(vault.parts):])
    if any(part == ".." or ":" in part for part in requested.parts):
        raise SearchError("SEARCH_SCOPE_INVALID")
    prefix = folded(requested.as_posix()).strip("/")
    return "" if prefix in ("", ".") else prefix + "/"


class _BrokenScope(Exception):
    """A requested folder whose name cannot be written as UTF-8 (a read problem, not a refusal)."""

    def __init__(self, path):
        super().__init__("NAME_UNENCODABLE")
        self.path = path


def _exact_scope(vault, prefix, cfg):
    """Resolve each component by name (case and Unicode form folded); never guess from files.

    Raises _BrokenScope when a component's name fails the same UTF-8 check as the walk.
    """
    current = vault
    for part in PurePosixPath(prefix.strip("/")).parts:
        if excluded((current / part).relative_to(vault).as_posix(), cfg, is_directory=True):
            raise SearchError("SEARCH_SCOPE_EMPTY")
        try:
            with os.scandir(current) as entries:
                matches = [Path(item.path) for item in entries if folded(item.name) == part]
        except OSError:
            raise SearchError("SEARCH_SCOPE_UNRESOLVED") from None
        if len(matches) > 1:
            raise SearchError("SEARCH_SCOPE_AMBIGUOUS")
        if not matches:
            return None
        current = matches[0]
        if not _name_ok(current.name):
            raise _BrokenScope(current)
        try:
            info = _safe_components(vault, current)
        except OSError:
            raise SearchError("SEARCH_SCOPE_UNRESOLVED") from None
        if not stat.S_ISDIR(info.st_mode):
            raise SearchError("SEARCH_SCOPE_EMPTY")
    return current


def resolve_scopes(vault, within, cfg, resolved_aliases=None, problems=None, problem_paths=None):
    """Exact vault-relative paths first. A missing one-component name may match one unique folder.

    Absolute or multi-component values never fall back to a name match, and a
    name never matches a folder under `private_dirs` (give its full path instead).
    A requested folder whose name cannot be written as UTF-8 is left out and
    counted in problems (NAME_UNENCODABLE) once, however many scopes stop at
    it; the other scopes go on, and when no
    scope is left the result is empty, never the whole vault.
    """
    vault = Path(normalize_extended(vault))
    roots, broken_at = [], set()
    for value in within or ():
        prefix = scope_prefix(vault, value)
        if prefix == "":
            raise SearchError("SEARCH_SCOPE_INVALID")  # the vault root is the unscoped search
        try:
            root = _exact_scope(vault, prefix, cfg)
        except _BrokenScope as broken:
            # Scopes that stop at the same broken folder (`bad` twice, `bad` and `bad/sub`): one problem.
            if os.fspath(broken.path) not in broken_at:
                broken_at.add(os.fspath(broken.path))
                if problems is not None:
                    problems.append("NAME_UNENCODABLE")
                if problem_paths is not None:
                    problem_paths.append(broken.path)
            continue
        if root is None:
            requested = Path(value.strip().replace("\\", "/"))
            if requested.is_absolute() or len(requested.parts) != 1:
                raise SearchError("SEARCH_SCOPE_EMPTY")
            # Its own list: `problems` collects the caller's read problems of the requested scopes.
            directories, name_problems = _walk(vault, vault, cfg, directories=True)
            if name_problems:  # cannot prove uniqueness across folders that could not be read
                raise SearchError("SEARCH_SCOPE_UNRESOLVED")
            matches = [path for path in directories
                       if not private_path(path, cfg, vault, is_directory=True)
                       and folded(path.name) == prefix.strip("/")]
            if len(matches) > 1:
                raise SearchError("SEARCH_SCOPE_AMBIGUOUS")
            if not matches:
                raise SearchError("SEARCH_SCOPE_EMPTY")
            root = matches[0]
            if resolved_aliases is not None:
                resolved_aliases.append(root.relative_to(vault).as_posix())
        if excluded(root.relative_to(vault).as_posix(), cfg, is_directory=True):
            raise SearchError("SEARCH_SCOPE_EMPTY")
        roots.append(root)
    if not roots and not within:
        roots = [vault]
    return sorted(set(roots), key=lambda path: (len(path.parts), folded(path.as_posix())))


def select_paths(vault, within, cfg, resolved_scopes=None, failed_paths=None, resolved_aliases=None):
    # Keyed by the exact spelling: Windows paths compare case-insensitively, so a set of Path
    # objects would merge two notes of a case-sensitive folder that differ only in case.
    paths, problems, failures_at = {}, [], []
    roots = resolve_scopes(vault, within, cfg, resolved_aliases, problems, failures_at)
    for root in roots:
        if any(root != other and root.is_relative_to(other) for other in roots):
            continue
        walk_paths = []
        found, failures = _walk(vault, root, cfg, problem_paths=walk_paths)
        for path in found:
            paths.setdefault(os.fspath(path), path)
        # A broken folder named with --in is found again by the walk of a parent scope: one problem.
        known = {os.fspath(path) for path in failures_at}
        for code, path in zip(failures, walk_paths):
            if os.fspath(path) not in known:
                problems.append(code)
                failures_at.append(path)
    paths = list(paths.values())
    if within and any(not any(path.is_relative_to(root) for path in paths)
                      and not any(path.is_relative_to(root) or root.is_relative_to(path) for path in failures_at)
                      for root in roots):
        raise SearchError("SEARCH_SCOPE_EMPTY")
    if resolved_scopes is not None:
        resolved_scopes.extend(roots)
    if failed_paths is not None:
        failed_paths.extend(failures_at)

    def order(path):
        # Names that differ only in Unicode form share the folded key; the spelling breaks the tie.
        relative = path.relative_to(vault).as_posix()
        return folded(relative), relative
    return sorted(paths, key=order), problems


# ---------- reading ----------

def read_page(vault, path):
    before = _safe_components(vault, path)
    _single_regular(before)  # refused before the file is opened
    raw = _read_regular(vault, path, before)
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = raw.decode(encoding)
        decode_failed = False
    except UnicodeError:
        text = raw.decode(encoding, errors="replace")
        decode_failed = True
    # Front matter rules are read only from UTF-8 text. UTF-16 is still read for search, and UTF-16
    # without a byte order mark can pass as UTF-8 with NUL between the letters.
    not_utf8 = encoding != "utf-8-sig" or decode_failed or "\x00" in text
    # The front matter lines end only at "\n"; a lone "\r" in the body becomes a line end afterwards.
    text = md.nfc(text.replace("\r\n", "\n"))
    front, body, unclosed = md.frontmatter_parts(text)
    body = body.replace("\r", "\n")
    open_unclear = md.opening_unclear(text)
    try:
        meta, unread = md.read_front(front)
        front_failed = False
    except md.FrontMatterError:
        meta, unread, front_failed = {}, frozenset(), True
    if open_unclear:
        # Another reader may see no front matter here, so its fields are used for nothing (labels, exact
        # terms, pin_field), with hide rules or without; the body is still searched.
        meta, unread = {}, frozenset()
    aliases = [md.nfc(a) for a in md.as_list(meta.get("aliases") or meta.get("alias")) if a]
    return md.Page(path, path.relative_to(vault).as_posix(), md.nfc(path.stem), aliases, front, body, meta,
                   decode_failed, front_failed, unread, not_utf8, unclosed, open_unclear)


def is_empty(page):
    return not (page.front.strip() or page.body.strip())


class Pages(list):
    """Pages plus how the scopes were resolved."""
    resolved_aliases = ()
    resolved_scopes = ()


def read_pages(vault, within, cfg):
    """Pages under the scopes, and a list of read problem codes (no paths)."""
    md.require()
    scopes, failures_at, aliases = [], [], []
    paths, problems = select_paths(vault, within, cfg, scopes, failures_at, aliases)
    pages = Pages()
    pages.resolved_aliases = tuple(dict.fromkeys(aliases))
    pages.resolved_scopes = tuple(scopes)
    for path in paths:
        try:
            page = read_page(vault, path)
        except (OSError, SearchError) as exc:
            multiple = isinstance(exc, SearchError) and exc.code == "SEARCH_MULTIPLE_NAMES"
            problems.append("MULTIPLE_NAMES" if multiple else "MARKDOWN_READ_FAILED")
            failures_at.append(path)
            continue
        if page.decode_failed:
            problems.append("MARKDOWN_DECODE_FAILED")
            failures_at.append(path)
        if not is_empty(page):
            pages.append(page)
    if within and any(not any(page.path.is_relative_to(root) for page in pages)
                      and not any(path.is_relative_to(root) or root.is_relative_to(path) for path in failures_at)
                      for root in scopes):
        raise SearchError("SEARCH_SCOPE_EMPTY")
    return pages, problems
