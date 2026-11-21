"""Front matter, section chunks and embedding windows.

Nothing here reads files. Chunk and window boundaries decide the embedding
inputs, and the inputs are the cache keys: changing these rules makes the next
run embed the changed inputs again.

Front matter is parsed with PyYAML and code blocks are found with
markdown-it-py (CommonMark block rules); `require()` reports a missing one as a
fixed code before any note is read.
"""
from __future__ import annotations

import posixpath
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    from markdown_it import MarkdownIt
except ImportError:  # reported by require() as SEARCH_DEPENDENCY_MISSING
    yaml = MarkdownIt = None

from .errors import SearchError

WIKILINK_RE = re.compile(r"!?\[\[([^\[\]\n]+?)\]\]")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# Only the opening of an ATX heading is a regex; the closing `#` run is cut with rstrip, because
# `\s+(.*?)\s*#*\s*$` backtracks cubically on a long run of spaces inside the title.
HEADING_START_RE = re.compile(r"\s{0,3}(#{1,6})\s+")
# A list line that holds only one wiki link (and an optional short note) is a catalog line.
CATALOG_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+!?\[\[[^\[\]\n]+\]\]\s*(?:[-–—:|].*)?$")


def nfc(value):
    return unicodedata.normalize("NFC", value)


def folded(value):
    return unicodedata.normalize("NFC", value).casefold()


def require():
    """Stop with a fixed code before reading notes when PyYAML or markdown-it-py is missing."""
    if yaml is None or MarkdownIt is None:
        raise SearchError("SEARCH_DEPENDENCY_MISSING")


@dataclass
class Page:
    path: Path
    rel: str          # vault-relative POSIX path, actual on-disk spelling
    name: str         # file name without .md, NFC
    aliases: list
    front: str
    body: str
    meta: dict
    decode_failed: bool = False
    # Front matter is not a readable YAML mapping of plain values (see read_front).
    front_failed: bool = False
    # Keys whose value is nested or that are given more than once.
    unread_fields: frozenset = field(default_factory=frozenset)
    not_utf8: bool = False        # the file is not UTF-8 text (or holds NUL), so its fields cannot be trusted
    front_unclosed: bool = False  # the first line is `---` but no `---` line closes it
    # The first line is `---` only after other white space is removed; meta is then empty.
    front_open_unclear: bool = False


@dataclass
class Chunk:
    page: Page
    heading: str
    text: str


# ---------- front matter ----------

def _marker(line):
    """True for a `---` line: the line without its "\\n" is `---` plus spaces or tabs."""
    return line.rstrip("\n").rstrip(" \t") == "---"


def frontmatter_parts(text):
    """(front matter, body, unclosed).

    Lines end only at "\\n" (the caller turns "\\r\\n" into "\\n" first): a lone
    "\\r", U+0085, U+2028 or U+2029 does not end a line, so it cannot make a
    closing `---` line. Only a `---` line (spaces or tabs may follow) closes the
    front matter, and it must start the line: an indented one belongs to a YAML
    block value, as in YAML itself. The first line opens it when it is `---`
    after any white space around it is removed (see opening_unclear). A
    first line `---` with no closing line gives no front matter, the whole text
    as body and unclosed=True, so a caller can tell it from a note without front
    matter.
    """
    lines = text.split("\n")
    lines = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if _marker(lines[i]):
                return "".join(lines[1:i]), "".join(lines[i + 1:]), False
        return "", text, True
    return "", text, False


def opening_unclear(text):
    """True when the first line opens the front matter only after other white space is removed.

    The first line is `---` once all white space around it is removed, but it is
    not a clean `---` line (`---` at the start, then only spaces or tabs): for
    example `---` followed by U+2028 or a no-break space, or ` ---`. Another
    reader may see no front matter there, so the hide rules cannot be checked
    and its fields are not used (pages.read_page leaves meta empty).
    """
    first = text.split("\n", 1)[0]
    return first.strip() == "---" and not _marker(first)


def split_frontmatter(text):
    """(front matter, body); see frontmatter_parts."""
    front, body, _ = frontmatter_parts(text)
    return front, body


class FrontMatterError(ValueError):
    """The front matter is not a YAML mapping of plain values."""


if yaml is not None:
    # The pure Python parser, also where the C extension (libyaml) is installed: the two accept
    # different text (`k: {x:}` only in Python, `k: [foo?bar]` only in C), and one parser keeps the
    # result the same on every machine.
    class _NodeLoader(yaml.SafeLoader):
        """Builds the node tree only; no value is constructed, so every scalar stays the text written."""
    _NodeLoader.yaml_implicit_resolvers = {}


def _yaml_tree(front):
    """The one YAML node of the front matter (None when it holds only comments).

    Raises FrontMatterError for text that is not one YAML document, and for any
    tag (`!!int`, `!x`, `!`), anchor or alias: their effect depends on how a
    YAML library builds values, so the text alone does not say what a field holds.
    """
    try:
        # A tag starts with `!`, an anchor with `&` and an alias with `*`: without any of them the
        # token pass cannot find one, and most front matter is read with one pass.
        if "!" in front or "&" in front or "*" in front:
            loader = _NodeLoader(front)  # the reader checks the characters here, so this is inside try
            try:
                while loader.check_token():
                    if isinstance(loader.get_token(), (yaml.TagToken, yaml.AnchorToken, yaml.AliasToken)):
                        raise FrontMatterError("tag, anchor or alias in front matter")
            finally:
                loader.dispose()
        loader = _NodeLoader(front)
        try:
            return loader.get_single_node()
        finally:
            loader.dispose()
    except FrontMatterError:
        raise
    except Exception:  # any reader, scanner, parser or composer failure, deep nesting included
        raise FrontMatterError("unreadable front matter") from None


def _check_tree(node):
    """Every mapping key at any depth is a scalar, and none is an unquoted `<<` (a merge key)."""
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, yaml.MappingNode):
            for key, value in current.value:
                if not isinstance(key, yaml.ScalarNode):
                    raise FrontMatterError("front matter key is not text")
                if not key.style and key.value == "<<":
                    raise FrontMatterError("merge key in front matter")
                pending.append(value)
        elif isinstance(current, yaml.SequenceNode):
            pending.extend(current.value)


def one_line(value):
    """A value as one line: trimmed, and a value of several lines ("\\n") joined with spaces.

    The one form front matter values and hide rule values are compared and shown in.
    """
    value = value.strip()
    if "\n" in value:
        value = " ".join(line.strip() for line in value.splitlines() if line.strip())
    return value


# Control characters (C0, DEL, C1) other than "\n" and "\t", and the line and paragraph separators.
# YAML ends a line at "\r", U+0085, U+2028 and U+2029, where front matter lines here do not end, so
# such a front matter could hold other fields for YAML than for another reader.
_UNCLEAR_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]")


def read_front(front):
    """(meta, unread keys) of the front matter.

    Only plain YAML is read: scalars, lists and mappings without any tag,
    anchor, alias or merge key, every scalar kept as the text written (`true`,
    `1.10` and `2026-01-02` stay text). meta holds top-level values that are
    text or lists of text (an empty value is an empty list). A key whose value
    is a mapping or holds a nested item, or that is given more than once (after
    trimming and NFC), is left out of meta and named in the unread keys. Raises
    FrontMatterError when the text is not one YAML mapping of that kind, when it
    holds a control character or line separator other than "\\n" and "\\t"
    (a lone "\\r", U+0085, U+2028, U+2029, U+000B, U+000C and the like), and
    when it is not empty but holds only white space. An empty front matter
    (`---` right after `---`) is an empty mapping.
    """
    require()
    if _UNCLEAR_CHAR_RE.search(front):
        raise FrontMatterError("control character or line separator in front matter")
    if not front:
        return {}, frozenset()
    if not front.strip():
        raise FrontMatterError("front matter of white space only")
    # The key-value pairs are read from the node tree, before a dict would keep only the last of
    # two equal keys.
    node = _yaml_tree(front)
    if node is None:
        return {}, frozenset()
    if not isinstance(node, yaml.MappingNode):
        raise FrontMatterError("front matter is not a mapping")
    _check_tree(node)
    pairs = [(key.value, value) for key, value in node.value]
    counts = Counter(nfc(raw_key.strip()) for raw_key, _ in pairs)
    meta, unread = {}, set()
    for raw_key, value in pairs:
        key = nfc(raw_key.strip())
        if counts[key] > 1:
            unread.add(key)
        elif isinstance(value, yaml.ScalarNode):
            meta[key] = one_line(value.value) or []
        elif isinstance(value, yaml.SequenceNode) and all(isinstance(item, yaml.ScalarNode) for item in value.value):
            meta[key] = [one_line(item.value) for item in value.value if one_line(item.value)]
        else:
            unread.add(key)
    return meta, frozenset(unread)


def parse_front(front):
    """Top-level front matter values (see read_front)."""
    return read_front(front)[0]


def as_list(value):
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def field_value(page, key):
    """One front matter value as text. For a list, the first item."""
    value = page.meta.get(nfc(key))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def field_values(page, key):
    return [str(v).strip() for v in as_list(page.meta.get(nfc(key))) if str(v).strip()]


def is_true(value):
    return str(value).strip().lower() == "true"


# ---------- section chunks ----------

def link_text(match):
    if match.group(0).startswith("!"):
        return ""
    inner = match.group(1)
    if "|" in inner:
        return inner.split("|", 1)[1]
    target = re.split(r"[#^]", inner, maxsplit=1)[0].strip()
    return posixpath.basename(target) or inner.lstrip("#^")  # keep the page name, drop the folder


_BLOCK_PARSER = []


def _block_parser():
    if not _BLOCK_PARSER:
        require()
        # Block rules only: code block boundaries do not depend on inline syntax.
        _BLOCK_PARSER.append(MarkdownIt("commonmark").disable(["inline", "text_join"], ignoreInvalid=True))
    return _BLOCK_PARSER[0]


def drop_code_blocks(text):
    """Remove the source lines of fenced and indented code blocks (CommonMark block rules),
    including blocks inside lists and quotes."""
    if "```" not in text and "~~~" not in text and "    " not in text and "\t" not in text:
        return text  # no fence and no indentation: no code block is possible
    lines = text.split("\n")  # markdown-it numbers source lines by "\n"
    dropped = set()
    for token in _block_parser().parse(text):
        if token.type in ("fence", "code_block") and token.map:
            dropped.update(range(*token.map))
    if not dropped:
        return text
    return "\n".join(line for number, line in enumerate(lines) if number not in dropped)


def unwrap_code(text):
    """Drop code blocks; keep the content of inline code without backticks."""
    return INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], drop_code_blocks(text))


def is_thematic_break(line):
    return bool(re.fullmatch(r" {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})", line))


def is_markdown_scaffold(line):
    """Known empty Markdown scaffolding only. Any other symbolic line stays as body."""
    if not line.strip() or is_thematic_break(line):
        return True
    patterns = (
        r" {0,3}(?:>[ \t]*)+",                         # empty blockquote
        r" {0,3}#{1,6}[ \t]*",                        # empty ATX heading
        r" {0,3}(?:`{3,}|~{3,}|=+)[ \t]*",             # empty fence / setext underline
        r" {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]+\[[ xX]\])?[ \t]*",   # empty list item or task
        r" {0,3}\|(?:[ \t]*\|)+[ \t]*",                # empty table row
        r" {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*",
    )
    return any(re.fullmatch(pattern, line) for pattern in patterns)


def chunk_has_body(text):
    return any(not is_markdown_scaffold(line) for line in text.splitlines())


def heading_title(line):
    """The title of an ATX heading line ("" for an empty heading), or None for any other line.

    The trailing whitespace, closing `#` run and whitespace before it are not part of the title.
    """
    match = HEADING_START_RE.match(line)
    if not match:
        return None
    return line[match.end():].rstrip().rstrip("#").rstrip()


def _markdown_links_in_line(line):
    """One line with `[text](target)` replaced by text and `![alt](target)` removed.

    Same matches as the regex `(!?)\\[([^\\]\\n]*)\\]\\((?:[^()\\n]|\\([^()\\n]*\\))+\\)`
    under re.sub, in linear time: every `[` before the same `]` shares that
    `]`, and where the target after `](` ends is computed once per position from
    right to left, so a long line of `[` is not read again for each one.
    """
    n = len(line)
    next_paren = [n] * (n + 1)
    for k in range(n - 1, -1, -1):
        next_paren[k] = k if line[k] in "()" else next_paren[k + 1]
    # closing[k]: index of the `)` that ends a target read from k, or -1 when none does.
    closing = [-1] * (n + 1)
    for k in range(n - 1, -1, -1):
        char = line[k]
        if char == ")":
            closing[k] = k
        elif char == "(":
            m = next_paren[k + 1]
            closing[k] = closing[m + 1] if m < n and line[m] == ")" else -1
        else:
            closing[k] = closing[k + 1]
    out, pos, previous = [], 0, -1
    for j in (index for index, char in enumerate(line) if char == "]"):
        start, previous = max(pos, previous + 1), j
        if j < pos:
            continue
        i = line.find("[", start, j)
        if i < 0 or j + 1 >= n or line[j + 1] != "(":
            continue
        end = closing[j + 2]
        if end <= j + 2:  # no `)` ends the target, or the target is empty
            continue
        image = i > pos and line[i - 1] == "!"
        out.append(line[pos:i - 1 if image else i])
        out.append("" if image else line[i + 1:j])
        pos = end + 1
    out.append(line[pos:])
    return "".join(out)


def drop_markdown_links(text):
    return "\n".join(_markdown_links_in_line(line) if "](" in line else line for line in text.split("\n"))


def make_chunks(page, code_free=None):
    """One chunk per heading section. Link syntax becomes its visible text.

    code_free: page.body without code blocks (drop_code_blocks), when the caller already has it.
    """
    code_free = drop_code_blocks(page.body) if code_free is None else code_free
    body = INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], code_free)
    body = "\n".join(line for line in body.splitlines() if not CATALOG_RE.match(line))
    body = WIKILINK_RE.sub(link_text, body)
    body = drop_markdown_links(body)
    heading, lines, chunks = "", [], []
    for line in body.splitlines():
        title = heading_title(line)
        if title is not None:
            text = "\n".join(lines).strip()
            if chunk_has_body(text):
                chunks.append(Chunk(page, heading, text))
            heading, lines = title, []
        else:
            lines.append("" if is_thematic_break(line) else line)
    text = "\n".join(lines).strip()
    if chunk_has_body(text):
        chunks.append(Chunk(page, heading, text))
    return chunks


# ---------- embedding windows ----------

def retrieval_windows(text, max_chars=500, overlap=100, *, tokenizer=None, prefix="", max_tokens=512):
    """Overlapping windows of one chunk, each within the model's token budget.

    A hit on any window selects the whole chunk's page. With a tokenizer, each
    window (with `prefix`, the title header) is cut so it fits `max_tokens`
    including special tokens, so the model never truncates silently.
    """
    if not 0 <= overlap < max_chars:
        raise ValueError("invalid embedding window overlap")
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if tokenizer is not None:
            def fits(stop):
                ids = tokenizer.encode(prefix + text[start:stop], add_special_tokens=True, truncation=False)
                return len(ids) <= max_tokens
            if not fits(end):
                lo, hi = start + 1, end
                if not fits(lo):
                    raise ValueError("retrieval header or one character exceeds the token budget")
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid - 1
                end = lo
            if not fits(end):
                raise ValueError("retrieval tokenizer budget failed")
        yield text[start:end]
        if end == len(text):
            break
        start = end - min(overlap, end - start - 1)


def retrieval_views(chunks, tokenizer=None, prefix="", max_tokens=512):
    texts, owners = [], []
    for index, chunk in enumerate(chunks):
        header = f"{prefix}{chunk.page.name[:40]}\n" if tokenizer is not None else ""
        for text in retrieval_windows(chunk.text, tokenizer=tokenizer, prefix=header, max_tokens=max_tokens):
            texts.append(text)
            owners.append(index)
    return texts, owners


def retrieval_inputs(chunks, model, prefix):
    """Embedding inputs: prefix + first 40 characters of the page name + window text."""
    limit = min(512, model.max_seq_length)
    if type(limit) is not int or limit < 8:
        raise ValueError("finite retrieval model token budget required")
    views, owners = retrieval_views(chunks, model.tokenizer, prefix, limit)
    return [f"{prefix}{chunks[i].page.name[:40]}\n{text}" for text, i in zip(views, owners)], owners, views
