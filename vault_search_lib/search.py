"""Exact-term plus semantic retrieval, result output and cache warming.

Search reads only the selected Markdown files, never writes notes, and writes
only its own embedding cache. Results are candidates: the count is a top-k
limit, not a total, and an empty or short list is not evidence of absence.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import unicodedata
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

from . import embedding
from . import markdown as md
from . import pages as pg
from .errors import SearchError

folded = md.folded


# ---------- hiding ----------

PUBLIC, HIDDEN, UNSURE = "public", "hidden", "unsure"


def privacy_state(page, cfg, vault):
    """How much of a page may be shown, decided once from everything read about it.

    HIDDEN (a private folder, or a rule matched): link and labels only; no
    snippet, no section title, no front matter terms.
    UNSURE (only with hide rules; the rules cannot clear the page): link only.
    Its labels are not shown and its front matter is used neither for exact
    terms nor for pin_field. A page is UNSURE when it is not UTF-8 text (or
    holds NUL), its first line is `---` only after other white space is removed
    (md.opening_unclear), its front matter has no closing `---` line or is not
    a YAML mapping of plain values (md.read_front: no tag, anchor, alias or
    merge key, no control character or line separator but "\\n" and "\\t",
    not white space only), or a rule field is nested or given twice. UNSURE
    wins over a private folder. Without hide rules such a page is not UNSURE,
    but a front matter that is unreadable or opened by an unclear first line
    gives it no fields (pages.read_page), so it has no labels or front matter terms.
    PUBLIC: everything.
    Note values and rule values both compare in one form: md.one_line (trimmed,
    several lines joined with spaces), then NFC and casefold.
    """
    if cfg.hide_snippets_when and (page.not_utf8 or page.front_unclosed or page.front_open_unclear
                                   or page.front_failed or any(
            md.nfc(rule.field) in page.unread_fields for rule in cfg.hide_snippets_when)):
        return UNSURE
    if pg.private_path(page.path, cfg, vault):
        return HIDDEN

    def same(items):
        return {folded(md.one_line(str(item))) for item in items}
    for rule in cfg.hide_snippets_when:
        values = same(md.field_values(page, rule.field))
        if rule.allow is not None:
            if not values & same(rule.allow):
                return HIDDEN
        elif values & same(rule.values):
            return HIDDEN
    return PUBLIC


def hidden_page(page, cfg, vault):
    """No snippet, no section title and no front matter terms for this page (HIDDEN or UNSURE)."""
    return privacy_state(page, cfg, vault) != PUBLIC


# ---------- exact terms ----------

def exact_terms(value):
    """NFC/casefold word equality; never substrings, stemming or inferred aliases."""
    return {term.casefold() for term in re.findall(r"\w+(?:[-.]\w+)*", unicodedata.normalize("NFC", value))
            if len(term) > 1 and not term.isnumeric()}


def page_identity(page):
    return pg.normalize_extended(page.path)


def page_exact_terms(page, cfg, hidden):
    values = [page.name]
    if not hidden:
        values.extend(page.aliases)
        for key in cfg.term_fields:
            values.extend(md.field_values(page, key))
    return set().union(*(exact_terms(value) for value in values))


def merge_candidates(query, pages, semantic, top, cfg, state_of, bodies=None):
    """Put exact title and front matter term matches first, without changing the embeddings.

    Exact matches take at most half of the slots; the semantic order is kept for
    the rest. A term that appears in one fifth or more of the pages is ignored;
    a page's body counts without its code blocks. A page with `pin_field: true`
    leads among exact matches (not an UNSURE page). An exact-only match has no
    similarity score and no snippet. state_of(page) is privacy_state; bodies
    maps id(page) to its body without code blocks (computed here when missing).
    """
    def body_of(page):
        if bodies is not None and id(page) in bodies:
            return bodies[id(page)]
        return md.drop_code_blocks(page.body)
    states = {id(page): state_of(page) for page in pages}
    surfaces = [(page, page_exact_terms(page, cfg, states[id(page)] != PUBLIC)) for page in pages]
    frequency = Counter(term for page, tokens in surfaces
                        for term in tokens | (set() if states[id(page)] != PUBLIC else exact_terms(body_of(page))))
    terms = {term for term in exact_terms(query) if frequency[term] * 5 < len(pages)}
    by_page = {page_identity(chunk.page): (score, chunk) for score, chunk in semantic}
    exact = []
    for page, tokens in surfaces:
        matches = terms & tokens
        if not matches:
            continue
        score, source = by_page.get(page_identity(page), (None, md.Chunk(page, "", "")))
        if source.page is not page:
            score, source = None, md.Chunk(page, "", "")
        chunk = md.Chunk(page, source.heading, source.text)
        chunk.search_channel = "exact_terms"
        pinned = (bool(cfg.pin_field) and states[id(page)] != UNSURE
                  and md.is_true(md.field_value(page, cfg.pin_field)))
        exact.append((not pinned, -len(matches), -(score if score is not None else -2),
                      folded(page.rel), page.rel, score, chunk))
    exact.sort(key=lambda row: row[:5])
    exact_limit = top // 2 if semantic else max(1, top // 2)
    results = [(row[5], row[6]) for row in exact[:exact_limit]]
    seen = {page_identity(chunk.page) for _, chunk in results}
    results.extend((score, chunk) for score, chunk in semantic if page_identity(chunk.page) not in seen)
    return results[:top]


# ---------- pipeline ----------

@dataclass
class SearchRun:
    vault: object
    pages: list
    chunk_count: int
    problems: list
    queries: list = field(default_factory=list)   # [(query, [(score or None, chunk), ...])]
    new_inputs: int = 0
    input_count: int = 0

    @property
    def complete(self):
        return not self.problems


def _check_backend(modules):
    if any(importlib.util.find_spec(module) is None for module in modules):
        raise SearchError("SEARCH_DEPENDENCY_MISSING")
    try:
        import numpy  # noqa: F401
    except (ImportError, OSError):
        raise SearchError("SEARCH_DEPENDENCY_MISSING") from None


def _cache_path(vault, cfg):
    """The vault's cache file, or None when its folder cannot be named (embed then treats every save as failed)."""
    try:
        return embedding.cache_file(vault, cfg)
    except OSError:
        return None


def run_queries(vault, queries, within, top, cfg, loader=None, log=None, scanned=None, require_save=False,
                warn=None):
    """Scan once, embed once, then rank every query. `loader(cfg)` returns the encoder.

    scanned: the (pages, problems) pair of pages.read_pages for the same vault and scopes.
    Only a complete whole-vault scan (no --in, no read problem) drops cached
    vectors, also when it finds no page or no section (then no model is needed);
    a scan with no page is SEARCH_SCOPE_EMPTY afterwards when there are queries.
    require_save: a failed final cache save stops the run (warm); otherwise search
    goes on with the vectors in memory and the warning goes to warn. A cache
    folder that cannot be named counts as a failed save, also on a run with no
    section that keeps the cache as it is.
    """
    # Only the built-in ONNX loader needs onnxruntime and tokenizers; another loader needs numpy.
    backend = embedding.BACKEND_MODULES if loader in (None, embedding.load_encoder) else ("numpy",)
    loader = loader or embedding.load_encoder
    pages, problems = scanned if scanned is not None else pg.read_pages(vault, within, cfg)
    problems = list(problems)
    prune = not within and not problems
    # Code blocks are dropped once per page; chunks and the exact-term frequency use the same text.
    bodies = {id(page): md.drop_code_blocks(page.body) for page in pages}
    chunks = [chunk for page in pages for chunk in md.make_chunks(page, bodies[id(page)])]
    run = SearchRun(vault, pages, len(chunks), problems)
    states = {}

    def state_of(page):
        if id(page) not in states:
            states[id(page)] = privacy_state(page, cfg, vault)
        return states[id(page)]

    if not chunks:
        if prune:
            _check_backend(("numpy",))
            embedding.embed(_cache_path(vault, cfg), cfg.model_name, [], None, prune=True, log=log,
                            require_save=require_save, warn=warn)
        elif _cache_path(vault, cfg) is None:
            # Nothing is saved and the kept vectors stay, but a cache folder that cannot be named is
            # reported as on a run with sections: warm stops, search warns.
            if require_save:
                raise SearchError("SEARCH_CACHE_WRITE_FAILED")
            if warn or log:
                (warn or log)(embedding.SAVE_WARNING)
        run.queries = [(q, merge_candidates(q, pages, [], top, cfg, state_of, bodies)) for q in queries]
        if not problems and queries and not any(results for _, results in run.queries):
            raise SearchError("SEARCH_SCOPE_EMPTY")
        return run
    _check_backend(backend)
    import numpy as np
    holder = {}

    def get_model():
        if "model" not in holder:
            holder["model"] = loader(cfg)
        return holder["model"]
    model = get_model()
    texts, owners, views = md.retrieval_inputs(chunks, model, cfg.passage_prefix)
    vectors, run.new_inputs = embedding.embed(_cache_path(vault, cfg), cfg.model_name, texts,
                                              get_model, prune=prune, log=log, require_save=require_save,
                                              warn=warn)
    run.input_count = len(texts)
    if not queries:
        return run
    # The same input is embedded once: a repeated question, or a question whose input is a window's input
    # (then the window's vector is used). Every question keeps its own result rows, in order.
    window_row = {}
    for row, text in enumerate(texts):
        window_row.setdefault(text, row)
    inputs = list(dict.fromkeys(cfg.query_prefix + q for q in queries))
    vector_of = {text: vectors[window_row[text]] for text in inputs if text in window_row}
    inputs = [text for text in inputs if text not in vector_of]
    if inputs:
        raw = get_model().encode(inputs, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True)
        # The rule a window's vector follows (finite, unit length, the same width), else SEARCH_FAILED;
        # ranked with the float16 rows that passed it, as a window's vector is.
        checked = embedding.checked_vectors(np, raw, len(inputs), vectors.shape[1])
        vector_of.update(zip(inputs, checked.astype("float32")))
    query_vectors = [vector_of[cfg.query_prefix + q] for q in queries]
    for query, vector in zip(queries, query_vectors):
        similarities = vectors @ vector
        results, seen = [], set()
        for index in np.argsort(-similarities, kind="stable"):
            source = chunks[owners[int(index)]]
            chunk = md.Chunk(source.page, source.heading, views[int(index)])
            if page_identity(chunk.page) in seen:
                continue
            seen.add(page_identity(chunk.page))
            results.append((float(similarities[int(index)]), chunk))
            if len(results) >= top:
                break
        run.queries.append((query, merge_candidates(query, pages, results, top, cfg, state_of, bodies)))
    return run


# ---------- output ----------

def output_mode(requested, cfg, vault, scopes):
    if requested not in ("auto", "summary", "paths", "details"):
        raise SearchError("SEARCH_OUTPUT_INVALID")
    bounded = bool(scopes) and all(scope != vault for scope in scopes)
    if requested == "auto":
        return "details" if bounded or not cfg.require_scope_for_paths else "summary"
    if requested in ("paths", "details") and cfg.require_scope_for_paths and not bounded:
        raise SearchError("SEARCH_SCOPE_REQUIRED")
    return requested


def link(page, with_extension=False):
    return f"[[{page.rel if with_extension else page.rel[:-3]}]]"


def extension_links(pages):
    """The vault-relative paths whose link keeps the file extension.

    A link leaves out the extension, so `Note.md` and `Note.MD` would both be
    `[[Note]]`. Every searched page whose link would equal another searched
    page's link keeps its extension, also when the link with the extension
    (`[[Note.md]]`) would equal the plain link of a page named `Note.md.md`.
    Decided on all searched pages, so a link reads the same whatever --top is.
    """
    full, changed = set(), True
    while changed:
        shown = Counter(page.rel if page.rel in full else page.rel[:-3] for page in pages)
        clashing = {page.rel for page in pages if shown[page.rel if page.rel in full else page.rel[:-3]] > 1}
        changed = not clashing <= full
        full |= clashing
    return full


def label(page, cfg):
    parts = []
    for name in cfg.label_fields:
        value = ", ".join(md.field_values(page, name))
        if value:
            parts.append(f"{name}: {value}")
    return parts


def emit(run, mode, cfg, top, no_snippet=False, write=print):
    status = "SEARCH_COMPLETE" if run.complete else "SEARCH_PARTIAL"
    aliases = getattr(run.pages, "resolved_aliases", ())
    full = extension_links(run.pages) if mode != "summary" else set()
    for index, (query, results) in enumerate(run.queries, 1):
        if mode == "summary":
            write(json.dumps({"status": status, "query_index": index, "returned_candidates": len(results),
                              "searched_pages": len(run.pages), "searched_chunks": run.chunk_count,
                              "coverage_complete": run.complete, "read_problem_count": len(run.problems),
                              "output": "summary", "candidate_limit": top,
                              "absence_inference_allowed": False, "scope_alias_resolved": bool(aliases),
                              "retrieval": "exact_terms_plus_semantic_top_k"}, sort_keys=True))
            continue
        if aliases:
            write("scope_alias_resolved: " + " | ".join(aliases))
        write(f"query: {query}")
        write(f"{status}: coverage_complete={str(run.complete).lower()}, candidates={len(results)}, "
              f"chunks={run.chunk_count}, read_problems={len(run.problems)}")
        write("Candidates are exact-term matches plus the top semantic matches. "
              "The count is a limit, not a total, and is not evidence of absence.")
        hide_all = no_snippet or mode == "paths"
        for rank, (score, chunk) in enumerate(results, 1):
            state = privacy_state(chunk.page, cfg, run.vault)
            hidden = state != PUBLIC
            where = f" › {chunk.heading}" if chunk.heading and not hide_all and not hidden else ""
            notes = []
            if getattr(chunk, "search_channel", "") == "exact_terms":
                notes.append("exact term")
            notes.append(f"similarity {score:.2f}" if score is not None else "no similarity")
            if state != UNSURE:
                notes.extend(label(chunk.page, cfg))
            write(f"{rank}. {link(chunk.page, chunk.page.rel in full)}{where} ({', '.join(notes)})")
            if hide_all:
                continue
            if hidden:
                write("   (snippet hidden: private folder or front matter rule)")
                continue
            snippet = " / ".join(line.strip() for line in chunk.text.splitlines() if line.strip())
            write(f"   {snippet if len(snippet) <= 200 else snippet[:200] + ' …'}")


def _quiet():
    """Library chatter is not an output surface; only emit() prints results."""
    sink = open(os.devnull, "w", encoding="utf-8")
    return sink, redirect_stdout(sink), redirect_stderr(sink)


def search_command(args, cfg, write=print, loader=None, log=None):
    """log: where a cache save warning goes (the CLI passes stderr); library output stays quiet."""
    if type(args.top) is not int or not 1 <= args.top <= 100:
        raise SearchError("SEARCH_TOP_INVALID")  # the CLI parser already refuses it; kept for direct callers
    vault = pg.checked_root(args.vault)
    within = list(args.within or [])
    scopes = pg.resolve_scopes(vault, within, cfg)
    mode = output_mode(args.output, cfg, vault, scopes if within else [])  # stop before reading bodies
    sink, out, err = _quiet()
    with sink, out, err:
        run = run_queries(vault, args.query, within, args.top, cfg, loader=loader, warn=log)
    emit(run, mode, cfg, args.top, no_snippet=args.no_snippet, write=write)
    return run


def warm_command(args, cfg, write=print, log=None, loader=None):
    """Embed every chunk of the whole vault into the cache and drop unused vectors.

    A cache that cannot be saved, or whose folder cannot be named, stops with SEARCH_CACHE_WRITE_FAILED (exit code 2),
    so a scheduled run without a console still reports the failure.
    """
    vault = pg.checked_root(args.vault)
    sink, out, err = _quiet()
    with sink, out, err:
        run = run_queries(vault, [], [], 1, cfg, loader=loader, log=log, require_save=True)
    write(json.dumps({"status": "WARM_COMPLETE" if run.complete else "WARM_PARTIAL",
                      "searched_pages": len(run.pages), "searched_chunks": run.chunk_count,
                      "embedding_inputs": run.input_count, "new_inputs": run.new_inputs,
                      "coverage_complete": run.complete, "read_problem_count": len(run.problems)},
                     sort_keys=True))
    return run
