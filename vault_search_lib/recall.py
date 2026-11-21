"""recall@k with your own question set. Search behaviour is not changed.

Each question names one or more expected pages ("gold"). recall@k is the share
of questions with at least one gold page in the top k candidates (k = 1, 5, 10).
All questions run in one search, so the vault is scanned and embedded once.

A zero denominator is a failure, not 0%: an empty question set, a question
without gold pages, or a gold page that search would not read (outside the
scopes, excluded, empty or missing) stops before any embedding (exit code 2).
A partial scan (a file could not be read) also stops: its ranks would not be
comparable. The report names only question ids, queries, gold pages and their
ranks, never the other candidates.

A gold page is a vault-relative path (`/` or `\\` separators, `.md` optional,
no leading `/`, drive, `.` or `..`; on Windows no `:`). With `.md` it is compared with the whole
file name, so `Note.md` and `Note.MD` are two notes; without it, with the path
without the extension. It names the scanned note with exactly that spelling;
if there is none, the one note whose path is equal after Unicode NFC, and two
such notes stop the run (RECALL_GOLD_AMBIGUOUS). Scoring compares the notes'
whole paths, extension included.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import markdown as md
from . import pages as pg
from . import search
from .errors import SearchError

KS = (1, 5, 10)


class RecallBlocked(Exception):
    def __init__(self, code, detail=""):
        super().__init__(code)
        self.code, self.detail = code, detail


def has_extension(gold):
    return gold.casefold().endswith(".md")


_DRIVE_RE = re.compile(r"[A-Za-z]:")


def gold_key(value):
    """A gold path with '/' separators, as written (with or without .md), or None when it is not a
    vault-relative note path.

    A drive (`C:`) at the start is refused everywhere. On Windows a `:` cannot
    be part of a note's name (it would name a drive or a stream), so any `:` is
    refused there; elsewhere `Meeting: agenda.md` is a valid note name.
    """
    text = value.replace("\\", "/")
    if (text.startswith("/") or "\x00" in text or _DRIVE_RE.match(text)
            or (os.name == "nt" and ":" in text)):
        return None
    stem = text[:-3] if has_extension(text) else text
    if any(part in ("", ".", "..") for part in stem.split("/")):
        return None
    return text


def page_key(page):
    """The result-link form of a scanned page: its vault-relative path, as spelled on disk, without .md."""
    return page.rel[:-3]


def validate(data):
    if not isinstance(data, dict):
        raise RecallBlocked("RECALL_QUESTIONS_INVALID", "top level must be a JSON object")
    scopes, questions = data.get("scopes", []), data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise RecallBlocked("RECALL_QUESTIONS_EMPTY")
    if not isinstance(scopes, list) or not all(isinstance(s, str) and s.strip() for s in scopes):
        raise RecallBlocked("RECALL_SCOPES_INVALID", "scopes must be a list of folder paths")
    seen_ids, seen_queries, out = set(), set(), []
    for item in questions:
        if not isinstance(item, dict):
            raise RecallBlocked("RECALL_QUESTIONS_INVALID", "each question must be an object")
        qid, query, gold = item.get("id"), item.get("query"), item.get("gold")
        if not isinstance(qid, str) or not qid or qid in seen_ids:
            raise RecallBlocked("RECALL_QUESTIONS_INVALID", f"missing or duplicate id: {qid}")
        if (not isinstance(query, str) or not query.strip() or "\n" in query or "\r" in query
                or md.nfc(query.strip()) in seen_queries):
            raise RecallBlocked("RECALL_QUESTIONS_INVALID", f"empty, multi-line or duplicate query: {qid}")
        if not isinstance(gold, list) or not gold or not all(isinstance(g, str) and g.strip() for g in gold):
            raise RecallBlocked("RECALL_GOLD_EMPTY", qid)
        keys = [gold_key(g) for g in gold]
        bad = [g for g, key in zip(gold, keys) if key is None]
        if bad:
            raise RecallBlocked("RECALL_GOLD_INVALID", f"{qid}: {bad[0]}")
        seen_ids.add(qid)
        seen_queries.add(md.nfc(query.strip()))
        out.append({"id": qid, "query": query.strip(), "gold": keys})
    return [s.strip() for s in scopes], out


def load_questions(path, vault=None):
    """vault: the checked root; a question file inside it follows the pages.read_file rules."""
    try:
        raw = pg.read_file(vault, path) if vault is not None else Path(path).read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
    except SearchError as exc:
        raise RecallBlocked("RECALL_QUESTIONS_INVALID", pg.refused_reason(exc.code)) from None
    except (OSError, ValueError):
        raise RecallBlocked("RECALL_QUESTIONS_INVALID", "cannot read the question file as JSON") from None
    return validate(data)


def resolve_gold(questions, pages):
    """Replace each gold path by the vault-relative path (Page.rel) of the one scanned page it names."""
    # [with the extension, without it]: a gold path with `.md` is compared with whole paths only.
    exact, normal = ({}, {}), ({}, {})
    for page in pages:
        for table, key in ((0, page.rel), (1, page_key(page))):
            exact[table].setdefault(key, []).append(page)
            normal[table].setdefault(md.nfc(key), []).append(page)
    outside, ambiguous = [], []
    for question in questions:
        resolved = []
        for gold in question["gold"]:
            table = 0 if has_extension(gold) else 1
            found = exact[table].get(gold) or normal[table].get(md.nfc(gold), [])
            if len(found) == 1:
                resolved.append(found[0].rel)
            else:
                shown = gold if has_extension(gold) else gold + ".md"
                (ambiguous if found else outside).append(f"{question['id']}: {shown}")
        question["gold"] = resolved
    if ambiguous:
        raise RecallBlocked("RECALL_GOLD_AMBIGUOUS", " | ".join(ambiguous))
    if outside:
        raise RecallBlocked("RECALL_GOLD_OUT_OF_SCOPE", " | ".join(outside))


def output_path(vault, value):
    """The --json-out path, checked before the run so a refused path costs no embedding."""
    if not isinstance(value, str) or not value.strip():
        raise RecallBlocked("RECALL_OUTPUT_FAILED", "cannot write the result file")
    try:
        return pg.checked_file(vault, value, must_exist=False, new_file=True)
    except SearchError as exc:
        raise RecallBlocked("RECALL_OUTPUT_FAILED", pg.refused_reason(exc.code)) from None
    except OSError:
        raise RecallBlocked("RECALL_OUTPUT_FAILED", "cannot write the result file") from None


def save(vault, path, result):
    """Write the full result; checked again because the run may have taken minutes.

    The result is encoded before the file is opened, so text that cannot be
    UTF-8 (a lone surrogate in a question id) leaves no file behind.
    """
    try:
        data = json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8")
    except (UnicodeError, ValueError, TypeError):
        raise RecallBlocked("RECALL_OUTPUT_FAILED", "cannot write the result file") from None
    output_path(vault, path)
    try:
        pg.create_file(vault, path, data)
    except SearchError as exc:
        raise RecallBlocked("RECALL_OUTPUT_FAILED", pg.refused_reason(exc.code)) from None
    except OSError:
        raise RecallBlocked("RECALL_OUTPUT_FAILED", "cannot write the result file") from None


def score(questions, ranked, ks=KS):
    """questions[i]["gold"] and ranked[i] hold vault-relative paths with the extension (Page.rel)."""
    rows = []
    for question, links in zip(questions, ranked):
        gold = set(question["gold"])
        ranks = [i for i, item in enumerate(links, 1) if item in gold]
        rows.append({"id": question["id"], "query": question["query"],
                     "gold": [g[:-3] for g in question["gold"]],
                     "first_rank": ranks[0] if ranks else None,
                     "gold_ranks": {g: (links.index(g) + 1 if g in links else None)
                                    for g in question["gold"]}})
    total = len(rows)
    if total == 0:
        raise RecallBlocked("RECALL_QUESTIONS_EMPTY")
    hits = {k: sum(1 for r in rows if r["first_rank"] is not None and r["first_rank"] <= k) for k in ks}
    return {"questions": total, "hits": hits, "recall": {k: hits[k] / total for k in ks}, "rows": rows}


def writable(scopes, questions):
    """RECALL_OUTPUT_FAILED when the question file holds text that cannot be UTF-8 (a lone surrogate).

    The result file repeats the ids, queries, gold paths and scopes, so such a
    file could never be saved; the check runs before any page is read or embedded.
    """
    try:
        json.dumps({"scopes": scopes, "questions": questions}, ensure_ascii=False).encode("utf-8")
    except (UnicodeError, ValueError, TypeError):
        raise RecallBlocked("RECALL_OUTPUT_FAILED", "cannot write the result file") from None


def measure(vault, questions_path, cfg, top=10, loader=None, *, save_result=False):
    """save_result: a result file was asked for, so text it could not hold stops the run first."""
    vault = pg.checked_root(vault)
    scopes, questions = load_questions(questions_path, vault)
    if save_result:
        writable(scopes, questions)
    try:
        pages, problems = pg.read_pages(vault, scopes, cfg)
    except SearchError as exc:
        raise RecallBlocked("RECALL_SCOPE_BLOCKED", exc.code) from None
    if problems:
        raise RecallBlocked("RECALL_SCAN_PARTIAL", f"{len(problems)} read problems")
    resolve_gold(questions, pages)
    try:
        run = search.run_queries(vault, [q["query"] for q in questions], scopes, top, cfg, loader=loader,
                                 scanned=(pages, problems))
    except SearchError as exc:
        raise RecallBlocked("RECALL_SEARCH_BLOCKED", exc.code) from None
    if not run.complete:
        raise RecallBlocked("RECALL_SEARCH_PARTIAL", f"{len(run.problems)} read problems")
    ranked = [[chunk.page.rel for _, chunk in results] for _, results in run.queries]
    result = score(questions, ranked)
    result.update({"scopes": scopes, "top": top, "model_name": cfg.model_name})
    return result


def report(result, write=print):
    total = result["questions"]
    for row in result["rows"]:
        rank = row["first_rank"]
        mark = "HIT" if rank is not None and rank <= max(KS) else "MISS"
        where = f"rank={rank}" if rank is not None else f"rank=>{result['top']}"
        write(f"{mark} {row['id']} {where} query: {row['query']}")
        if rank is None or rank > 1:
            for gold, gold_rank in row["gold_ranks"].items():
                write(f"     gold {gold} -> {gold_rank if gold_rank is not None else 'not in top'}")
    parts = [f"top{k}={result['hits'][k]}/{total} ({result['recall'][k] * 100:.1f}%)" for k in KS]
    write(f"RECALL_COMPLETE questions={total} " + " ".join(parts))
