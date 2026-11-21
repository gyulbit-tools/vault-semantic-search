#!/usr/bin/env python3
"""vault-semantic-search: local semantic search for a folder of Markdown notes.

  python -I -B vault_search.py search VAULT "question" [more questions] [--in FOLDER] [--top 10]
  python -I -B vault_search.py warm VAULT
  python -I -B vault_search.py recall VAULT --questions questions.json [--json-out result.json]

Exit codes: 0 done (including SEARCH_PARTIAL / WARM_PARTIAL, which say so in
their output, and --help), 2 blocked (a fixed code in one JSON line, or
RECALL_BLOCKED). Invalid arguments (including a --top out of range) are
SEARCH_ARGS_INVALID, without the arguments themselves. Set VAULT_SEARCH_DEBUG=1 to print the Python traceback
of an unexpected error to stderr.
"""
import os
import sys

if sys.flags.isolated:  # -I leaves the script folder off sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402

from vault_search_lib import config as config_mod  # noqa: E402
from vault_search_lib import markdown as markdown_mod  # noqa: E402
from vault_search_lib import pages as pages_mod  # noqa: E402
from vault_search_lib import recall as recall_mod  # noqa: E402
from vault_search_lib import search as search_mod  # noqa: E402
from vault_search_lib.errors import SearchError  # noqa: E402


def _streams():
    """pythonw (a scheduled warm run) has no console: sys.stdout and sys.stderr are None."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    out, err = sys.stdout, sys.stderr

    def write(text):
        if out is not None:
            print(text, file=out, flush=True)

    def log(text):
        if err is not None:
            print(text, file=err, flush=True)
    return write, log


def clean_arg(value):
    # PowerShell passes "C:\folder\" with the closing quote inside the value.
    return value.strip().rstrip('"').strip() if value else value


class ArgsInvalid(Exception):
    """argparse's own message names the bad value, which may be a question; it is not printed."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgsInvalid()


def top_in(low, high):
    """--top as an int from low to high; any other value is an argument error (SEARCH_ARGS_INVALID)."""
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError("out of range")
        return number
    return parse


def parser():
    ap = Parser(prog="vault_search.py", description="Local semantic search for Markdown notes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search", help="find pages whose sections are close in meaning to a question")
    p.add_argument("vault", help="vault folder")
    p.add_argument("query", nargs="+", help="question(s); several run in one scan and one model load")
    p.add_argument("--top", type=top_in(1, 100), default=10, help="candidates per question, 1-100 (default 10)")
    p.add_argument("--in", dest="within", action="append", metavar="FOLDER",
                   help="search only this folder (vault-relative path, or a unique folder name); repeatable")
    p.add_argument("--output", choices=("auto", "summary", "paths", "details"), default="auto",
                   help="details: links, section titles and snippets; paths: links with similarity and "
                        "label fields, no titles or snippets; summary: one JSON line of counts")
    p.add_argument("--no-snippet", action="store_true", help="leave out snippets and section titles")
    p.add_argument("--config", help="configuration file (JSON)")
    p = sub.add_parser("warm", help="embed the whole vault into the cache (for a nightly scheduled run)")
    p.add_argument("vault", help="vault folder")
    p.add_argument("--config", help="configuration file (JSON)")
    p = sub.add_parser("recall", help="measure recall@1/5/10 with your own question set")
    p.add_argument("vault", help="vault folder")
    p.add_argument("--questions", required=True, help="question set JSON (see README)")
    p.add_argument("--top", type=top_in(max(recall_mod.KS), 100), default=10,
                   help="candidates per question, 10-100 (default 10)")
    p.add_argument("--json-out", help="also write the full result to this JSON file")
    p.add_argument("--config", help="configuration file (JSON)")
    return ap


def main(argv=None, loader=None):
    write, log = _streams()
    try:
        args = parser().parse_args(argv)
    except ArgsInvalid:
        write(json.dumps({"status": "SEARCH_BLOCKED", "code": "SEARCH_ARGS_INVALID"}, sort_keys=True))
        return 2
    except SystemExit as exc:  # --help
        return exc.code if isinstance(exc.code, int) else 0
    args.vault = clean_arg(args.vault)
    if getattr(args, "within", None):
        args.within = [clean_arg(s) for s in args.within]
    try:
        try:
            # The root check (UNC, links on or above the vault) runs before the vault is looked into
            # for a config file, so a refused root is never touched.
            vault_root = pages_mod.checked_root(args.vault)
            markdown_mod.require()
        except SearchError as exc:
            if args.cmd == "recall":
                write(f"RECALL_BLOCKED code={exc.code}")
            else:
                write(json.dumps({"status": "SEARCH_BLOCKED", "code": exc.code}, sort_keys=True))
            return 2
        try:
            cfg = config_mod.load(vault=vault_root, path=args.config)
        except SearchError as exc:
            payload = {"status": "SEARCH_BLOCKED", "code": exc.code}
            if exc.detail:
                payload["detail"] = exc.detail
            write(json.dumps(payload, sort_keys=True))
            return 2
        if args.cmd == "search":
            search_mod.search_command(args, cfg, write=write, loader=loader, log=log)
        elif args.cmd == "warm":
            search_mod.warm_command(args, cfg, write=write, log=log, loader=loader)
        else:
            try:
                # An empty --json-out is a requested report too: it is refused, never skipped.
                if args.json_out is not None:
                    recall_mod.output_path(vault_root, args.json_out)
                result = recall_mod.measure(args.vault, args.questions, cfg, args.top, loader=loader,
                                            save_result=args.json_out is not None)
                # The numbers are printed only after the requested report is saved.
                if args.json_out is not None:
                    recall_mod.save(vault_root, args.json_out, result)
            except recall_mod.RecallBlocked as exc:
                write(f"RECALL_BLOCKED code={exc.code}" + (f" detail={exc.detail}" if exc.detail else ""))
                return 2
            except SearchError as exc:
                write(f"RECALL_BLOCKED code={exc.code}")
                return 2
            recall_mod.report(result, write=write)
        return 0
    except SearchError as exc:
        write(json.dumps({"status": "SEARCH_BLOCKED", "code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        if os.environ.get("VAULT_SEARCH_DEBUG") == "1":
            log(traceback.format_exc())
        write(json.dumps({"status": "SEARCH_BLOCKED", "code": "SEARCH_FAILED"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
