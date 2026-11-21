"""recall@k: counting, and the zero-denominator rules (fake encoder)."""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from support import FRONT, FakeLoader, VaultCase, write_vault

from vault_search_lib import pages


class Recall(VaultCase):
    files = {**{f"Alpha/p{n:02d}.md": FRONT + "lorem ipsum dolor sit amet " * 3 + "\n"
                for n in range(1, 9)},
             **{f"Beta/p{n:02d}.md": FRONT + "lorem ipsum dolor sit amet " * 3 + "\n"
                for n in range(9, 13)},
             "Archive/p99.md": FRONT + "archived page\n",
             "Other/p50.md": FRONT + "outside the scopes\n"}
    settings = {"exclude_dirs": ["Archive"]}

    def questions(self, items, scopes=("Alpha", "Beta")):
        path = self.temp / "questions.json"
        data = {"questions": items}
        if scopes is not None:
            data["scopes"] = list(scopes)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def run_recall(self, path, *extra):
        return self.run_cli("recall", str(self.vault), "--questions", str(path), *extra)

    def test_counts_top_k(self):
        # All bodies are equal, so the semantic order is the page order; an exact title word goes first.
        path = self.questions([
            {"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]},
            {"id": "q2", "query": "lorem ipsum", "gold": ["Alpha/p03.md"]},
            {"id": "q3", "query": "dolor sit", "gold": ["Beta/p09.md"]},
            {"id": "q4", "query": "amet", "gold": ["Beta/p12.md"]},
        ])
        out = self.temp / "result.json"
        code, text = self.run_recall(path, "--json-out", str(out))
        self.assertEqual(code, 0, text)
        self.assertIn("RECALL_COMPLETE questions=4 top1=1/4 (25.0%) top5=2/4 (50.0%) top10=3/4 (75.0%)", text)
        self.assertIn("HIT q1 rank=1", text)
        self.assertIn("HIT q2 rank=3", text)
        self.assertIn("HIT q3 rank=9", text)
        self.assertIn("MISS q4 rank=>10", text)
        self.assertEqual(self.loader.loads, 1)
        result = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(result["hits"], {"1": 1, "5": 2, "10": 3})
        self.assertEqual(result["rows"][3]["gold_ranks"], {"Beta/p12.md": None})

    def test_report_names_only_gold_pages(self):
        path = self.questions([{"id": "q1", "query": "lorem", "gold": ["Beta/p10.md"]}])
        code, text = self.run_recall(path)
        self.assertEqual(code, 0, text)
        self.assertNotIn("p01", text)

    def test_scopes_are_optional(self):
        path = self.questions([{"id": "q1", "query": "outside the scopes", "gold": ["Other/p50.md"]}], scopes=None)
        code, text = self.run_recall(path)
        self.assertEqual(code, 0, text)
        self.assertIn("top1=1/1", text)

    def test_zero_denominator_is_a_failure(self):
        cases = [([], "RECALL_BLOCKED code=RECALL_QUESTIONS_EMPTY"),
                 ([{"id": "q1", "query": "x", "gold": []}], "RECALL_BLOCKED code=RECALL_GOLD_EMPTY detail=q1")]
        for gold in ("Archive/p99.md", "Other/p50.md", "Alpha/missing.md", ".hidden/p.md"):
            cases.append(([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]},
                           {"id": "q2", "query": "lost", "gold": [gold]}],
                          f"RECALL_BLOCKED code=RECALL_GOLD_OUT_OF_SCOPE detail=q2: {gold}"))
        for items, expected in cases:
            with self.subTest(expected=expected):
                self.loader.loads = 0
                code, text = self.run_recall(self.questions(items))
                self.assertEqual(code, 2, text)
                self.assertIn(expected, text)
                self.assertNotIn("RECALL_COMPLETE", text)
                self.assertNotIn("0.0%", text)
                self.assertEqual(self.loader.loads, 0, "no model load before the denominator stands")

    def test_invalid_question_files(self):
        for items in ([{"id": "q1", "query": "a"}, {"id": "q1", "query": "b", "gold": ["Alpha/p01.md"]}],
                      [{"id": "q1", "query": "line\nbreak", "gold": ["Alpha/p01.md"]}],
                      [{"id": "q1", "query": "same", "gold": ["Alpha/p01.md"]},
                       {"id": "q2", "query": "same", "gold": ["Alpha/p02.md"]}]):
            with self.subTest(items=items):
                code, text = self.run_recall(self.questions(items))
                self.assertEqual(code, 2, text)
                self.assertTrue(text.startswith("RECALL_BLOCKED code=RECALL_"), text)
        broken = self.temp / "broken.json"
        broken.write_text("{", encoding="utf-8")
        code, text = self.run_recall(broken)
        self.assertEqual(code, 2)
        self.assertIn("RECALL_QUESTIONS_INVALID", text)

    def test_partial_scan_blocks(self):
        (self.vault / "Alpha" / "bad.md").write_bytes(b"caf\xe9")
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]))
        self.assertEqual(code, 2, text)
        self.assertIn("RECALL_BLOCKED code=RECALL_SCAN_PARTIAL", text)

    def test_top_must_cover_k(self):
        path = self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}])
        for top in ("5", "101"):
            with self.subTest(top=top):
                code, text = self.run_recall(path, "--top", top)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(text), {"code": "SEARCH_ARGS_INVALID", "status": "SEARCH_BLOCKED"})
        for top in ("10", "100"):
            with self.subTest(top=top):
                code, text = self.run_recall(path, "--top", top)
                self.assertEqual(code, 0, text)

    def test_numbers_come_after_the_report_is_saved(self):
        folder = self.temp / "a folder"
        folder.mkdir()
        for target in (folder, self.vault / "Alpha" / "p01.md" / "out.json"):
            with self.subTest(target=target.name):
                code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]),
                                             "--json-out", str(target))
                self.assertEqual(code, 2, text)
                self.assertEqual(text.strip(),
                                 "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file")

    def test_result_inside_the_vault_is_only_a_new_file(self):
        path = self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}])
        note = self.vault / "Alpha" / "p01.md"
        before = note.read_bytes()
        existing = self.vault / "Alpha" / "old.json"
        existing.write_text("ORIGINAL", encoding="utf-8")
        targets = [note, self.vault / "Alpha" / "New.MD", existing]
        if os.name == "nt":
            targets.append(self.vault / "Alpha" / "p02.md:stream")  # a stream would change the note file
        for target in targets:
            with self.subTest(target=target.name):
                code, text = self.run_recall(path, "--json-out", str(target))
                self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED "
                                                           "detail=existing file or note inside the vault"))
        self.assertEqual(note.read_bytes(), before)
        self.assertFalse((self.vault / "Alpha" / "New.MD").exists())
        self.assertEqual(existing.read_text(encoding="utf-8"), "ORIGINAL")
        self.assertEqual(self.loader.loads, 0)
        code, text = self.run_recall(path, "--json-out", str(self.vault / "Alpha" / "result.json"))
        self.assertEqual(code, 0, text)
        outside = self.temp / "outside.json"
        outside.write_text("ORIGINAL", encoding="utf-8")
        code, text = self.run_recall(path, "--json-out", str(outside))
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads(outside.read_text(encoding="utf-8"))["questions"], 1)

    def test_result_file_that_appears_during_the_run_is_kept(self):
        late = self.vault / "Alpha" / "late.json"

        class LateLoader(FakeLoader):
            def __call__(self, cfg):
                late.write_text("ORIGINAL", encoding="utf-8")
                return super().__call__(cfg)
        path = self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}])
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(path), "--json-out", str(late),
                                  loader=LateLoader())
        self.assertEqual(code, 2, text)
        self.assertIn("RECALL_BLOCKED code=RECALL_OUTPUT_FAILED", text)
        self.assertEqual(late.read_text(encoding="utf-8"), "ORIGINAL")
        # Between the last check and the write: the exclusive create refuses it too.
        later = self.vault / "Alpha" / "later.json"
        real, checks = pages._refuse_existing, []

        def appear(target, info):
            real(target, info)
            checks.append(target)
            if len(checks) == 3:  # before the run, again in save, and last in create_file
                later.write_text("ORIGINAL", encoding="utf-8")
        with mock.patch.object(pages, "_refuse_existing", side_effect=appear):
            code, text = self.run_recall(path, "--json-out", str(later))
        self.assertEqual((code, text.strip()),
                         (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file"))
        self.assertEqual(later.read_text(encoding="utf-8"), "ORIGINAL")

    def test_empty_result_path_is_refused(self):
        path = self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}])
        for value in ("", "   "):
            with self.subTest(value=value):
                code, text = self.run_recall(path, "--json-out", value)
                self.assertEqual((code, text.strip()),
                                 (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file"))
        self.assertEqual(self.loader.loads, 0)

    def test_result_that_cannot_be_utf8_leaves_no_file(self):
        path = self.temp / "questions.json"
        path.write_text('{"scopes": ["Alpha"], "questions": [{"id": "q\\udc80", "query": "p05", '
                        '"gold": ["Alpha/p05.md"]}]}', encoding="utf-8")
        out = self.temp / "out.json"
        code, text = self.run_recall(path, "--json-out", str(out))
        self.assertEqual((code, text.strip()),
                         (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file"))
        self.assertFalse(out.exists())
        code, text = self.run_recall(path)
        self.assertEqual(code, 0, text)

    def test_question_that_cannot_be_utf8_stops_before_the_model_when_a_result_is_asked_for(self):
        path = self.temp / "questions.json"
        path.write_text('{"scopes": ["Alpha"], "questions": [{"id": "q1", "query": "lorem \\ud800 ipsum", '
                        '"gold": ["Alpha/p05.md"]}]}', encoding="utf-8")
        out = self.temp / "out.json"
        code, text = self.run_recall(path, "--json-out", str(out))
        self.assertEqual((code, text.strip()),
                         (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file"))
        self.assertFalse(out.exists())
        self.assertEqual(self.loader.loads, 0)
        good = self.questions([{"id": "q1", "query": "로렘 입숨 lorem", "gold": ["Alpha/p05.md"]}])
        code, text = self.run_recall(good, "--json-out", str(out))
        self.assertEqual(code, 0, text)
        self.assertIn("RECALL_COMPLETE questions=1", text)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["rows"][0]["query"], "로렘 입숨 lorem")

    def test_gold_with_another_extension_spelling_is_out_of_scope(self):
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.MD"]}]))
        self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_GOLD_OUT_OF_SCOPE detail=q1: Alpha/p05.MD"))
        self.assertEqual(self.loader.loads, 0)
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]))
        self.assertEqual(code, 0, text)

    @unittest.skipUnless(os.name == "nt", "alternate data streams are an NTFS feature")
    def test_refused_stream_is_not_made(self):
        path = self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}])
        stream = str(self.vault / "Alpha" / "p02.md") + ":stream"
        code, text = self.run_recall(path, "--json-out", stream)
        self.assertEqual(code, 2, text)
        with self.assertRaises(FileNotFoundError):
            open(stream, "rb").close()

    def test_named_scope_order_does_not_hide_a_read_problem(self):
        try:
            write_vault(self.vault, {"Beta/Only here/p13.md": FRONT + "lorem ipsum\n",
                                     "Private/bad\udc80 dir/x.md": FRONT + "x\n"})
        except (OSError, UnicodeError):
            self.skipTest("cannot create a name with a lone surrogate here")
        config = self.config_file(private_dirs=["Private"])
        for scopes in (("Only here", "Private/bad\udc80 dir"), ("Private/bad\udc80 dir", "Only here")):
            with self.subTest(first=scopes[0][:8]):
                path = self.temp / "questions.json"
                path.write_text(json.dumps({"scopes": list(scopes), "questions": [  # \udc80 escaped as JSON
                    {"id": "q1", "query": "lorem", "gold": ["Beta/Only here/p13.md"]}]}), encoding="utf-8")
                code, text = self.run_cli("recall", str(self.vault), "--questions", str(path), config=config)
                self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_SCAN_PARTIAL "
                                                           "detail=1 read problems"))
        self.assertEqual(self.loader.loads, 0)

    def test_gold_with_extension_tells_note_md_from_note_MD(self):
        from types import SimpleNamespace
        from vault_search_lib import markdown as md
        from vault_search_lib import recall
        root = pages.checked_root(self.vault)
        lower = md.Page(root / "Alpha" / "Note.md", "Alpha/Note.md", "Note", [], "", "x", {})
        upper = md.Page(root / "Alpha" / "Note.MD", "Alpha/Note.MD", "Note", [], "", "x", {})
        ranked = [(0.9, md.Chunk(upper, "", "x")), (0.8, md.Chunk(lower, "", "x"))]
        calls = []

        def run_queries(vault, queries, *args, **kwargs):
            calls.append(queries)
            return SimpleNamespace(complete=True, problems=[], queries=[(q, ranked) for q in queries])
        path = self.questions([{"id": "q1", "query": "one", "gold": ["Alpha/Note.md"]},
                               {"id": "q2", "query": "two", "gold": ["Alpha/Note.MD"]}])
        with mock.patch.object(recall.pg, "read_pages", return_value=([lower, upper], [])), \
                mock.patch.object(recall.search, "run_queries", side_effect=run_queries):
            result = recall.measure(self.vault, path, self.cfg())
            self.assertEqual([row["first_rank"] for row in result["rows"]], [2, 1])
            self.assertEqual([row["gold_ranks"] for row in result["rows"]],
                             [{"Alpha/Note.md": 2}, {"Alpha/Note.MD": 1}])
            self.assertEqual([row["gold"] for row in result["rows"]], [["Alpha/Note"], ["Alpha/Note"]])
            calls.clear()
            with self.assertRaises(recall.RecallBlocked) as caught:
                recall.measure(self.vault, self.questions([{"id": "q1", "query": "one", "gold": ["Alpha/Note"]}]),
                               self.cfg())
            self.assertEqual((caught.exception.code, caught.exception.detail),
                             ("RECALL_GOLD_AMBIGUOUS", "q1: Alpha/Note.md"))
            self.assertEqual(calls, [], "no search before the denominator stands")

    def test_gold_must_be_a_vault_relative_path(self):
        for gold in ("/Alpha/p05.md", "C:/vault/Alpha/p05.md", "Alpha/../Alpha/p05.md", "Alpha//p05.md",
                     "./Alpha/p05.md"):
            with self.subTest(gold=gold):
                self.loader.loads = 0
                code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": [gold]}]))
                self.assertEqual(code, 2, text)
                self.assertIn(f"RECALL_BLOCKED code=RECALL_GOLD_INVALID detail=q1: {gold}", text)
                self.assertEqual(self.loader.loads, 0)

    def test_colon_in_a_gold_path(self):
        from vault_search_lib import recall

        def gold(value):
            return recall.validate({"questions": [{"id": "q1", "query": "x", "gold": [value]}]})[1][0]["gold"]
        with mock.patch("os.name", "posix"):  # a `:` is a valid name character there
            self.assertEqual(gold("Projects/Meeting: agenda.md"), ["Projects/Meeting: agenda.md"])
            self.assertEqual(gold("Projects\\a:b"), ["Projects/a:b"])
            for value in ("C:/vault/Alpha/p05.md", "c:notes.md"):
                with self.subTest(value=value), self.assertRaises(recall.RecallBlocked) as caught:
                    gold(value)
                self.assertEqual(caught.exception.code, "RECALL_GOLD_INVALID")
        with mock.patch("os.name", "nt"):  # a drive or a stream there
            with self.assertRaises(recall.RecallBlocked) as caught:
                gold("Projects/Meeting: agenda.md")
            self.assertEqual(caught.exception.code, "RECALL_GOLD_INVALID")

    def test_gold_names_the_note_with_its_own_spelling(self):
        write_vault(self.vault, {"Alpha/Caf\u00e9.md": FRONT + "lorem ipsum dolor sit amet\n",
                                 "Alpha/Cafe\u0301.md": FRONT + "quokka marmalade sandwiches at noon\n"})
        if len([p for p in (self.vault / "Alpha").iterdir() if p.stem.startswith("Caf")]) < 2:
            self.skipTest("this file system merges the two Unicode forms")
        code, text = self.run_recall(self.questions([
            {"id": "q1", "query": "quokka marmalade sandwiches", "gold": ["Alpha/Caf\u00e9.md"]}]))
        self.assertEqual(code, 0, text)
        self.assertNotIn("HIT q1 rank=1 ", text)
        self.assertIn("RECALL_COMPLETE questions=1 top1=0/1", text)

    def test_gold_equal_to_two_notes_after_nfc_is_ambiguous(self):
        write_vault(self.vault, {"Alpha/\u01d5.md": FRONT + "composed name\n",
                                 "Alpha/U\u0308\u0304.md": FRONT + "decomposed name\n"})
        if len([p for p in (self.vault / "Alpha").iterdir() if p.stem.startswith(("\u01d5", "U\u0308"))]) < 2:
            self.skipTest("this file system merges the two Unicode forms")
        mixed = "Alpha/\u00dc\u0304.md"
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "name", "gold": [mixed]}]))
        self.assertEqual(code, 2, text)
        self.assertIn(f"RECALL_BLOCKED code=RECALL_GOLD_AMBIGUOUS detail=q1: {mixed}", text)
        self.assertEqual(self.loader.loads, 0)
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "name", "gold": ["Alpha/\u01d5.md"]}]))
        self.assertEqual(code, 0, text)

    def test_hard_link_makes_the_scan_partial(self):
        try:
            os.link(self.vault / "Alpha" / "p01.md", self.vault / "Beta" / "twin.md")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a hard link here")
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]))
        self.assertEqual(code, 2, text)
        self.assertIn("RECALL_BLOCKED code=RECALL_SCAN_PARTIAL detail=2 read problems", text)

    def test_link_named_like_an_excluded_folder_does_not_block(self):
        outside = self.temp / "outside"
        write_vault(outside, {"p77.md": FRONT + "outside page\n"})
        try:
            os.symlink(outside, self.vault / "Alpha" / "Archive", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a directory symlink here")
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]))
        self.assertEqual(code, 0, text)
        self.assertIn("RECALL_COMPLETE questions=1", text)
        os.symlink(outside, self.vault / "Alpha" / "Elsewhere", target_is_directory=True)
        code, text = self.run_recall(self.questions([{"id": "q1", "query": "p05", "gold": ["Alpha/p05.md"]}]))
        self.assertIn("RECALL_BLOCKED code=RECALL_SCAN_PARTIAL", text)

    def test_example_question_file_is_valid(self):
        from support import ROOT
        from vault_search_lib import recall
        scopes, questions = recall.load_questions(ROOT / "examples" / "recall-questions.example.json")
        self.assertTrue(questions)
        for question in questions:
            for gold in question["gold"]:
                self.assertNotIn(gold.rsplit("/", 1)[-1].casefold(), question["query"].casefold())


if __name__ == "__main__":
    unittest.main()
