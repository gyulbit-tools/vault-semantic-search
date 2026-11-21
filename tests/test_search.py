"""End-to-end search, warm and output rules on a synthetic vault with a fake encoder."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from support import FRONT, FakeEncoder, FakeLoader, VaultCase, WithoutLibyaml, _environ, cli, write_vault

from vault_search_lib import embedding, pages
from vault_search_lib.errors import SearchError


def links(text):
    return [line.split("]]", 1)[0].split("[[", 1)[1] for line in text.splitlines()
            if line[:1].isdigit() and "[[" in line]


class SearchOutput(VaultCase):
    def test_semantic_match_comes_first_with_heading_and_snippet(self):
        code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_COMPLETE: coverage_complete=true", text)
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")
        self.assertIn("› Timer", text)
        self.assertIn("twenty minutes every morning", text)

    def test_excluded_dot_and_empty_files_are_never_results(self):
        code, text = self.run_cli("search", str(self.vault), "old drip line timer twenty minutes", "--top", "20")
        self.assertEqual(code, 0, text)
        found = links(text)
        self.assertNotIn("Archive/Old irrigation", found)
        self.assertNotIn(".obsidian/workspace", found)
        self.assertNotIn("Projects/Empty", found)
        self.assertEqual(len(found), 7)

    def test_private_folder_hides_snippet_and_heading_but_keeps_link(self):
        code, text = self.run_cli("search", str(self.vault), "woke up at three and slept badly")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Private/Health log")
        first = text.split("1. ", 1)[1].split("\n2. ", 1)[0]
        self.assertNotIn("› Sleep", first)
        self.assertNotIn("Slept badly", text)
        self.assertIn("snippet hidden", first)

    def test_hide_rule_by_field_value(self):
        code, text = self.run_cli("search", str(self.vault), "greenhouse watering schedule", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertNotIn("greenhouse", text.split("Candidates are", 1)[1])
        self.assertIn("balcony plants", text)

    def test_allow_rule_hides_pages_without_an_allowed_value(self):
        config = self.config_file(hide_snippets_when=[{"field": "sensitivity", "allow": ["low"]}])
        code, text = self.run_cli("search", str(self.vault), "brake pads cable", config=config)
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Projects/Bicycle repair")
        self.assertNotIn("brake pads", text.split("Candidates are", 1)[1])
        self.assertIn("balcony plants", text)

    def test_exact_title_term_is_promoted(self):
        code, text = self.run_cli("search", str(self.vault), "bicycle", "--top", "5")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Projects/Bicycle repair")
        first = next(line for line in text.splitlines() if line.startswith("1. "))
        self.assertIn("exact term", first)

    def test_front_matter_term_field_is_promoted(self):
        code, text = self.run_cli("search", str(self.vault), "sprinkler", "--top", "4")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")
        config = self.config_file(term_fields=[])
        code, text = self.run_cli("search", str(self.vault), "sprinkler", "--top", "4", config=config)
        self.assertNotIn("exact term", text)

    def test_hidden_page_front_matter_terms_are_not_used(self):
        files = {"A/One.md": "---\nkeywords: [zephyrine]\nsensitivity: high\n---\nplain text about rivers and boats\n",
                 **{f"B/Filler {n}.md": FRONT + f"filler page number {n} about mountains\n" for n in range(8)}}
        write_vault(self.vault, files)
        code, text = self.run_cli("search", str(self.vault), "zephyrine", "--top", "3")
        self.assertEqual(code, 0, text)
        self.assertNotIn("exact term", text)

    def test_pin_field_leads_exact_matches(self):
        files = {"Pins/Alpha lantern.md": FRONT + "a lantern note\n",
                 "Pins/Beta lantern.md": "---\nfeatured: true\n---\nanother lantern note\n",
                 **{f"Pins/Filler {n}.md": FRONT + f"filler page number {n} about rivers\n" for n in range(6)}}
        write_vault(self.vault, files)
        config = self.config_file(pin_field="featured")
        code, text = self.run_cli("search", str(self.vault), "lantern", "--top", "4", config=config)
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[:2], ["Pins/Beta lantern", "Pins/Alpha lantern"])

    def test_korean_query(self):
        code, text = self.run_cli("search", str(self.vault), "김치찌개에 두부를 넣고 끓이는 방법")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Recipes/Kimchi stew")

    def test_paths_and_no_snippet_leave_out_text(self):
        for option in (["--output", "paths"], ["--no-snippet"]):
            with self.subTest(option=option):
                code, text = self.run_cli("search", str(self.vault), "drip line morning", *option)
                self.assertEqual(code, 0, text)
                self.assertNotIn("twenty minutes", text)
                self.assertNotIn("›", text)
                self.assertTrue(links(text))

    def test_label_fields(self):
        config = self.config_file(label_fields=["type"])
        code, text = self.run_cli("search", str(self.vault), "drip line morning", config=config)
        self.assertEqual(code, 0, text)
        self.assertIn("type: project", text)

    def test_summary_is_one_json_line_per_query(self):
        code, text = self.run_cli("search", str(self.vault), "drip", "brakes", "--output", "summary")
        self.assertEqual(code, 0, text)
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([r["query_index"] for r in rows], [1, 2])
        self.assertTrue(all(r["status"] == "SEARCH_COMPLETE" and r["absence_inference_allowed"] is False
                            for r in rows))
        self.assertNotIn("Garden", text)

    def test_require_scope_for_paths(self):
        config = self.config_file(require_scope_for_paths=True)
        code, text = self.run_cli("search", str(self.vault), "drip", config=config)
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads(text.splitlines()[0])["output"], "summary")
        self.loader.loads = 0
        code, text = self.run_cli("search", str(self.vault), "drip", "--output", "paths", config=config)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text), {"code": "SEARCH_SCOPE_REQUIRED", "status": "SEARCH_BLOCKED"})
        self.assertEqual(self.loader.loads, 0, "a blocked output mode stops before the model loads")
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", "Projects", config=config)
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")

    def test_several_queries_load_the_model_once(self):
        code, text = self.run_cli("search", str(self.vault), "drip", "brakes", "train to the coast")
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("query: "), 3)
        self.assertEqual(self.loader.loads, 1)
        self.assertEqual(len(self.loader.passage_batches()), 1)

    def test_top_limit(self):
        code, text = self.run_cli("search", str(self.vault), "drip", "--top", "2")
        self.assertEqual(code, 0, text)
        self.assertEqual(len(links(text)), 2)
        for top in ("0", "101"):
            with self.subTest(top=top):
                code, text = self.run_cli("search", str(self.vault), "drip", "--top", top)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(text), {"code": "SEARCH_ARGS_INVALID", "status": "SEARCH_BLOCKED"})
        code, text = self.run_cli("search", str(self.vault), "drip", "--top", "100")
        self.assertEqual(code, 0, text)
        code, text = self.run_cli("search", str(self.vault), "drip", "--top", "1")
        self.assertEqual((code, len(links(text))), (0, 1), text)

    def test_search_command_keeps_its_own_top_check(self):
        from types import SimpleNamespace
        from vault_search_lib import search
        args = SimpleNamespace(top=0, vault=str(self.vault), within=[], output="auto", no_snippet=False,
                               query=["drip"])
        with self.assertRaises(SearchError) as caught:
            search.search_command(args, self.cfg(), write=lambda text: None, loader=self.loader)
        self.assertEqual(caught.exception.code, "SEARCH_TOP_INVALID")

    def test_undecodable_file_makes_search_partial(self):
        write_vault(self.vault, {"Projects/Broken.md": b"\xff\xfe\x00\xd8 broken utf-16"})
        write_vault(self.vault, {"Projects/Bad utf8.md": b"caf\xe9 au lait and drip coffee"})
        code, text = self.run_cli("search", str(self.vault), "drip")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)

    def test_markdown_only(self):
        write_vault(self.vault, {"Projects/notes.txt": "drip line timer", "Projects/data.csv": "drip"})
        code, text = self.run_cli("search", str(self.vault), "drip", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertFalse([l for l in links(text) if "notes" in l or "data" in l])


class Scopes(VaultCase):
    def test_scope_limits_results(self):
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", "Journal", "--in", "Recipes")
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(links(text)), ["Journal/Spring trip", "Recipes/Kimchi stew"])

    def test_scope_errors(self):
        write_vault(self.vault, {"Projects/Deep/Inner.md": FRONT + "inner text\n",
                                 "Journal/Deep/Other.md": FRONT + "other text\n",
                                 "Projects/Only here/Unique.md": FRONT + "unique text\n"})
        cases = {"Nowhere": "SEARCH_SCOPE_EMPTY", "Deep": "SEARCH_SCOPE_AMBIGUOUS", "../x": "SEARCH_SCOPE_INVALID",
                 ".": "SEARCH_SCOPE_INVALID", "Archive": "SEARCH_SCOPE_EMPTY", ".obsidian": "SEARCH_SCOPE_EMPTY",
                 str(self.temp): "SEARCH_SCOPE_INVALID", "Health": "SEARCH_SCOPE_EMPTY"}
        for scope, expected in cases.items():
            with self.subTest(scope=scope):
                code, text = self.run_cli("search", str(self.vault), "text", "--in", scope)
                self.assertEqual(code, 2, text)
                self.assertEqual(json.loads(text)["code"], expected)
        self.assertEqual(self.loader.loads, 0)

    def test_unique_folder_name_resolves_and_says_so(self):
        write_vault(self.vault, {"Projects/Only here/Unique.md": FRONT + "unique text about otters\n"})
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "only here")
        self.assertEqual(code, 0, text)
        self.assertIn("scope_alias_resolved: Projects/Only here", text)
        self.assertEqual(links(text), ["Projects/Only here/Unique"])

    def test_folder_name_never_resolves_into_a_private_folder(self):
        write_vault(self.vault, {"Private/Inner/Secret.md": FRONT + "secret text\n"})
        code, text = self.run_cli("search", str(self.vault), "secret", "--in", "Inner")
        self.assertEqual(code, 2, text)
        self.assertEqual(json.loads(text)["code"], "SEARCH_SCOPE_EMPTY")
        code, text = self.run_cli("search", str(self.vault), "secret", "--in", "Private/Inner")
        self.assertEqual(code, 0, text)
        self.assertIn("snippet hidden", text)

    def test_vault_root_inside_private_folder_still_hides(self):
        code, text = self.run_cli("search", str(self.vault / "Private"), "slept badly")
        self.assertEqual(code, 0, text)
        self.assertIn("snippet hidden", text)
        self.assertNotIn("Slept badly", text)

    def test_anchored_patterns(self):
        write_vault(self.vault, {"Projects/Archive/Kept.md": FRONT + "kept text about lanterns\n"})
        config = self.config_file(exclude_dirs=["/Archive"])
        code, text = self.run_cli("search", str(self.vault), "lanterns", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        self.assertIn("Projects/Archive/Kept", links(text))
        self.assertNotIn("Archive/Old irrigation", links(text))
        config = self.config_file(exclude_dirs=["Archive"])
        code, text = self.run_cli("search", str(self.vault), "lanterns", "--top", "20", config=config)
        self.assertNotIn("Projects/Archive/Kept", links(text))

    def test_glob_pattern_excludes_subfolders_only(self):
        write_vault(self.vault, {"Backups/Top level.md": FRONT + "top level note about kites\n",
                                 "Backups/2026-01/Copy.md": FRONT + "copied note about kites\n"})
        config = self.config_file(exclude_dirs=["Backups/*"])
        code, text = self.run_cli("search", str(self.vault), "kites", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        self.assertIn("Backups/Top level", links(text))
        self.assertNotIn("Backups/2026-01/Copy", links(text))

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows feature")
    def test_junction_inside_vault_is_not_followed(self):
        import _winapi
        outside = self.temp / "outside"
        write_vault(outside, {"Leak.md": FRONT + "leaked text about drip lines\n"})
        _winapi.CreateJunction(str(outside), str(self.vault / "Projects" / "Linked"))
        code, text = self.run_cli("search", str(self.vault), "leaked drip lines", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertNotIn("Leak", text)
        self.assertIn("SEARCH_PARTIAL", text)

    def test_symlink_inside_vault_is_not_followed(self):
        outside = self.temp / "outside"
        write_vault(outside, {"Leak.md": FRONT + "leaked text about drip lines\n"})
        try:
            os.symlink(outside / "Leak.md", self.vault / "Projects" / "Leak.md")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a symlink here")
        code, text = self.run_cli("search", str(self.vault), "leaked drip lines", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertNotIn("leaked text", text)
        self.assertIn("SEARCH_PARTIAL", text)


class RootSafety(VaultCase):
    def blocked(self, *argv):
        code, text = self.run_cli(*argv)
        self.assertEqual(code, 2, text)
        self.assertEqual(self.loader.loads, 0)
        return text

    def test_unc_vault_root_is_refused(self):
        for root in (r"\\server\share\vault", r"\\?\UNC\server\share\vault", "//server/share/vault",
                     r"\\.\UNC\server\share\vault"):
            with self.subTest(root=root):
                text = self.blocked("search", root, "drip")
                self.assertEqual(json.loads(text), {"code": "SEARCH_ROOT_INVALID", "status": "SEARCH_BLOCKED"})
                text = self.blocked("recall", root, "--questions", str(self.temp / "none.json"))
                self.assertEqual(text.strip(), "RECALL_BLOCKED code=SEARCH_ROOT_INVALID")

    def test_unc_scope_is_refused(self):
        text = self.blocked("search", str(self.vault), "drip", "--in", r"\\server\share\vault\Projects")
        self.assertEqual(json.loads(text)["code"], "SEARCH_ROOT_INVALID")

    @unittest.skipUnless(os.name == "nt", "extended paths are a Windows feature")
    def test_local_extended_path_is_allowed(self):
        code, text = self.run_cli("search", "\\\\?\\" + str(self.vault), "drip line morning")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows feature")
    def test_junction_on_or_above_vault_is_refused(self):
        import _winapi
        above = self.temp / "above-link"
        _winapi.CreateJunction(str(self.temp), str(above))
        on = self.temp / "vault-link"
        _winapi.CreateJunction(str(self.vault), str(on))
        for root in (above / "vault", above / "vault" / "Projects", on):
            with self.subTest(root=root.name):
                text = self.blocked("search", str(root), "drip")
                self.assertEqual(json.loads(text)["code"], "SEARCH_ROOT_INVALID")
        code, text = self.run_cli("search", str(self.vault), "drip")
        self.assertEqual(code, 0, text)

    def test_symlink_above_vault_is_refused(self):
        above = self.temp / "above-symlink"
        try:
            os.symlink(self.temp, above, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a directory symlink here")
        text = self.blocked("search", str(above / "vault"), "drip")
        self.assertEqual(json.loads(text)["code"], "SEARCH_ROOT_INVALID")

    def test_refused_root_is_not_read_for_a_config_file(self):
        from unittest import mock
        from vault_search_lib import config as config_mod
        with mock.patch.object(config_mod, "load", side_effect=AssertionError("config read")):
            text = self.blocked("search", r"\\server\share\vault", "drip")
        self.assertEqual(json.loads(text)["code"], "SEARCH_ROOT_INVALID")


class CacheAndWarm(VaultCase):
    def warm(self, config=None):
        code, text = self.run_cli("warm", str(self.vault), config=config)
        self.assertEqual(code, 0, text)
        return json.loads(text)

    def cache_keys(self):
        path = embedding.cache_file(pages.checked_root(self.vault), self.cfg())
        with np.load(path) as data:
            return data["keys"].astype(str).tolist()

    def test_warm_then_search_embeds_only_queries(self):
        first = self.warm()
        self.assertEqual(first["status"], "WARM_COMPLETE")
        self.assertGreater(first["new_inputs"], 0)
        self.assertEqual(first["new_inputs"], first["embedding_inputs"])
        self.assertEqual(self.warm()["new_inputs"], 0)
        self.loader.encoder.batches.clear()
        code, text = self.run_cli("search", str(self.vault), "drip")
        self.assertEqual(code, 0, text)
        self.assertEqual(self.loader.passage_batches(), [])

    def test_whole_vault_run_prunes_scoped_run_keeps(self):
        self.warm()
        before = set(self.cache_keys())
        (self.vault / "Journal" / "Spring trip.md").unlink()
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", "Projects")
        self.assertEqual(code, 0, text)
        self.assertEqual(set(self.cache_keys()), before, "a scoped run never drops vectors")
        self.warm()
        self.assertLess(len(self.cache_keys()), len(before))

    def test_model_name_change_rebuilds(self):
        self.warm()
        config = self.config_file(model_name="other/model")
        self.assertGreater(self.warm(config)["new_inputs"], 0)

    def test_cache_lives_in_cache_dir_not_in_vault(self):
        self.warm()
        self.assertTrue(list(self.cache.rglob("search.npz")))
        self.assertFalse([p for p in self.vault.rglob("*") if p.suffix in (".npz", ".tmp")])

    def test_warm_with_pythonw_like_streams(self):
        import sys
        from support import cli
        saved = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = None
            code = cli.main(["warm", str(self.vault), "--config", str(self.config_file())], loader=self.loader)
        finally:
            sys.stdout, sys.stderr = saved
        self.assertEqual(code, 0)
        self.assertTrue(list(self.cache.rglob("search.npz")))


class ModelAndConfig(VaultCase):
    def test_model_not_configured(self):
        with self.assertRaises(SearchError) as caught:
            embedding.load_encoder(self.cfg())
        self.assertEqual(caught.exception.code, "SEARCH_MODEL_NOT_CONFIGURED")

    def test_model_folder_without_files(self):
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime and tokenizers are not installed")
        with self.assertRaises(SearchError) as caught:
            embedding.load_encoder(self.cfg(model_dir=str(self.temp)))
        self.assertEqual(caught.exception.code, "SEARCH_MODEL_UNAVAILABLE")

    def test_cli_reports_missing_model_as_fixed_code(self):
        from vault_search_lib import embedding as real
        code, text = self.run_cli("search", str(self.vault), "drip", loader=real.load_encoder)
        self.assertEqual(code, 2)
        self.assertIn(json.loads(text)["code"], ("SEARCH_MODEL_NOT_CONFIGURED", "SEARCH_DEPENDENCY_MISSING"))

    def test_bad_config(self):
        path = self.temp / "bad.json"
        for body, detail in (('{"exclude": []}', "unknown key exclude"), ('{"private_dirs": "x"}', "private_dirs"),
                             ('{"hide_snippets_when": [{"field": "a"}]}',
                              "hide_snippets_when needs exactly one of values, allow"),
                             ("[1]", "top level must be a JSON object"), ("{", "not valid JSON")):
            with self.subTest(body=body):
                path.write_text(body, encoding="utf-8")
                code, text = self.run_cli("search", str(self.vault), "drip", config=path)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(text), {"code": "CONFIG_INVALID", "detail": detail,
                                                    "status": "SEARCH_BLOCKED"})

    def test_config_in_vault_root_is_used(self):
        (self.vault / ".vault-search.json").write_text(json.dumps(
            {"cache_dir": str(self.cache), "exclude_dirs": ["Projects"]}), encoding="utf-8")
        import contextlib
        import io
        from support import _environ, cli
        out = io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
        with contextlib.redirect_stdout(out), _environ(env):
            code = cli.main(["search", str(self.vault), "drip", "--top", "20"], loader=self.loader)
        self.assertEqual(code, 0, out.getvalue())
        self.assertFalse([l for l in links(out.getvalue()) if l.startswith("Projects/")])

    def test_empty_config_path_is_refused(self):
        from vault_search_lib import config as config_mod
        (self.vault / ".vault-search.json").write_text(json.dumps(
            {"cache_dir": str(self.cache), "exclude_dirs": ["Projects"]}), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
        refused = {"code": "CONFIG_UNREADABLE", "detail": "empty configuration path", "status": "SEARCH_BLOCKED"}
        root = pages.checked_root(self.vault)
        for value in ("", "  "):
            with self.subTest(value=repr(value)):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), _environ(env):
                    code = cli.main(["search", str(self.vault), "drip", "--config", value], loader=self.loader)
                self.assertEqual((code, json.loads(out.getvalue())), (2, refused))
                with self.assertRaises(SearchError) as caught:
                    config_mod.load(root, path=value, environ={})
                self.assertEqual((caught.exception.code, caught.exception.detail),
                                 ("CONFIG_UNREADABLE", "empty configuration path"))
        self.assertEqual(self.loader.loads, 0)
        # An empty variable is no setting: the vault's own file is used, as without the variable.
        self.assertEqual(config_mod.load(root, environ={"VAULT_SEARCH_CONFIG": ""}).exclude_dirs, ["Projects"])
        self.assertEqual(config_mod.load(root, environ={}).exclude_dirs, ["Projects"])

    def test_missing_vault(self):
        code, text = self.run_cli("search", str(self.temp / "nope"), "drip")
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text)["code"], "SEARCH_VAULT_NOT_FOUND")

    def test_unexpected_error_is_a_fixed_code(self):
        def broken(cfg):
            raise RuntimeError("note text that must not be printed")
        code, text = self.run_cli("search", str(self.vault), "drip", loader=broken)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text)["code"], "SEARCH_FAILED")
        self.assertNotIn("must not be printed", text)


def candidates(text):
    return text.split("Candidates are", 1)[1]


class FrontMatterPrivacy(VaultCase):
    def test_hide_rule_reads_comments_quoted_keys_and_blocks(self):
        write_vault(self.vault, {
            "Notes/Comment.md": "---\nsensitivity: high # reviewed\n---\n## Secret heading A\nzebra crossing alpha\n",
            "Notes/Quoted.md": '---\n"sensitivity": high\n---\n## Secret heading B\nzebra crossing beta\n',
            "Notes/Block.md": ("---\nnote: |\n  first\n  ---\n  second\nsensitivity: high\n---\n"
                               "## Secret heading C\nzebra crossing gamma\n")})
        code, text = self.run_cli("search", str(self.vault), "zebra crossing", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertEqual(sum(1 for link in links(text) if link.startswith("Notes/") and link != "Notes/Shown by field"
                             and link != "Notes/Hidden by field"), 3)
        self.assertNotIn("Secret heading", text)
        self.assertNotIn("zebra crossing", candidates(text))

    def test_unreadable_front_matter_hides_only_under_rules(self):
        write_vault(self.vault, {
            "Notes/Broken.md": "---\nsensitivity: [high\n---\n## Broken heading\nquokka broken text\n",
            "Notes/Nested.md": "---\nsensitivity:\n  level: low\n---\n## Nested heading\nquokka nested text\n"})
        code, text = self.run_cli("search", str(self.vault), "quokka text", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertNotIn("Broken heading", text)
        self.assertNotIn("Nested heading", text)
        self.assertNotIn("quokka", candidates(text))
        code, text = self.run_cli("search", str(self.vault), "quokka text", "--top", "20",
                                  config=self.config_file(hide_snippets_when=[]))
        self.assertEqual(code, 0, text)
        self.assertIn("Broken heading", text)
        self.assertIn("quokka nested text", text)

    UNSURE = {
        "Unsure/Cp949.md": "---\ntype: secretlabel\nsensitivity: low\n---\n## Secret heading\n잠긴 글 quokka\n"
                           .encode("cp949"),
        "Unsure/Utf16 bare.md": "---\ntype: secretlabel\n---\n## Secret heading\nquokka bare\n".encode("utf-16-le"),
        "Unsure/Utf16 bom.md": "---\ntype: secretlabel\n---\n## Secret heading\nquokka bom\n".encode("utf-16"),
        "Unsure/Dots.md": "---\ntype: secretlabel\nsensitivity: low\n...\n## Secret heading\nquokka dots\n",
        "Unsure/Open.md": "---\ntype: secretlabel\nsensitivity: low\n## Secret heading\nquokka open\n",
        "Unsure/Twice.md": "---\ntype: secretlabel\nsensitivity: high\nsensitivity: low\n---\n"
                           "## Secret heading\nquokka twice\n",
        "Unsure/Nested.md": "---\ntype: secretlabel\nsensitivity:\n  level: low\n---\n## Secret heading\nquokka nested\n",
    }

    def test_notes_the_rules_cannot_clear_show_only_the_link(self):
        write_vault(self.vault, self.UNSURE)
        config = self.config_file(label_fields=["type"])
        for option in ([], ["--output", "paths"], ["--no-snippet"]):
            with self.subTest(option=option):
                code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20", *option, config=config)
                self.assertEqual(code, 0, text)
                shown = text.replace("\x00", "")
                self.assertNotIn("Secret heading", shown)
                self.assertNotIn("secretlabel", shown)
                self.assertNotIn("quokka", candidates(shown))
                self.assertNotIn("잠긴", shown)
                self.assertTrue({p[:-3] for p in self.UNSURE} <= set(links(text)), links(text))
                self.assertIn("SEARCH_PARTIAL", text)  # the CP949 note is also a read problem
                if not option:
                    self.assertIn("balcony plants", text)  # other notes are shown as before

    def test_value_that_does_not_fit_its_tag_hides_the_note_and_search_goes_on(self):
        write_vault(self.vault, {"Unsure/Tagged.md": "---\ntype: secretlabel\nsensitivity: !!bool maybe\n---\n"
                                                     "## Secret heading\nquokka tagged\n"})
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20",
                                  config=self.config_file(label_fields=["type"]))
        self.assertEqual(code, 0, text)
        self.assertIn("Unsure/Tagged", links(text))
        self.assertNotIn("Secret heading", text)
        self.assertNotIn("secretlabel", text)
        self.assertIn("SEARCH_COMPLETE", text)

    def test_valid_tagged_values_make_the_note_unsure(self):
        write_vault(self.vault, {"Levels/Ten.md": "---\nlevel: !!int 10\n---\n## Ten heading\nmarmot ten\n",
                                 "Levels/Three.md": "---\nlevel: !!int 3\n---\n## Three heading\nmarmot three\n",
                                 "Levels/Flag.md": "---\nflag: !!bool true\n---\n## Flag heading\nmarmot flag\n",
                                 "Levels/Plain.md": "---\nlevel: 3\n---\n## Plain heading\nmarmot plain\n"})
        config = self.config_file(hide_snippets_when=[{"field": "level", "values": ["3"]},
                                                      {"field": "flag", "values": ["true"]}],
                                  label_fields=["level"])
        code, text = self.run_cli("search", str(self.vault), "marmot", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        for heading in ("Ten heading", "Three heading", "Flag heading", "Plain heading"):
            self.assertNotIn(heading, text)
        self.assertNotIn("level: 10", text)
        self.assertEqual(text.count("level: 3"), 1, "only the plain note a rule hides keeps its labels")
        self.assertTrue({"Levels/Ten", "Levels/Three", "Levels/Flag", "Levels/Plain"} <= set(links(text)))

    def test_without_rules_those_notes_read_as_before(self):
        write_vault(self.vault, {k: v for k, v in self.UNSURE.items() if k.endswith(("Dots.md", "Open.md"))})
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20",
                                  config=self.config_file(hide_snippets_when=[], label_fields=["sensitivity"]))
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("Secret heading"), 2)
        code, text = self.run_cli("search", str(self.vault), "greenhouse watering", "--top", "20",
                                  config=self.config_file(label_fields=["sensitivity"]))
        hidden = next(line for line in text.splitlines() if "[[Notes/Hidden by field]]" in line)
        self.assertIn("sensitivity: high", hidden)

    def test_unsure_note_front_matter_is_not_used_for_ranking(self):
        files = {"Pins/Alpha lantern.md": FRONT + "lantern lantern lantern light\n",
                 "Pins/Beta lantern.md": ("---\nfeatured: true\nkeywords: [zephyrine]\nsensitivity: low\n"
                                          "sensitivity: low\n---\nrivers and boats\n"),
                 # Enough pages that a word in two titles is not ignored as common.
                 **{f"Pins/Filler {n}.md": FRONT + f"filler page number {n} about mountains\n" for n in range(6)}}
        write_vault(self.vault, files)
        order = {}
        for pin in ("featured", None):
            code, text = self.run_cli("search", str(self.vault), "lantern", "--top", "4",
                                      config=self.config_file(pin_field=pin))
            self.assertEqual(code, 0, text)
            order[pin] = links(text)
        self.assertEqual(order["featured"], order[None])
        self.assertEqual(order[None][0], "Pins/Alpha lantern")
        code, text = self.run_cli("search", str(self.vault), "zephyrine", "--top", "4")
        self.assertNotIn("exact term", text)
        write_vault(self.vault, {"Pins/Beta lantern.md": "---\nfeatured: true\n---\nrivers and boats\n"})
        code, text = self.run_cli("search", str(self.vault), "lantern", "--top", "4",
                                  config=self.config_file(pin_field="featured"))
        self.assertEqual(links(text)[0], "Pins/Beta lantern", "a page the rules clear is still pinned")


def _y1_note(front, word):
    return f"---\ntype: secretlabel\n{front}---\n## Secret heading\nquokka {word}\n"


class PlainYamlPrivacy(VaultCase):
    """Y1 and Y2 end to end; run as installed and with libyaml hidden."""
    REFUSED = {
        "Yaml/Null tag.md": _y1_note("sensitivity: !!null sample\n", "nulltag"),
        "Yaml/Binary tag.md": _y1_note("sensitivity: !!binary '????'\n", "binary"),
        "Yaml/Seq tag.md": "---\n!!seq {type: secretlabel, sensitivity: low}\n---\n## Secret heading\nquokka seq\n",
        "Yaml/Merge.md": _y1_note("<<: {sensitivity: high}\n", "merge"),
        "Yaml/Anchor merge.md": _y1_note("base: &b {sensitivity: high}\n<<: *b\n", "anchormerge"),
        "Yaml/Str tag.md": _y1_note("sensitivity: !!str low\n", "strtag"),
        "Yaml/Other field tag.md": _y1_note("level: !!int 10\nsensitivity: low\n", "otherfield"),
        "Yaml/Collection tag.md": _y1_note("other: !!str {a: 1}\nsensitivity: low\n", "collection"),
        "Yaml/Unused anchor.md": _y1_note("sensitivity: &s low\n", "anchor"),
        "Yaml/Alias.md": _y1_note("first: &f low\nsensitivity: *f\n", "alias"),
        "Yaml/Control.md": _y1_note("sensitivity: low\x01\n", "control"),
        "Yaml/Lone cr.md": _y1_note("sensitivity: low\r---\rsensitivity: high\n", "cr"),
        **{f"Yaml/Sep {ord(sep):x} {where}.md": _y1_note(front, f"sep{ord(sep):x}{where}")
           for sep in ("\u2028", "\u2029", "\x85")
           for where, front in (("before", f"sensitivity: low{sep}---\nsensitivity: high\n"),
                                ("after", f"sensitivity: low\n---{sep}sensitivity: high\n"))},
    }

    def test_notes_with_yaml_features_show_only_the_link(self):
        write_vault(self.vault, self.REFUSED)
        config = self.config_file(label_fields=["type"])
        for option in ([], ["--output", "paths"], ["--no-snippet"]):
            with self.subTest(option=option):
                code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "60", *option, config=config)
                self.assertEqual(code, 0, text)
                self.assertIn("SEARCH_COMPLETE", text)
                self.assertNotIn("Secret heading", text)
                self.assertNotIn("secretlabel", text)
                self.assertNotIn("quokka", candidates(text))
                self.assertTrue({p[:-3] for p in self.REFUSED} <= set(links(text)),
                                sorted({p[:-3] for p in self.REFUSED} - set(links(text))))
                if not option:
                    self.assertIn("balcony plants", text)  # other notes are shown as before

    UNCLEAR_OPENING = {f"Yaml/Open {name}.md": first + "\ntype: secretlabel\nsensitivity: low\n---\n"
                                               f"## Secret heading\nquokka open{name.replace(' ', '')}\n"
                       for name, first in (("2028", "---\u2028"), ("2029", "---\u2029"), ("85", "---\x85"),
                                           ("nbsp", "---\u00a0"), ("space before", " ---"))}

    def test_unclear_first_line_hides_under_values_and_allow_rules(self):
        write_vault(self.vault, {**self.UNCLEAR_OPENING,
                                 "Yaml/Clean.md": "---  \ntype: plainlabel\nsensitivity: low\n---\n"
                                                  "## Clean heading\nquokka clean\n"})
        for rule in ({"field": "sensitivity", "values": ["high"]}, {"field": "sensitivity", "allow": ["low"]}):
            with self.subTest(rule=sorted(rule)):
                code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "30",
                                          config=self.config_file(hide_snippets_when=[rule], label_fields=["type"]))
                self.assertEqual(code, 0, text)
                self.assertNotIn("Secret heading", text)
                self.assertNotIn("secretlabel", text)
                self.assertTrue({p[:-3] for p in self.UNCLEAR_OPENING} <= set(links(text)))
                self.assertIn("Clean heading", text)
                self.assertIn("type: plainlabel", text)
    def test_without_rules_an_unclear_first_line_gives_no_front_matter(self):
        """Owner decision V1: like an unreadable front matter, the body is searched and no field is used."""
        unclear = {path: text.replace("sensitivity: low\n", "keywords: [zephyrine]\nfeatured: true\n")
                   for path, text in self.UNCLEAR_OPENING.items()}
        write_vault(self.vault, {**unclear,
                                 "Yaml/Clean.md": "---  \ntype: plainlabel\n---\n## Clean heading\nquokka clean\n",
                                 **{f"Yaml/Filler {n}.md": FRONT + f"filler page {n} about mountains\n"
                                    for n in range(30)}})
        config = self.config_file(hide_snippets_when=[], label_fields=["type"], pin_field="featured")
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "60", config=config)
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("Secret heading"), len(unclear), "the body is searched")
        self.assertNotIn("secretlabel", text)
        self.assertIn("type: plainlabel", text)
        code, text = self.run_cli("search", str(self.vault), "zephyrine", "--top", "60", config=config)
        self.assertEqual(code, 0, text)
        self.assertNotIn("exact term", text)
        root = pages.checked_root(self.vault)
        for rel in unclear:
            with self.subTest(note=rel):
                page = pages.read_page(root, root / rel)
                self.assertTrue(page.front_open_unclear)
                self.assertEqual((page.meta, page.aliases), ({}, []), "so no label, exact term or pin_field")

    def test_null_tag_does_not_pass_an_allow_rule(self):
        write_vault(self.vault, {"Yaml/Null low.md": _y1_note("sensitivity: !!null low\n", "nulllow"),
                                 "Yaml/Plain low.md": "---\nsensitivity: low\n---\n## Plain heading\nquokka plain\n"})
        config = self.config_file(hide_snippets_when=[{"field": "sensitivity", "allow": ["low"]}])
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        self.assertNotIn("Secret heading", text)
        self.assertIn("Plain heading", text)

    def test_their_front_matter_is_not_used_for_ranking(self):
        files = {"Pins/Alpha lantern.md": FRONT + "lantern lantern lantern light\n",
                 "Pins/Beta lantern.md": ("---\nfeatured: true\nkeywords: [zephyrine]\n<<: {sensitivity: low}\n"
                                          "---\nrivers and boats\n"),
                 "Pins/Gamma lantern.md": "---\nfeatured: true\n---\nlakes and canoes\n",
                 **{f"Pins/Filler {n}.md": FRONT + f"filler page number {n} about mountains\n" for n in range(6)}}
        write_vault(self.vault, files)
        code, text = self.run_cli("search", str(self.vault), "lantern", "--top", "4",
                                  config=self.config_file(pin_field="featured"))
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text)[0], "Pins/Gamma lantern", "a plain note is still pinned")
        self.assertNotEqual(links(text)[1], "Pins/Beta lantern")
        code, text = self.run_cli("search", str(self.vault), "zephyrine", "--top", "4")
        self.assertNotIn("exact term", text)

    def test_without_rules_the_body_is_searched_and_the_front_matter_is_not_used(self):
        write_vault(self.vault, {"Yaml/Merge.md": self.REFUSED["Yaml/Merge.md"],
                                 "Yaml/Plain.md": "---\ntype: plainlabel\n---\n## Plain heading\nquokka plain\n"})
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20",
                                  config=self.config_file(hide_snippets_when=[], label_fields=["type"]))
        self.assertEqual(code, 0, text)
        self.assertIn("Secret heading", text)
        self.assertIn("quokka merge", text)
        self.assertNotIn("secretlabel", text)
        self.assertIn("type: plainlabel", text)

    SEPARATED = {f"Sep/Clipped {ord(char):x}.md": f"---\ntype: secretlabel\ntitle: clipped{char}sensitivity: low\n"
                                                  f"---\n## Secret heading\nquokka clipped{ord(char):x}\n"
                 for char in ("\u2028", "\u2029", "\x85", "\r", "\x0c")}

    def test_line_separator_in_a_value_does_not_pass_an_allow_rule(self):
        write_vault(self.vault, {**self.SEPARATED,
                                 "Sep/Plain.md": "---\ntype: plainlabel\nsensitivity: low\n---\n"
                                                 "## Plain heading\nquokka plain\n"})
        config = self.config_file(hide_snippets_when=[{"field": "sensitivity", "allow": ["low"]}],
                                  label_fields=["type"])
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "30", config=config)
        self.assertEqual(code, 0, text)
        self.assertNotIn("Secret heading", text)
        self.assertNotIn("secretlabel", text)
        self.assertTrue({p[:-3] for p in self.SEPARATED} <= set(links(text)), links(text))
        self.assertIn("Plain heading", text)
        self.assertIn("type: plainlabel", text)
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "30",
                                  config=self.config_file(hide_snippets_when=[], label_fields=["type"]))
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("Secret heading"), len(self.SEPARATED), "without rules the body is searched")
        self.assertNotIn("secretlabel", text, "and that front matter is not used")

    def test_front_matter_of_white_space_only_is_unsure(self):
        write_vault(self.vault, {"Blank/Tab.md": "---\n\t\n---\n## Secret heading\nquokka tab\n",
                                 "Blank/Spaces.md": "---\n   \n---\n## Secret heading\nquokka spaces\n",
                                 "Blank/Empty.md": "---\n---\n## Empty heading\nquokka empty\n"})
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "30")
        self.assertEqual(code, 0, text)
        self.assertNotIn("Secret heading", text)
        self.assertTrue({"Blank/Tab", "Blank/Spaces"} <= set(links(text)), links(text))
        self.assertIn("Empty heading", text, "an empty front matter has no fields and is cleared")

    def test_plain_front_matter_still_hides_and_shows(self):
        write_vault(self.vault, {
            "Ok/Crlf.md": b"---\r\nsensitivity: high\r\n---\r\n## Crlf heading\r\nquokka crlf\r\n",
            "Ok/Low.md": "---\nsensitivity: low\n---\n## Low heading\nquokka low\n",
            "Ok/Scalars.md": "---\nsensitivity: low\nprivate: true\nupdated: 2026-01-02\nversion: 1.10\n---\n"
                             "## Scalar heading\nquokka scalars\n"})
        code, text = self.run_cli("search", str(self.vault), "quokka", "--top", "20",
                                  config=self.config_file(label_fields=["updated"]))
        self.assertEqual(code, 0, text)
        self.assertNotIn("Crlf heading", text)
        self.assertIn("Ok/Crlf", links(text))
        self.assertIn("Low heading", text)
        self.assertIn("Scalar heading", text)
        self.assertIn("updated: 2026-01-02", text)
        self.assertIn("SEARCH_COMPLETE", text)


class PlainYamlPrivacyWithoutLibyaml(WithoutLibyaml, PlainYamlPrivacy):
    pass


class CodeBlockRanking(VaultCase):
    files = {"Space/Rocket.md": FRONT + "## Launch\nengines and fuel tanks\n",
             **{f"Space/Filler {n}.md": FRONT + f"## Part\nfiller page {n} about oceans\n" for n in range(5)}}
    settings = {}

    def first(self):
        code, text = self.run_cli("search", str(self.vault), "rocket", "--top", "4")
        self.assertEqual(code, 0, text)
        return next(line for line in text.splitlines() if line.startswith("1. "))

    def test_words_in_code_blocks_do_not_change_exact_ranking(self):
        base = self.first()
        self.assertIn("[[Space/Rocket]]", base)
        self.assertIn("exact term", base)
        for name, code in {"fenced": "```\nrocket\n```\n", "indented": "\n    rocket\n\n",
                           "in a list": "- item\n\n  ```\n  rocket\n  ```\n",
                           "in a quote": "> ```\n> rocket\n> ```\n"}.items():
            with self.subTest(code=name):
                write_vault(self.vault, {"Space/Filler 0.md": FRONT + "## Part\nfiller page about oceans\n" + code})
                self.assertEqual(self.first(), base)
        for name, body in {"plain": "rocket\n", "inline code": "the `rocket` word\n"}.items():
            with self.subTest(body=name):
                write_vault(self.vault, {"Space/Filler 0.md": FRONT + "## Part\nfiller page about oceans\n" + body})
                self.assertNotIn("exact term", self.first())


class FolderNames(VaultCase):
    def test_skip_dir_names_skip_folders_only(self):
        write_vault(self.vault, {"Notes/Index.md": FRONT + "index note about herons\n",
                                 "Index.md/Inner.md": FRONT + "inner note about herons\n",
                                 "Notes/.draft.md": FRONT + "dot note about herons\n",
                                 "node_modules/pkg.md": FRONT + "package note about herons\n"})
        config = self.config_file(skip_dir_names=["Index.md", "node_modules"])
        code, text = self.run_cli("search", str(self.vault), "herons", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        found = links(text)
        self.assertIn("Notes/Index", found)
        self.assertNotIn("Index.md/Inner", found)
        self.assertNotIn("Notes/.draft", found)
        self.assertNotIn("node_modules/pkg", found)
        self.assertIn("SEARCH_COMPLETE", text)

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows feature")
    def test_link_with_a_skipped_folder_name_is_skipped(self):
        import _winapi
        outside = self.temp / "outside"
        write_vault(outside, {"Leak.md": FRONT + "leaked text about herons\n"})
        _winapi.CreateJunction(str(outside), str(self.vault / "Projects" / "node_modules"))
        code, text = self.run_cli("search", str(self.vault), "leaked herons", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_COMPLETE", text)
        self.assertNotIn("Leak", text)

    def broken(self, files):
        try:
            write_vault(self.vault, files)
        except (OSError, UnicodeError):
            self.skipTest("cannot create a name with a lone surrogate here")

    def test_unencodable_scope_folder_is_one_read_problem(self):
        self.broken({"bad\udc80 dir/Note.md": FRONT + "## Bad\nsurrogate folder note about drip\n"})
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", "bad\udc80 dir", "--in", "Journal")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)
        self.assertIn("read_problems=1", text)
        self.assertEqual(links(text), ["Journal/Spring trip"])
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", "bad\udc80 dir")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false, candidates=0", text)
        self.assertEqual(links(text), [], "a broken scope never widens to the whole vault")

    def test_folder_name_resolves_next_to_a_broken_note(self):
        self.broken({"Projects/bad\udc80.md": FRONT + "broken note\n",
                     "Projects/Only here/Unique.md": FRONT + "unique text about otters\n"})
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "only here")
        self.assertEqual(code, 0, text)
        self.assertIn("scope_alias_resolved: Projects/Only here", text)
        self.assertEqual(links(text), ["Projects/Only here/Unique"])
        self.broken({"Other/bad\udc80 dir/x.md": FRONT + "x\n"})
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "only here")
        self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_SCOPE_UNRESOLVED"),
                         "a folder that cannot be searched may hide another folder of that name")


    def test_read_problem_of_a_named_scope_does_not_depend_on_the_order(self):
        self.broken({"Nested/OnlyHere/Good.md": FRONT + "good note about otters\n",
                     "Private/bad\udc80 dir/x.md": FRONT + "x\n"})
        cfg = self.cfg()
        root = pages.checked_root(self.vault)
        for within in (["OnlyHere", "Private/bad\udc80 dir"], ["Private/bad\udc80 dir", "OnlyHere"]):
            with self.subTest(first=within[0][:8]):
                found, problems = pages.select_paths(root, within, cfg)
                self.assertEqual([p.relative_to(root).as_posix() for p in found], ["Nested/OnlyHere/Good.md"])
                self.assertEqual(problems, ["NAME_UNENCODABLE"])
                code, text = self.run_cli("search", str(self.vault), "otters", "--in", within[0], "--in", within[1])
                self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)
                self.assertIn("read_problems=1", text)
        found, problems = pages.select_paths(root, ["OnlyHere", "Projects"], cfg)
        self.assertEqual(problems, [])
        self.assertTrue(found)

    def test_broken_folder_named_inside_another_scope_is_one_read_problem(self):
        self.broken({"Other/Good.md": FRONT + "good note about otters\n", "Other/bad\udc80 dir/x.md": FRONT + "x\n"})
        cfg = self.cfg()
        root = pages.checked_root(self.vault)
        for within in (["Other"], ["Other", "Other/bad\udc80 dir"], ["Other/bad\udc80 dir", "Other"]):
            with self.subTest(within=len(within), first=within[0][:6]):
                found, problems = pages.select_paths(root, within, cfg)
                self.assertEqual([p.relative_to(root).as_posix() for p in found], ["Other/Good.md"])
                self.assertEqual(problems, ["NAME_UNENCODABLE"])

    def test_scopes_that_stop_at_the_same_broken_folder_are_one_read_problem(self):
        self.broken({"bad\udc80 dir/sub/x.md": FRONT + "x\n", "Other/Good.md": FRONT + "good note about otters\n"})
        cfg = self.cfg()
        root = pages.checked_root(self.vault)
        for within in (["bad\udc80 dir", "bad\udc80 dir"], ["bad\udc80 dir", "bad\udc80 dir/sub"],
                       ["bad\udc80 dir/one", "bad\udc80 dir/two"]):
            with self.subTest(within=[value[len("bad\udc80 dir"):] for value in within]):
                found, problems = pages.select_paths(root, within, cfg)
                self.assertEqual((found, problems), ([], ["NAME_UNENCODABLE"]))
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "bad\udc80 dir", "--in",
                                  "bad\udc80 dir/sub", "--in", "Other")
        self.assertEqual(code, 0, text)
        self.assertIn("read_problems=1", text)
        self.assertEqual(links(text), ["Other/Good"])

    def test_named_scope_next_to_a_junction(self):
        if os.name != "nt":
            self.skipTest("junctions are a Windows feature")
        import _winapi
        outside = self.temp / "outside"
        write_vault(outside, {"Leak.md": FRONT + "leaked text\n"})
        write_vault(self.vault, {"Only here/Top.md": FRONT + "top note about otters\n",
                                 "Sub/Deep/Inner.md": FRONT + "deep note about otters\n"})
        _winapi.CreateJunction(str(outside), str(self.vault / "Projects" / "Door"))
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "only here")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text), ["Only here/Top"], "a folder at the vault root is taken first")
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "deep")
        self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_SCOPE_UNRESOLVED"))
        code, text = self.run_cli("search", str(self.vault), "otters", "--in", "deep",
                                  config=self.config_file(exclude_dirs=["Archive", "Door"]))
        self.assertEqual(code, 0, text)
        self.assertIn("scope_alias_resolved: Sub/Deep", text)


class FileLinks(VaultCase):
    """A link to a file is a read problem even when its name matches a folder rule."""

    def link_file(self, name):
        target = self.temp / "outside" / "Leak.md"
        write_vault(target.parent, {"Leak.md": FRONT + "leaked text about herons\n"})
        try:
            os.symlink(target, self.vault / name)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a symlink here")

    def test_file_link_named_like_a_skipped_folder_is_a_read_problem(self):
        write_vault(self.vault, {"Notes/Index.md": FRONT + "index note about herons\n"})
        self.link_file("Projects/Index.md")
        config = self.config_file(skip_dir_names=["Index.md"])
        code, text = self.run_cli("search", str(self.vault), "herons", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)
        self.assertIn("read_problems=1", text)
        self.assertIn("Notes/Index", links(text))
        self.assertNotIn("leaked", text)

    def test_file_link_named_like_an_excluded_folder_keeps_the_cache(self):
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        before = embedding._load(embedding.cache_file(pages.checked_root(self.vault), self.cfg()),
                                 self.cfg().model_name)[0].keys()
        (self.vault / "Journal" / "Spring trip.md").unlink()
        self.link_file("Projects/Archive")
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual((code, json.loads(text)["status"]), (0, "WARM_PARTIAL"))
        after = embedding._load(embedding.cache_file(pages.checked_root(self.vault), self.cfg()),
                                self.cfg().model_name)[0].keys()
        self.assertEqual(set(after), set(before), "a partial scan keeps the vectors of notes it did not read")


class RepeatedQuestions(VaultCase):
    def test_repeated_question_is_embedded_once(self):
        code, text = self.run_cli("search", str(self.vault), "drip line", "drip line", "brakes")
        self.assertEqual(code, 0, text)
        asked = [t for batch in self.loader.encoder.batches for t in batch if t.startswith("query: ")]
        self.assertEqual(asked, ["query: drip line", "query: brakes"])
        blocks = text.split("query: ")[1:]
        self.assertEqual(len(blocks), 3)
        self.assertEqual(links(blocks[0]), links(blocks[1]))
        self.assertNotEqual(links(blocks[0]), links(blocks[2]))

    def test_question_equal_to_a_window_input_is_embedded_once(self):
        write_vault(self.vault, {"Solo.md": "sample words about owls\n"})
        config = self.config_file(query_prefix="", passage_prefix="")
        code, text = self.run_cli("search", str(self.vault), "Solo\nsample words about owls", "drip line",
                                  config=config)
        self.assertEqual(code, 0, text)
        embedded = [t for batch in self.loader.encoder.batches for t in batch]
        self.assertEqual(embedded.count("Solo\nsample words about owls"), 1)
        self.assertEqual(embedded.count("drip line"), 1)
        self.assertEqual(links(text.split("query: ")[1])[0], "Solo")
        self.assertIn("similarity 1.00", text.split("query: ")[1])


class FolderPatterns(VaultCase):
    def test_brackets_are_literal(self):
        write_vault(self.vault, {"[Old] Notes/Heron.md": FRONT + "a note about herons\n",
                                 "O Notes/Heron two.md": FRONT + "another note about herons\n",
                                 "Diary [2026]/Day.md": FRONT + "## Day heading\nherons at the lake\n",
                                 "Drafts/Wip.md": FRONT + "herons draft\n", "Draft/Kept.md": FRONT + "herons kept\n"})
        config = self.config_file(exclude_dirs=["Archive", "[Old] Notes", "Draft?"], private_dirs=["Diary [2026]"])
        code, text = self.run_cli("search", str(self.vault), "herons", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        found = links(text)
        self.assertNotIn("[Old] Notes/Heron", found)
        self.assertIn("O Notes/Heron two", found)
        self.assertIn("Diary [2026]/Day", found)
        self.assertNotIn("Day heading", text)
        self.assertNotIn("Drafts/Wip", found)
        self.assertIn("Draft/Kept", found)

    def test_private_pattern_never_matches_the_file_name(self):
        write_vault(self.vault, {"Backups/Direct.md": FRONT + "## Direct heading\ndirect note about kites\n",
                                 "Backups/2026/Inner.md": FRONT + "## Inner heading\ninner note about kites\n"})
        config = self.config_file(private_dirs=["Backups/*"])
        code, text = self.run_cli("search", str(self.vault), "note about kites", "--top", "20", config=config)
        self.assertEqual(code, 0, text)
        self.assertIn("direct note about kites", text)
        self.assertNotIn("inner note about kites", text)
        self.assertNotIn("Inner heading", text)


class FileAccess(VaultCase):
    def run_default_config(self, *argv):
        out = io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
        env["VAULT_SEARCH_CACHE_DIR"] = str(self.cache)
        with contextlib.redirect_stdout(out), _environ(env):
            code = cli.main(list(argv), loader=self.loader)
        return code, out.getvalue()

    def test_default_config_must_be_a_regular_file(self):
        outside = self.temp / "outside.json"
        outside.write_text(json.dumps({"exclude_dirs": ["Projects"], "secret_key_name": 1}), encoding="utf-8")
        refused = {"code": "CONFIG_UNREADABLE", "detail": "link or special file inside the vault",
                   "status": "SEARCH_BLOCKED"}
        try:
            os.symlink(outside, self.vault / ".vault-search.json")
        except (OSError, NotImplementedError):
            pass
        else:
            code, text = self.run_default_config("search", str(self.vault), "drip")
            self.assertEqual((code, json.loads(text)), (2, refused))
            (self.vault / ".vault-search.json").unlink()
        (self.vault / ".vault-search.json").mkdir()
        code, text = self.run_default_config("search", str(self.vault), "drip")
        self.assertEqual((code, json.loads(text)), (2, refused))
        self.assertEqual(self.loader.loads, 0)

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows feature")
    def test_files_in_the_vault_are_not_opened_through_a_junction(self):
        import _winapi
        outside = self.temp / "outside"
        outside.mkdir()
        (outside / "c.json").write_text(json.dumps({"cache_dir": str(self.cache)}), encoding="utf-8")
        (outside / "q.json").write_text(json.dumps({"questions": [
            {"id": "q1", "query": "drip line", "gold": ["Projects/Garden irrigation.md"]}]}), encoding="utf-8")
        (outside / "out.json").write_text("ORIGINAL", encoding="utf-8")
        (self.temp / "q.json").write_text((outside / "q.json").read_text(encoding="utf-8"), encoding="utf-8")
        _winapi.CreateJunction(str(outside), str(self.vault / "Linked"))
        linked = self.vault / "Linked"
        config = self.config_file(exclude_dirs=["Archive", "Linked"])
        code, text = self.run_cli("search", str(self.vault), "drip", config=linked / "c.json")
        self.assertEqual((code, json.loads(text)["code"]), (2, "CONFIG_UNREADABLE"))
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(linked / "q.json"), config=config)
        self.assertEqual(code, 2, text)
        self.assertIn("RECALL_BLOCKED code=RECALL_QUESTIONS_INVALID detail=link or special file", text)
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(self.temp / "q.json"),
                                  "--json-out", str(linked / "out.json"), config=config)
        self.assertEqual(code, 2, text)
        self.assertIn("RECALL_BLOCKED code=RECALL_OUTPUT_FAILED", text)
        self.assertNotIn("RECALL_COMPLETE", text)
        self.assertEqual((outside / "out.json").read_text(encoding="utf-8"), "ORIGINAL")
        self.assertEqual(self.loader.loads, 0)
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(self.temp / "q.json"),
                                  "--json-out", str(self.vault / "Projects" / "result.json"), config=config)
        self.assertEqual(code, 0, text)
        self.assertTrue((self.vault / "Projects" / "result.json").is_file())

    def outside_files(self):
        outside = self.temp / "outside"
        outside.mkdir(exist_ok=True)
        (outside / "c.json").write_text(json.dumps({"cache_dir": str(self.cache), "exclude_dirs": ["Projects"]}),
                                        encoding="utf-8")
        (outside / "q.json").write_text(json.dumps({"questions": [
            {"id": "q1", "query": "drip line", "gold": ["Projects/Garden irrigation.md"]}]}), encoding="utf-8")
        (outside / "out.json").write_text("ORIGINAL", encoding="utf-8")
        return outside

    def hard_link(self, source, name):
        try:
            os.link(source, self.vault / name)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a hard link here")
        return self.vault / name

    def test_files_with_two_names_in_the_vault_are_refused(self):
        outside = self.outside_files()
        twin = "file with more than one name inside the vault"
        config = self.hard_link(outside / "c.json", "c.json")
        code, text = self.run_cli("search", str(self.vault), "drip", config=config)
        self.assertEqual((code, json.loads(text)), (2, {"code": "CONFIG_UNREADABLE", "detail": twin,
                                                        "status": "SEARCH_BLOCKED"}))
        self.hard_link(outside / "c.json", ".vault-search.json")
        code, text = self.run_default_config("search", str(self.vault), "drip")
        self.assertEqual((code, json.loads(text)["detail"]), (2, twin))
        (self.vault / ".vault-search.json").unlink()
        questions = self.hard_link(outside / "q.json", "q.json")
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(questions))
        self.assertEqual((code, text.strip()), (2, f"RECALL_BLOCKED code=RECALL_QUESTIONS_INVALID detail={twin}"))
        result = self.hard_link(outside / "out.json", "out.json")
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(result))
        self.assertEqual((code, text.strip()), (2, f"RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail={twin}"))
        self.assertEqual((outside / "out.json").read_text(encoding="utf-8"), "ORIGINAL")
        self.assertEqual(self.loader.loads, 0)

    @unittest.skipUnless(os.name == "nt", "8.3 short names and junctions are Windows features")
    def test_short_and_case_spellings_get_the_vault_rules(self):
        import _winapi
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(str(self.vault), buffer, len(buffer)):
            self.skipTest("cannot read the short name")
        short = Path(buffer.value)
        if os.fspath(short).casefold() == os.fspath(self.vault).casefold():
            self.skipTest("8.3 short names are off on this volume")
        outside = self.outside_files()
        _winapi.CreateJunction(str(outside), str(self.vault / "Linked"))
        refused = {"code": "CONFIG_UNREADABLE", "detail": "link or special file inside the vault",
                   "status": "SEARCH_BLOCKED"}
        for spelling in (short, Path(os.fspath(self.vault).swapcase())):
            with self.subTest(spelling=os.fspath(spelling)):
                code, text = self.run_cli("search", str(self.vault), "drip", config=spelling / "Linked" / "c.json")
                self.assertEqual((code, json.loads(text)), (2, refused))
                code, text = self.run_cli("recall", str(self.vault), "--questions",
                                          str(spelling / "Linked" / "q.json"))
                self.assertIn("RECALL_BLOCKED code=RECALL_QUESTIONS_INVALID detail=link or special file", text)
                note = self.vault / "Projects" / "Garden irrigation.md"
                before = note.read_bytes()
                code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                          "--json-out", str(spelling / "Projects" / "Garden irrigation.md"))
                self.assertEqual(text.strip(), "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED "
                                               "detail=existing file or note inside the vault")
                self.assertEqual(note.read_bytes(), before)
        twin = self.hard_link(outside / "c.json", "twin.json")
        code, text = self.run_cli("search", str(self.vault), "drip", config=short / twin.name)
        self.assertEqual((code, json.loads(text)["detail"]), (2, "file with more than one name inside the vault"))
        (self.vault / "plain.json").write_text(json.dumps({"cache_dir": str(self.cache)}), encoding="utf-8")
        code, text = self.run_cli("search", str(self.vault), "drip", config=short / "plain.json")
        self.assertEqual(code, 0, text)
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(short / "Projects" / "new result.json"),
                                  config=self.config_file(exclude_dirs=["Archive", "Linked"]))
        self.assertEqual(code, 0, text)
        self.assertTrue((self.vault / "Projects" / "new result.json").is_file())

    @unittest.skipUnless(os.name == "nt", "case-sensitive folders here are a Windows (NTFS) attribute")
    def test_folder_that_differs_from_the_vault_only_in_case_is_outside(self):
        import subprocess
        parent = self.temp / "cased"
        parent.mkdir()
        done = subprocess.run(["fsutil.exe", "file", "setCaseSensitiveInfo", str(parent), "enable"],
                              capture_output=True)
        if done.returncode != 0:
            self.skipTest("cannot make a case-sensitive folder here")
        vault, other = parent / "Vault", parent / "vault"
        write_vault(vault, {"Notes/Kite.md": FRONT + "a note about kites\n"})
        other.mkdir()
        if len(list(parent.iterdir())) != 2:
            self.skipTest("the folder did not become case-sensitive")
        (other / "c.json").write_text(json.dumps({"cache_dir": str(self.cache)}), encoding="utf-8")
        code, text = self.run_cli("search", str(vault), "kites", config=other / "c.json")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text), ["Notes/Kite"])
        os.link(other / "c.json", vault / "c.json")
        code, text = self.run_cli("search", str(vault), "kites", config=other / "c.json")
        self.assertEqual(code, 0, text)  # the file outside the vault is the user's own choice
        code, text = self.run_cli("search", str(vault), "kites", config=vault / "c.json")
        self.assertEqual((code, json.loads(text)["detail"]), (2, "file with more than one name inside the vault"))
        (vault / "c.json").unlink()
        (other / "Notes").mkdir()
        for scope in (other / "Notes", other / "Missing"):
            with self.subTest(scope=scope.name):
                code, text = self.run_cli("search", str(vault), "kites", "--in", str(scope), config=other / "c.json")
                self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_SCOPE_INVALID"))
        code, text = self.run_cli("search", str(vault), "kites", "--in", str(vault / "Notes"), config=other / "c.json")
        self.assertEqual(code, 0, text)
        self.assertEqual(links(text), ["Notes/Kite"])

    def test_other_case_on_a_volume_that_keeps_the_given_spelling(self):
        """macOS-like: realpath keeps the letter case given, and the volume ignores case.

        The temporary volume must ignore case (NTFS, APFS by default); _spellings is
        replaced by one that keeps the given spelling, as POSIX realpath does.
        """
        other = self.temp / "VAULT"
        if not other.exists():
            self.skipTest("this volume tells letter case apart")
        (self.vault / "old.json").write_text("ORIGINAL", encoding="utf-8")
        kept = mock.patch.object(pages, "_spellings", lambda given: (given, given, True, False))
        with kept, mock.patch.object(pages, "_SPELLING_KEPT", True, create=True):
            for target in (other / "Projects" / "Garden irrigation.md", other / "Projects" / "New.md",
                           other / "old.json"):
                with self.subTest(target=target.name):
                    with self.assertRaises(SearchError) as caught:
                        pages.checked_file(self.vault, target, must_exist=False, new_file=True)
                    self.assertEqual(caught.exception.code, "SEARCH_OUTPUT_REFUSED")
            made = pages.checked_file(self.vault, other / "Projects" / "result.json", must_exist=False,
                                      new_file=True)
            self.assertEqual(made.parts, (self.vault / "Projects" / "result.json").parts, "spelled from the vault")
            pages.create_file(self.vault, other / "Projects" / "result.json", b"{}")
            self.assertEqual((self.vault / "Projects" / "result.json").read_bytes(), b"{}")
            with self.assertRaises(SearchError):
                pages.create_file(self.vault, other / "Projects" / "result.json", b"changed")
            self.assertEqual(pages.scope_prefix(self.vault, str(other / "Projects")), "projects/")
            outside = self.temp / "outside.json"
            self.assertEqual(pages.checked_file(self.vault, outside, must_exist=False, new_file=True), outside)
        self.assertEqual((self.vault / "old.json").read_text(encoding="utf-8"), "ORIGINAL")
        self.assertEqual(pages._SPELLING_KEPT, os.name != "nt", "Windows resolve() already gives the real case")

    def test_folder_identity_on_posix_paths(self):
        from pathlib import PurePosixPath as P
        vault = P("/work/Notes")

        def identity(same):
            numbers = {}  # one file number per path (per folded path where the volume ignores case)

            def fake_stat(path):
                text = os.fspath(path)
                if text.casefold().startswith("/missing"):
                    raise FileNotFoundError(text)
                key = text.casefold() if same else text
                return os.stat_result((0o40000, numbers.setdefault(key, len(numbers) + 1), 1, 1, 0, 0, 0, 0, 0, 0))
            return fake_stat
        with mock.patch.object(pages, "_SPELLING_KEPT", True):
            with mock.patch.object(pages.os, "stat", identity(True)):  # a volume that ignores case
                self.assertEqual(pages._under(P("/work/notes/sub/a.json"), vault, True), P("/work/Notes/sub/a.json"))
                self.assertEqual(pages._under(P("/work/Notes/a.json"), vault, True), P("/work/Notes/a.json"))
                self.assertIsNone(pages._under(P("/work/notes"), vault, True), "the vault itself is not under it")
                self.assertIsNone(pages._under(P("/missing/notes/a.json"), P("/missing/Notes"), True))
                self.assertIsNone(pages._under(P("/work/other/a.json"), vault, True))
            with mock.patch.object(pages.os, "stat", identity(False)):  # a volume that tells case apart
                self.assertIsNone(pages._under(P("/work/notes/a.json"), vault, True))
                self.assertEqual(pages._under(P("/work/Notes/a.json"), vault, True), P("/work/Notes/a.json"))

    def test_cache_key_of_posix_spellings_of_one_vault(self):
        """POSIX paths: other letter case of one folder shares the cache key; two folders do not."""
        from pathlib import PurePosixPath as P
        cfg = self.cfg()

        def volume(listing, same):
            numbers = {}

            def fake_listdir(path):
                text = os.fspath(path)
                if text not in listing:
                    raise PermissionError(text)
                return list(listing[text])

            def fake_stat(path):
                text = os.fspath(path)
                key = text.casefold() if same else text
                return os.stat_result((0o40000, numbers.setdefault(key, len(numbers) + 1), 1, 1, 0, 0, 0, 0, 0, 0))
            return mock.patch.object(pages.os, "listdir", fake_listdir), mock.patch.object(pages.os, "stat", fake_stat)

        def key(path):
            return embedding.cache_file(pages._listed_spelling(P(path)), cfg)
        listed, stat = volume({"/": ["work"], "/work": ["Notes", "Other"]}, same=True)  # ignores case
        with listed, stat:
            self.assertEqual(pages._listed_spelling(P("/WORK/notes")), P("/work/Notes"))
            self.assertEqual(key("/work/notes"), key("/work/Notes"))
            self.assertNotEqual(key("/work/Other"), key("/work/Notes"))
        listed, stat = volume({"/": ["work"], "/work": ["Notes", "notes"]}, same=False)  # tells case apart
        with listed, stat:
            self.assertNotEqual(key("/work/notes"), key("/work/Notes"))
        listed, stat = volume({"/": ["work"]}, same=True)  # /work cannot be listed
        with listed, stat:
            self.assertEqual(pages._listed_spelling(P("/work/notes")), P("/work/notes"), "kept as given")

    def test_checked_root_spells_the_vault_as_listed_where_the_given_spelling_is_kept(self):
        """macOS-like: resolve() keeps the letter case given, as POSIX realpath does.

        The temporary volume must ignore case (NTFS, APFS by default).
        """
        other = self.temp / "VAULT"
        if not other.exists():
            self.skipTest("this volume tells letter case apart")
        kept = mock.patch.object(type(self.vault), "resolve", lambda path, strict=False: Path(os.path.abspath(path)))
        with kept, mock.patch.object(pages, "_SPELLING_KEPT", True):
            root = pages.checked_root(other)
            self.assertEqual(root.name, "vault")
            self.assertEqual(embedding.cache_file(root, self.cfg()),
                             embedding.cache_file(pages.checked_root(self.vault), self.cfg()))

    @unittest.skipUnless(os.name == "nt", "8.3 short names are a Windows feature")
    def test_absolute_scope_in_other_spellings_of_the_vault(self):
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        spellings = [Path(os.fspath(self.vault).swapcase())]
        if (ctypes.windll.kernel32.GetShortPathNameW(str(self.vault), buffer, len(buffer))
                and buffer.value.casefold() != os.fspath(self.vault).casefold()):
            spellings.append(Path(buffer.value))  # 8.3 short names are on for this volume
        for spelling in spellings:
            with self.subTest(spelling=os.fspath(spelling)):
                code, text = self.run_cli("search", str(self.vault), "drip", "--in", str(spelling / "Projects"),
                                          "--top", "20")
                self.assertEqual(code, 0, text)
                self.assertTrue(links(text))
                self.assertTrue(all(link.startswith("Projects/") for link in links(text)), links(text))
        code, text = self.run_cli("search", str(self.vault), "drip", "--in", str(self.temp / "Projects"))
        self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_SCOPE_INVALID"))

    def link(self, target, name, directory=False):
        try:
            os.symlink(target, self.temp / name, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a symlink here")
        return self.temp / name

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows feature")
    def test_links_outside_the_vault_that_lead_into_it_are_refused(self):
        import _winapi
        outside = self.outside_files()
        (self.vault / "c.json").write_text(json.dumps({"cache_dir": str(self.cache)}), encoding="utf-8")
        (self.vault / "q.json").write_text((outside / "q.json").read_text(encoding="utf-8"), encoding="utf-8")
        door = self.temp / "door"
        _winapi.CreateJunction(str(self.vault), str(door))
        config_link = self.link(self.vault / "c.json", "c-link.json")
        refused = {"code": "CONFIG_UNREADABLE", "detail": "link or special file inside the vault",
                   "status": "SEARCH_BLOCKED"}
        for config in (door / "c.json", config_link):
            with self.subTest(config=config.name):
                code, text = self.run_cli("search", str(self.vault), "drip", config=config)
                self.assertEqual((code, json.loads(text)), (2, refused))
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(door / "q.json"))
        self.assertEqual(text.strip(), "RECALL_BLOCKED code=RECALL_QUESTIONS_INVALID detail=link or special file "
                                       "inside the vault")
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(door / "Projects" / "new.json"))
        self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=link or special "
                                                   "file inside the vault"))
        self.assertFalse((self.vault / "Projects" / "new.json").exists())
        self.assertEqual(self.loader.loads, 0)
        # Paths outside the vault, and the vault's own files, as before.
        code, text = self.run_cli("search", str(self.vault), "drip", config=outside / "c.json")
        self.assertEqual(code, 0, text)
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(outside / "out.json"), config=self.vault / "c.json")
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads((outside / "out.json").read_text(encoding="utf-8"))["questions"], 1)
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(self.vault / "q.json"),
                                  "--json-out", str(self.vault / "Projects" / "new.json"))
        self.assertEqual(code, 0, text)
        self.assertTrue((self.vault / "Projects" / "new.json").is_file())

    def test_result_through_a_broken_link_into_the_vault_is_refused(self):
        outside = self.outside_files()
        note = self.vault / "Notes" / "New note.md"
        second = self.link(note, "second.json")
        cases = {"broken link": self.link(note, "dangling.json"),
                 "chain of links": self.link(second, "first.json"),
                 "link to a missing folder": self.link(self.vault / "Projects", "folder-link", directory=True)
                 / "Missing" / "new.json"}
        for name, target in cases.items():
            with self.subTest(case=name):
                code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                          "--json-out", str(target))
                self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=link "
                                                           "or special file inside the vault"))
        self.assertFalse(note.exists())
        self.assertFalse((self.vault / "Projects" / "Missing").exists())
        self.assertEqual(self.loader.loads, 0)
        loop = self.link(self.temp / "loop-b.json", "loop-a.json")
        self.link(loop, "loop-b.json")
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(loop))
        self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write "
                                                   "the result file"))
        self.assertEqual(self.loader.loads, 0)
        elsewhere = self.link(self.temp / "elsewhere" / "result.json", "outside-link.json")
        (self.temp / "elsewhere").mkdir()
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(outside / "q.json"),
                                  "--json-out", str(elsewhere))
        self.assertEqual(code, 0, text)
        self.assertTrue((self.temp / "elsewhere" / "result.json").is_file(), "a link that stays outside is used")


class HardLinks(VaultCase):
    def test_note_with_two_names_is_refused_before_reading(self):
        write_vault(self.vault, {"Private/Twin.md": FRONT + "## Twin heading\nhardlinked secret about walruses\n"})
        try:
            os.link(self.vault / "Private" / "Twin.md", self.vault / "Notes" / "Twin copy.md")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a hard link here")
        code, text = self.run_cli("search", str(self.vault), "secret about walruses", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)
        self.assertIn("read_problems=2", text)
        self.assertNotIn("walruses", candidates(text))
        self.assertFalse([link for link in links(text) if "Twin" in link])


class OddNames(VaultCase):
    def test_notes_differing_only_in_case_are_both_kept(self):
        first, second = self.vault / "Projects" / "Alpha.md", self.vault / "Projects" / "alpha.md"
        with mock.patch.object(pages, "_walk", return_value=([first, second], [])):
            found, problems = pages.select_paths(pages.checked_root(self.vault), [], self.cfg())
        self.assertEqual(sorted(map(os.fspath, found)), sorted([os.fspath(first), os.fspath(second)]))

    def test_unencodable_name_is_one_read_problem(self):
        try:
            write_vault(self.vault, {"Projects/bad\udc80 name.md": FRONT + "## Bad\nsurrogate note\n"})
        except (OSError, UnicodeError):
            self.skipTest("cannot create a file with this name here")
        code, text = self.run_cli("search", str(self.vault), "drip line morning")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_PARTIAL: coverage_complete=false", text)
        self.assertIn("read_problems=1", text)
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")

    def test_link_named_like_an_excluded_folder_is_skipped(self):
        outside = self.temp / "outside"
        write_vault(outside, {"Leak.md": FRONT + "leaked text about drip lines\n"})
        try:
            os.symlink(outside, self.vault / "Projects" / "Archive", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a directory symlink here")
        code, text = self.run_cli("search", str(self.vault), "leaked drip lines", "--top", "20")
        self.assertEqual(code, 0, text)
        self.assertIn("SEARCH_COMPLETE: coverage_complete=true", text)
        self.assertNotIn("Leak", text)
        os.symlink(outside, self.vault / "Projects" / "Other link", target_is_directory=True)
        code, text = self.run_cli("search", str(self.vault), "leaked drip lines", "--top", "20")
        self.assertIn("SEARCH_PARTIAL", text)
        self.assertNotIn("Leak", text)


class LinkSpelling(VaultCase):
    """Links that would be equal without the extension keep it (output strings only)."""

    def run_of(self, ranked_rels, rels):
        from vault_search_lib import markdown as md
        from vault_search_lib import search
        root = pages.checked_root(self.vault)
        made = {rel: md.Page(root / rel, rel, Path(rel).stem, [], "", "x", {}) for rel in rels}
        results = [(0.9 - n / 100, md.Chunk(made[rel], "", "text of " + rel)) for n, rel in enumerate(ranked_rels)]
        return search.SearchRun(root, list(made.values()), len(made), [], [("q", results)])

    def lines(self, run, mode):
        from vault_search_lib import search
        out = []
        search.emit(run, mode, self.cfg(hide_snippets_when=[], private_dirs=[]), 10, write=out.append)
        return [line.split(" (", 1)[0] for line in out if line[:1].isdigit()]

    def test_notes_that_differ_only_in_the_extension_are_told_apart(self):
        rels = ["Case/Note.md", "Case/Note.MD", "Case/Note.md.md", "Other/Plain.md"]
        for mode in ("details", "paths"):
            with self.subTest(mode=mode):
                run = self.run_of(rels, rels)
                self.assertEqual(self.lines(run, mode), ["1. [[Case/Note.md]]", "2. [[Case/Note.MD]]",
                                                         "3. [[Case/Note.md.md]]", "4. [[Other/Plain]]"])
                run = self.run_of(["Case/Note.MD"], rels)
                self.assertEqual(self.lines(run, mode), ["1. [[Case/Note.MD]]"], "the same link whatever --top is")
        run = self.run_of(["Other/Plain.md", "Case/Note.md"], ["Other/Plain.md", "Case/Note.md"])
        self.assertEqual(self.lines(run, "details"), ["1. [[Other/Plain]]", "2. [[Case/Note]]"])

    def test_links_compare_with_the_exact_spelling(self):
        rels = ["A/Note.md", "A/note.md"]
        self.assertEqual(self.lines(self.run_of(rels, rels), "details"), ["1. [[A/Note]]", "2. [[A/note]]"])
        rels = ["Case/Note.md", "Case/Note.md.md"]  # without Note.MD no two links are equal
        self.assertEqual(self.lines(self.run_of(rels, rels), "details"), ["1. [[Case/Note]]", "2. [[Case/Note.md]]"])


class RuleValues(VaultCase):
    def search(self, rule):
        code, text = self.run_cli("search", str(self.vault), "watering greenhouse balcony plants", "--top", "20",
                                  config=self.config_file(hide_snippets_when=[rule]))
        self.assertEqual(code, 0, text)
        return candidates(text)

    def test_rule_values_are_trimmed_like_note_values(self):
        shown = self.search({"field": "sensitivity", "values": ["  HIGH "]})
        self.assertNotIn("greenhouse", shown)
        self.assertIn("balcony plants", shown)
        shown = self.search({"field": "sensitivity", "allow": [" low  "]})
        self.assertIn("balcony plants", shown)
        self.assertNotIn("greenhouse", shown)

    def test_rule_values_compare_after_nfc_and_case(self):
        write_vault(self.vault, {"Notes/Accent.md": "---\nsensitivity: Café\n---\n## Plan\nwatering the cafe ferns\n"})
        shown = self.search({"field": "sensitivity", "values": ["CAFÉ"]})
        self.assertNotIn("cafe ferns", shown)
        self.assertIn("balcony plants", shown)
        self.assertIn("cafe ferns", self.search({"field": "sensitivity", "values": ["high"]}))
        shown = self.search({"field": "sensitivity", "allow": []})
        for text in ("cafe ferns", "balcony plants", "greenhouse"):
            self.assertNotIn(text, shown, "an empty allow list hides every page")

    def test_values_of_several_lines_compare_in_one_form(self):
        write_vault(self.vault, {"Notes/Quoted.md": '---\nsensitivity: "one\\ntwo"\n---\n## Plan\nwatering the quoted ferns\n',
                                 "Notes/Block.md": "---\nsensitivity: |\n  one\n  two\n---\n## Plan\nwatering the block ferns\n"})
        for rule_value in ("one\ntwo", "  one \n two\n", "one two"):
            with self.subTest(value=repr(rule_value)):
                shown = self.search({"field": "sensitivity", "values": [rule_value]})
                self.assertNotIn("quoted ferns", shown)
                self.assertNotIn("block ferns", shown)
                self.assertIn("balcony plants", shown)
                shown = self.search({"field": "sensitivity", "allow": [rule_value, "low"]})
                self.assertIn("quoted ferns", shown)
                self.assertIn("block ferns", shown)
                self.assertNotIn("greenhouse", shown)
        shown = self.search({"field": "sensitivity", "values": ["one"]})
        self.assertIn("quoted ferns", shown, "a part of the value is not the value")


class MissingDependencies(VaultCase):
    def test_missing_yaml_or_markdown_parser_stops_before_any_note_is_read(self):
        from vault_search_lib import markdown as md
        questions = self.temp / "q.json"
        questions.write_text(json.dumps({"questions": [{"id": "q1", "query": "drip line",
                                                        "gold": ["Projects/Garden irrigation.md"]}]}),
                             encoding="utf-8")
        blocked = json.dumps({"code": "SEARCH_DEPENDENCY_MISSING", "status": "SEARCH_BLOCKED"}, sort_keys=True)
        read = []
        real_read = pages.read_page

        def counting(vault, path):
            read.append(path)
            return real_read(vault, path)
        for module in ("yaml", "MarkdownIt"):
            for argv, expected in ((("search", str(self.vault), "drip"), blocked), (("warm", str(self.vault)), blocked),
                                   (("recall", str(self.vault), "--questions", str(questions)),
                                    "RECALL_BLOCKED code=SEARCH_DEPENDENCY_MISSING")):
                with self.subTest(missing=module, command=argv[0]):
                    with mock.patch.object(md, module, None), mock.patch.object(pages, "read_page", counting):
                        code, text = self.run_cli(*argv)
                    self.assertEqual((code, text.strip()), (2, expected))
                    self.assertEqual((read, self.loader.loads), ([], 0))
        with mock.patch.object(pages, "read_page", counting):
            code, text = self.run_cli("search", str(self.vault), "drip")
        self.assertEqual(code, 0, text)
        self.assertTrue(read)


class CliArguments(VaultCase):
    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(list(argv), loader=self.loader)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_bad_arguments_print_one_fixed_line(self):
        for argv in (["search", str(self.vault), "secret question", "--top", "three"], ["nonsense"],
                     ["recall", str(self.vault)]):
            with self.subTest(argv=argv[:2]):
                code, out, err = self.call(*argv)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out), {"code": "SEARCH_ARGS_INVALID", "status": "SEARCH_BLOCKED"})
                self.assertNotIn("three", out + err)
                self.assertNotIn("secret", out + err)

    def test_help_exits_zero(self):
        code, out, _ = self.call("search", "--help")
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)
        self.assertIn("links with similarity and label fields", " ".join(out.split()))


class BackendChecks(VaultCase):
    def test_fake_loader_needs_only_numpy(self):
        with mock.patch.object(embedding, "BACKEND_MODULES", ("numpy", "vss_missing_backend_module")):
            code, text = self.run_cli("search", str(self.vault), "drip line morning")
            self.assertEqual(code, 0, text)
            code, text = self.run_cli("search", str(self.vault), "drip", loader=embedding.load_encoder)
            self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_DEPENDENCY_MISSING"))


class CacheIntegrity(VaultCase):
    def warm(self, vault=None):
        code, text = self.run_cli("warm", str(vault or self.vault))
        self.assertEqual(code, 0, text)
        return json.loads(text)

    def path(self, vault=None):
        return embedding.cache_file(pages.checked_root(vault or self.vault), self.cfg())

    def arrays(self):
        with np.load(self.path()) as data:
            return {name: data[name] for name in data.files}

    def rewrite(self, **arrays):
        with open(self.path(), "wb") as stream:
            np.savez(stream, **arrays)

    def test_damaged_or_old_cache_is_rebuilt(self):
        first = self.warm()
        good = self.arrays()
        model = str(good["model"])

        def sealed(vecs, keys=None):
            keys = good["keys"] if keys is None else keys
            return dict(good, keys=keys, vecs=vecs, digest=np.array(embedding._digest(np, model, keys, vecs)))
        swapped = good["vecs"].copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        nan_row, zero_row = good["vecs"].copy(), good["vecs"].copy()
        nan_row[0, 0] = np.nan
        zero_row[0] = 0
        duplicate = good["keys"].copy()
        duplicate[1] = duplicate[0]
        cases = {"swapped rows": dict(good, vecs=swapped), "nan": sealed(nan_row), "zero vector": sealed(zero_row),
                 "duplicate key": sealed(good["vecs"], duplicate), "short rows": sealed(good["vecs"][:-1]),
                 "old format": {"model": good["model"], "keys": good["keys"], "vecs": good["vecs"]}}
        for name, arrays in cases.items():
            with self.subTest(case=name):
                self.rewrite(**arrays)
                self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
        self.path().write_bytes(self.path().read_bytes()[:100])
        self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
        self.assertEqual(self.warm()["new_inputs"], 0)
        self.rewrite(**dict(self.arrays(), vecs=swapped))
        code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")

    def damage_item(self, name):
        """Replace one item of the npz with bytes that are not a NumPy array (the zip stays readable)."""
        import zipfile
        path = self.path()
        with zipfile.ZipFile(path) as source:
            members = {info.filename: source.read(info.filename) for info in source.infolist()}
        members[name + ".npy"] = b"not a numpy array"
        with zipfile.ZipFile(path, "w") as target:
            for member, data in members.items():
                target.writestr(member, data)

    def test_damaged_item_is_rebuilt(self):
        first = self.warm()
        for name in ("keys", "vecs"):
            with self.subTest(item=name):
                self.damage_item(name)
                with np.load(self.path(), allow_pickle=False) as data:
                    self.assertIsInstance(data[name], bytes)  # what NumPy gives for a damaged item
                self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
                self.assertEqual(self.warm()["new_inputs"], 0)
                self.damage_item(name)
                code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
                self.assertEqual(code, 0, text)
                self.assertEqual(links(text)[0], "Projects/Garden irrigation")
                self.assertEqual(self.warm()["new_inputs"], 0, "the search saved the rebuilt cache")

    def test_format_must_be_the_integer_two(self):
        first = self.warm()
        good = self.arrays()
        for name, value in {"2.9": np.array(2.9), "text 2": np.array("2"), "list [2]": np.array([2]),
                            "true": np.array(True)}.items():
            with self.subTest(format=name):
                self.rewrite(**dict(good, format=value))
                self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
        self.assertEqual(self.warm()["new_inputs"], 0, "the format the writer stores is used as it is")
        self.assertEqual(self.arrays()["format"].dtype.kind, "i")

    def test_model_name_or_digest_that_is_not_valid_text_is_rebuilt(self):
        first = self.warm()
        good = self.arrays()
        # A text item whose one character is the code point 0x110000, above the last one Unicode has.
        broken = np.frombuffer(np.array([0x110000], dtype="<u4").tobytes(), dtype="<U1").reshape(())
        for name in ("model", "digest"):
            with self.subTest(item=name):
                self.rewrite(**dict(good, **{name: broken}))
                self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
                self.assertEqual(self.warm()["new_inputs"], 0)
                self.rewrite(**dict(self.arrays(), **{name: broken}))
                code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
                self.assertEqual(code, 0, text)
                self.assertEqual(links(text)[0], "Projects/Garden irrigation")
                good = self.arrays()

    def test_sealed_cache_with_key_bytes_that_are_not_ascii_is_rebuilt(self):
        first = self.warm()
        good = self.arrays()
        keys = good["keys"].copy()
        keys[0] = b"\xff" * 40
        self.rewrite(**dict(good, keys=keys,
                            digest=np.array(embedding._digest(np, str(good["model"]), keys, good["vecs"]))))
        self.assertEqual(self.warm()["new_inputs"], first["new_inputs"])
        self.assertEqual(self.warm()["new_inputs"], 0)

    def test_repeated_input_is_embedded_once(self):
        write_vault(self.vault, {"Dup/Twice.md": FRONT + "## One\nsame sentence about otters\n"
                                                         "## Two\nsame sentence about otters\n"})
        first = self.warm()
        texts = [text for batch in self.loader.passage_batches() for text in batch]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(first["new_inputs"], first["embedding_inputs"] - 1)
        code, text = self.run_cli("search", str(self.vault), "same sentence about otters")
        self.assertEqual(links(text)[0], "Dup/Twice")

    def test_cache_folder_follows_the_real_spelling(self):
        composed, decomposed = self.temp / "Café", self.temp / "Café"
        for root, word in ((composed, "pelican"), (decomposed, "flamingo")):
            write_vault(root, {"Note.md": FRONT + f"a note about the {word}\n"})
        if len({p.name for p in self.temp.iterdir()} & {"Café", "Café"}) < 2:
            self.skipTest("this file system merges the two Unicode forms")
        self.assertNotEqual(self.path(composed), self.path(decomposed))
        if os.name == "nt":
            self.assertEqual(self.path(str(self.vault).swapcase()), self.path())
        self.assertGreater(self.warm(composed)["new_inputs"], 0)
        self.assertGreater(self.warm(decomposed)["new_inputs"], 0)
        self.assertEqual(self.warm(composed)["new_inputs"], 0, "the other vault's warm run kept this cache")


class WarmFailures(VaultCase):
    def cache_keys(self):
        with np.load(embedding.cache_file(pages.checked_root(self.vault), self.cfg())) as data:
            return data["keys"].astype(str).tolist()

    def test_save_failure_stops_warm_but_not_search(self):
        with mock.patch.object(embedding, "_save", side_effect=PermissionError("denied")):
            code, text = self.run_cli("warm", str(self.vault))
            self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_CACHE_WRITE_FAILED"))
            code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
            self.assertEqual(code, 0, text)
            self.assertEqual(links(text)[0], "Projects/Garden irrigation")
            import sys
            saved = sys.stdout, sys.stderr
            try:
                sys.stdout = sys.stderr = None
                code = cli.main(["warm", str(self.vault), "--config", str(self.config_file())], loader=self.loader)
            finally:
                sys.stdout, sys.stderr = saved
            self.assertEqual(code, 2)

    def test_warm_drops_vectors_when_no_section_is_left(self):
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        self.assertTrue(self.cache_keys())
        for path in self.vault.rglob("*.md"):
            path.write_text("# Only a heading\n", encoding="utf-8")
        loads = self.loader.loads
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        self.assertEqual((json.loads(text)["status"], json.loads(text)["searched_chunks"]), ("WARM_COMPLETE", 0))
        self.assertEqual(self.cache_keys(), [])
        self.assertEqual(self.loader.loads, loads)

    def cache_path(self):
        return embedding.cache_file(pages.checked_root(self.vault), self.cfg())

    def reset_vault(self):
        shutil.rmtree(self.vault)
        self.vault.mkdir()
        write_vault(self.vault, self.files)

    def test_warm_cleans_the_cache_when_no_note_is_left(self):
        def delete():
            for path in list(self.vault.rglob("*.md")):
                path.unlink()

        def blank():
            for path in self.vault.rglob("*.md"):
                path.write_text("  \n\n", encoding="utf-8")
        for name, empty in (("deleted", delete), ("blank", blank)):
            with self.subTest(case=name):
                self.reset_vault()
                code, text = self.run_cli("warm", str(self.vault))
                self.assertEqual(code, 0, text)
                self.assertTrue(self.cache_keys())
                empty()
                loads = self.loader.loads
                code, text = self.run_cli("warm", str(self.vault))
                self.assertEqual(code, 0, text)
                self.assertEqual((json.loads(text)["status"], json.loads(text)["searched_pages"]),
                                 ("WARM_COMPLETE", 0))
                self.assertEqual(self.cache_keys(), [])
                self.assertEqual(self.loader.loads, loads)

    def test_search_of_an_empty_vault_cleans_the_cache_then_says_empty(self):
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        for path in list(self.vault.rglob("*.md")):
            path.unlink()
        code, text = self.run_cli("search", str(self.vault), "drip")
        self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_SCOPE_EMPTY"))
        self.assertEqual(self.cache_keys(), [])

    def test_old_cache_is_replaced_when_only_titles_are_left(self):
        code, text = self.run_cli("warm", str(self.vault))
        with np.load(self.cache_path()) as data:
            old = {"model": data["model"], "keys": data["keys"], "vecs": data["vecs"]}
        with open(self.cache_path(), "wb") as stream:
            np.savez(stream, **old)  # the format before format 2
        for path in self.vault.rglob("*.md"):
            path.write_text("# Only a heading\n", encoding="utf-8")
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        with np.load(self.cache_path()) as data:
            self.assertEqual((int(data["format"]), len(data["keys"])), (embedding.CACHE_FORMAT, 0))

    def test_no_cache_file_is_made_for_a_vault_without_sections(self):
        for path in self.vault.rglob("*.md"):
            path.write_text("# Only a heading\n", encoding="utf-8")
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        self.assertFalse(self.cache_path().exists())

    def run_with_stderr(self, *argv, config):
        out, err = io.StringIO(), io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), _environ(env):
            code = cli.main([*argv, "--config", str(config)], loader=self.loader)
        return code, out.getvalue(), err.getvalue()

    def test_cache_write_failures_are_reported_by_kind(self):
        warning = "warning: could not save the embedding cache"
        config = self.config_file(cache_dir=str(self.temp / "bad\x00dir"))
        code, out, err = self.run_with_stderr("warm", str(self.vault), config=config)
        self.assertEqual((code, json.loads(out)["code"]), (2, "SEARCH_CACHE_WRITE_FAILED"))
        code, out, err = self.run_with_stderr("search", str(self.vault), "how long does the drip line run each morning",
                                              config=config)
        self.assertEqual(code, 0, out)
        self.assertEqual(links(out)[0], "Projects/Garden irrigation")
        self.assertIn(warning, err)
        with mock.patch.object(embedding, "_save", side_effect=PermissionError("denied")):
            code, out, err = self.run_with_stderr("search", str(self.vault), "drip line", config=self.config_file())
        self.assertEqual(code, 0, out)
        self.assertIn(warning, err)
        self.assertNotIn("embedding", err.replace(warning, ""), "search prints no progress")

    def test_paths_with_a_home_folder_that_cannot_be_found(self):
        """`~` when the home folder cannot be determined: each path keeps its own failure code."""
        env = {k: v for k, v in os.environ.items() if k not in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME")}
        with _environ(env):
            try:
                Path("~").expanduser()
                self.skipTest("the home folder is still found without these variables")
            except RuntimeError:
                pass
            config = self.config_file(cache_dir="~/cache")
            code, out, err = self.run_with_stderr("search", str(self.vault), "how long does the drip line run each morning",
                                                  config=config)
            self.assertEqual(code, 0, out)
            self.assertEqual(links(out)[0], "Projects/Garden irrigation")
            self.assertIn("warning: could not save the embedding cache", err)
            code, out, err = self.run_with_stderr("warm", str(self.vault), config=config)
            self.assertEqual((code, json.loads(out)["code"]), (2, "SEARCH_CACHE_WRITE_FAILED"))
            for name in list(self.vault.rglob("*.md")):
                name.unlink()
            code, out, err = self.run_with_stderr("warm", str(self.vault), config=config)
            self.assertEqual((code, json.loads(out)["code"]), (2, "SEARCH_CACHE_WRITE_FAILED"),
                             "a cache that cannot be found cannot be cleaned either")
            write_vault(self.vault, self.files)
            questions = self.temp / "q.json"
            questions.write_text(json.dumps({"questions": [{"id": "q1", "query": "drip line",
                                                            "gold": ["Projects/Garden irrigation.md"]}]}),
                                 encoding="utf-8")
            loads = self.loader.loads
            code, text = self.run_cli("recall", str(self.vault), "--questions", str(questions),
                                      "--json-out", "~/result.json")
            self.assertEqual((code, text.strip()),
                             (2, "RECALL_BLOCKED code=RECALL_OUTPUT_FAILED detail=cannot write the result file"))
            self.assertEqual(self.loader.loads, loads, "refused before the model is loaded")
            code, text = self.run_cli("recall", str(self.vault), "--questions", "~/q.json")
            self.assertEqual((code, text.strip()), (2, "RECALL_BLOCKED code=RECALL_QUESTIONS_INVALID "
                                                       "detail=cannot read the question file as JSON"))
            code, text = self.run_cli("search", str(self.vault), "drip", config="~/c.json")
            self.assertEqual((code, json.loads(text)["code"]), (2, "CONFIG_UNREADABLE"))
            code, text = self.run_cli("search", "~/vault", "drip")
            self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_VAULT_NOT_FOUND"))

    def test_partial_scan_without_sections_and_without_a_home_folder(self):
        """No cache folder can be named and a read problem leaves no section: warm still fails, search warns."""
        removed = ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "LOCALAPPDATA", "XDG_CACHE_HOME")
        env = {k: v for k, v in os.environ.items() if k not in removed}
        real_scandir = os.scandir
        inside = os.path.normcase(str(self.vault))

        def fake_scandir(path="."):
            if os.path.normcase(os.fspath(path)).startswith(inside):
                raise PermissionError(os.fspath(path))
            return real_scandir(path)
        with _environ(env):
            try:
                Path.home()
                self.skipTest("the home folder is still found without these variables")
            except RuntimeError:
                pass
            config = self.config_file(cache_dir=None)
            with mock.patch.object(pages.os, "scandir", fake_scandir):
                code, out, err = self.run_with_stderr("warm", str(self.vault), config=config)
                self.assertEqual((code, json.loads(out).get("code")), (2, "SEARCH_CACHE_WRITE_FAILED"), out)
                code, out, err = self.run_with_stderr("search", str(self.vault), "drip line", config=config)
            self.assertEqual(code, 0, out)
            self.assertIn("SEARCH_PARTIAL: coverage_complete=false", out)
            self.assertIn("warning: could not save the embedding cache", err)

    def test_partial_scan_keeps_the_vectors_of_unread_notes(self):
        code, text = self.run_cli("warm", str(self.vault))
        before = set(self.cache_keys())
        try:
            os.link(self.vault / "Journal" / "Spring trip.md", self.vault / "Journal" / "Twin.md")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a hard link here")
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual((code, json.loads(text)["status"]), (0, "WARM_PARTIAL"))
        self.assertEqual(set(self.cache_keys()), before)


class BadRowEncoder(FakeEncoder):
    """Gives `bad` for texts that contain `marker`, the fake vectors otherwise."""

    def __init__(self, bad, marker="", width=256):
        super().__init__()
        self.bad, self.marker, self.dimension = bad, marker, width

    def encode(self, texts, **kwargs):
        texts = list(texts)
        out = super().encode(texts, **kwargs)
        rows = [i for i, text in enumerate(texts) if self.marker in text and not text.startswith("query: ")]
        out = np.array(out, dtype="float32")
        for i in rows:
            out[i] = self.bad
        return out


class NewVectors(VaultCase):
    def cache_path(self):
        return embedding.cache_file(pages.checked_root(self.vault), self.cfg())

    BAD = {"zero": np.zeros(256), "nan": np.full(256, np.nan), "infinite": np.full(256, np.inf),
           "not unit length": np.full(256, 0.5)}

    def test_bad_model_output_stops_before_it_is_saved_or_ranked(self):
        for name, row in self.BAD.items():
            for argv in (("warm", str(self.vault)), ("search", str(self.vault), "drip line")):
                with self.subTest(vector=name, command=argv[0]):
                    loader = FakeLoader()
                    loader.encoder = BadRowEncoder(row)
                    code, text = self.run_cli(*argv, loader=loader)
                    self.assertEqual((code, json.loads(text)), (2, {"code": "SEARCH_FAILED",
                                                                    "status": "SEARCH_BLOCKED"}))
                    self.assertFalse(self.cache_path().exists())

    def test_good_batches_saved_before_a_bad_one_are_kept(self):
        path = self.temp / "cache" / "unit" / "search.npz"
        texts = ["passage: a\nfirst good text", "passage: b\nsecond good text", "passage: c\nthird BAD text"]
        model = BadRowEncoder(np.zeros(256), marker="BAD")
        with self.assertRaises(SearchError) as caught:
            embedding.embed(path, "m", texts, lambda: model, prune=False, step=2)
        self.assertEqual(caught.exception.code, "SEARCH_FAILED")
        known, state = embedding._load(path, "m")
        self.assertEqual((state, len(known)), (embedding.USABLE, 2))
        model = BadRowEncoder(np.zeros(256), marker="never")
        vectors, new = embedding.embed(path, "m", texts, lambda: model, prune=False, step=2)
        self.assertEqual((vectors.shape, new), ((3, 256), 1))

    def test_bad_question_vector_stops_search_and_recall(self):
        questions = self.temp / "q.json"
        questions.write_text(json.dumps({"questions": [{"id": "q1", "query": "drip line",
                                                        "gold": ["Projects/Garden irrigation.md"]}]}),
                             encoding="utf-8")

        class BadQuestion(FakeEncoder):
            def __init__(self, row):
                super().__init__()
                self.row = row

            def encode(self, texts, **kwargs):
                texts = list(texts)
                out = super().encode(texts, **kwargs)
                if texts and texts[0].startswith("query: "):
                    return np.tile(np.asarray(self.row, dtype="float32"), (len(texts), 1))
                return out
        two_wide = np.zeros(2)
        two_wide[0] = 1
        for name, row in {**self.BAD, "other width": two_wide}.items():
            with self.subTest(vector=name):
                loader = FakeLoader()
                loader.encoder = BadQuestion(row)
                code, text = self.run_cli("search", str(self.vault), "drip line", loader=loader)
                self.assertEqual((code, json.loads(text)), (2, {"code": "SEARCH_FAILED", "status": "SEARCH_BLOCKED"}))
                code, text = self.run_cli("recall", str(self.vault), "--questions", str(questions), loader=loader)
                self.assertEqual((code, text.strip()),
                                 (2, "RECALL_BLOCKED code=RECALL_SEARCH_BLOCKED detail=SEARCH_FAILED"))
        code, text = self.run_cli("recall", str(self.vault), "--questions", str(questions))
        self.assertEqual(code, 0, text)
        self.assertIn("RECALL_COMPLETE questions=1 top1=1/1", text)

    def test_question_vector_is_ranked_as_it_was_checked(self):
        """A question vector inside the unit-length tolerance only as float16 is ranked as that float16 row."""
        from vault_search_lib import search

        class NearUnit(FakeEncoder):
            dimension = 2

            def encode(self, texts, **kwargs):
                texts = list(texts)
                self.batches.append(texts)
                return np.array([[1.0101, 0.0] if text.startswith("query: ") else [1.0, 0.0] for text in texts],
                                dtype="float32")
        loader = FakeLoader()
        loader.encoder = NearUnit()
        run = search.run_queries(pages.checked_root(self.vault), ["zzqx"], [], 5, self.cfg(), loader=loader)
        scores = [score for score, _ in run.queries[0][1] if score is not None]
        self.assertTrue(scores)
        self.assertEqual(scores[0], float(np.float32(np.float16(1.0101))))

    def test_one_letter_text_of_the_fake_encoder_is_a_unit_vector(self):
        vector = FakeEncoder().encode(["passage: T\nx"])[0]
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


class ConcurrentSaves(VaultCase):
    """Two runs that read the same cache and add other rows (the second run starts inside the first)."""

    def setUp(self):
        super().setUp()
        self.path = self.temp / "cache" / "unit" / "search.npz"

    def keys(self):
        known, state = embedding._load(self.path, "m")
        self.assertEqual(state, embedding.USABLE)
        return set(known)

    @staticmethod
    def key(text):
        import hashlib
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def run_inside(self, outer, inner, outer_prune=False, step=5000):
        encoder, started = FakeEncoder(), []
        path = self.path

        class Interleaved(FakeEncoder):
            def encode(self, texts, **kwargs):
                if not started:
                    started.append(True)
                    embedding.embed(path, "m", inner, lambda: encoder, prune=False)
                return super().encode(texts, **kwargs)
        model = Interleaved()
        return embedding.embed(path, "m", outer, lambda: model, prune=outer_prune, step=step)

    def test_both_runs_keep_their_rows(self):
        embedding.embed(self.path, "m", ["passage: x\nshared text"], lambda: FakeEncoder(), prune=False)
        outer, inner = ["passage: a\nfirst run text"], ["passage: b\nsecond run text"]
        self.run_inside(outer, inner)
        self.assertEqual(self.keys(), {self.key(t) for t in ["passage: x\nshared text", *outer, *inner]})
        vectors, new = embedding.embed(self.path, "m", outer + inner, lambda: FakeEncoder(), prune=False)
        self.assertEqual(new, 0)

    def test_progress_saves_keep_the_other_rows(self):
        outer = ["passage: a\nfirst run one", "passage: a\nfirst run two", "passage: a\nfirst run three"]
        inner = ["passage: b\nsecond run text"]
        self.run_inside(outer, inner, step=1)
        self.assertEqual(self.keys(), {self.key(t) for t in outer + inner})

    def test_only_the_final_save_of_a_whole_vault_run_drops_rows(self):
        outer, inner = ["passage: a\nwhole vault text"], ["passage: b\nother run text"]
        self.run_inside(outer, inner, outer_prune=True)
        self.assertEqual(self.keys(), {self.key(outer[0])})

    def test_partial_save_does_not_bring_back_rows_another_run_dropped(self):
        embedding.embed(self.path, "m", ["passage: x\nold row"], lambda: FakeEncoder(), prune=False)
        started, path = [], self.path

        class Interleaved(FakeEncoder):
            def encode(self, texts, **kwargs):
                if not started:
                    started.append(True)
                    embedding.embed(path, "m", [], None, prune=True)  # a whole-vault run that keeps nothing
                return super().encode(texts, **kwargs)
        model = Interleaved()
        embedding.embed(path, "m", ["passage: a\nthis run"], lambda: model, prune=False)
        self.assertEqual(self.keys(), {self.key("passage: a\nthis run")})

    def test_other_width_saved_meanwhile_stops_the_run(self):
        started, path = [], self.path

        class TwoWide(FakeEncoder):
            dimension = 2

            def encode(self, texts, **kwargs):
                return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(list(texts)), 1))

        class Interleaved(FakeEncoder):
            def encode(self, texts, **kwargs):
                if not started:
                    started.append(True)
                    embedding.embed(path, "m", ["passage: b\nother width"], lambda: TwoWide(), prune=False)
                return super().encode(texts, **kwargs)
        model = Interleaved()
        with self.assertRaises(SearchError) as caught:
            embedding.embed(path, "m", ["passage: a\nthis run"], lambda: model, prune=False)
        self.assertEqual(caught.exception.code, "SEARCH_FAILED")
        self.assertEqual(self.keys(), {self.key("passage: b\nother width")}, "the other run's file is kept")

    def test_a_held_lock_is_a_failed_save(self):
        warning = "warning: could not save the embedding cache"
        cache = embedding.cache_file(pages.checked_root(self.vault), self.cfg())
        with mock.patch.object(embedding, "LOCK_TIMEOUT", 0.2), embedding._locked(cache):
            code, text = self.run_cli("warm", str(self.vault))
            self.assertEqual((code, json.loads(text)["code"]), (2, "SEARCH_CACHE_WRITE_FAILED"))
            out, err = io.StringIO(), io.StringIO()
            env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), _environ(env):
                code = cli.main(["search", str(self.vault), "how long does the drip line run each morning",
                                 "--config", str(self.config_file())], loader=self.loader)
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(links(out.getvalue())[0], "Projects/Garden irrigation")
            self.assertIn(warning, err.getvalue())
        self.assertFalse(cache.exists())
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual((code, json.loads(text)["status"]), (0, "WARM_COMPLETE"))

    def test_the_lock_holds_across_processes(self):
        import subprocess
        import sys
        import time
        from support import ROOT
        ready = self.temp / "ready"
        script = ("import sys, time, pathlib\nsys.path.insert(0, sys.argv[1])\n"
                  "from vault_search_lib import embedding\n"
                  "with embedding._locked(pathlib.Path(sys.argv[2])):\n"
                  "    pathlib.Path(sys.argv[3]).write_text('1')\n    time.sleep(3)\n")
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", script, str(ROOT), str(self.path), str(ready)])
        self.addCleanup(child.wait)
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(ready.exists(), "the other process did not take the lock")
        with mock.patch.object(embedding, "LOCK_TIMEOUT", 0.2):
            with self.assertRaises(OSError):
                with embedding._locked(self.path):
                    pass
        child.wait()
        with embedding._locked(self.path):
            pass


class NoWidthEncoder(FakeEncoder):
    dimension = None


class CacheWidth(VaultCase):
    def cache_path(self):
        return embedding.cache_file(pages.checked_root(self.vault), self.cfg())

    def narrow_cache(self):
        """Seal the cache again with 2-wide vectors under the same model name."""
        with np.load(self.cache_path()) as data:
            arrays = {name: data[name] for name in data.files}
        vecs = np.zeros((len(arrays["keys"]), 2), dtype="float16")
        vecs[:, 0] = 1
        arrays.update(vecs=vecs, digest=np.array(embedding._digest(np, str(arrays["model"]), arrays["keys"], vecs)))
        with open(self.cache_path(), "wb") as stream:
            np.savez(stream, **arrays)
        return self.cache_path().read_bytes()

    def test_same_model_name_with_another_width_stops_before_the_cache_changes(self):
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(code, 0, text)
        before = self.narrow_cache()
        failed = {"code": "SEARCH_FAILED", "status": "SEARCH_BLOCKED"}
        for argv in (("warm", str(self.vault)), ("search", str(self.vault), "drip")):
            with self.subTest(command=argv[0]):
                code, text = self.run_cli(*argv)
                self.assertEqual((code, json.loads(text)), (2, failed))
                self.assertEqual(self.cache_path().read_bytes(), before)
        write_vault(self.vault, {"Projects/New.md": FRONT + "## New\na new note about herons\n"})
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual((code, json.loads(text)), (2, failed), "some inputs hit, some are new")
        self.assertEqual(self.cache_path().read_bytes(), before)
        self.loader.encoder = NoWidthEncoder()
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual((code, json.loads(text)), (2, failed), "an encoder that declares no width")
        self.assertEqual(self.cache_path().read_bytes(), before)

    def test_encoder_without_a_declared_width_embeds_each_input_once(self):
        self.loader.encoder = NoWidthEncoder()
        code, text = self.run_cli("warm", str(self.vault))
        first = json.loads(text)
        texts = [t for batch in self.loader.passage_batches() for t in batch]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(len(texts), first["new_inputs"])
        self.assertEqual(first["new_inputs"], first["embedding_inputs"])
        code, text = self.run_cli("warm", str(self.vault))
        self.assertEqual(json.loads(text)["new_inputs"], 0)
        code, text = self.run_cli("search", str(self.vault), "how long does the drip line run each morning")
        self.assertEqual(links(text)[0], "Projects/Garden irrigation")

    @unittest.skipUnless(os.environ.get("VAULT_SEARCH_MODEL_DIR"), "set VAULT_SEARCH_MODEL_DIR to the model folder")
    def test_onnx_encoder_declares_its_width(self):
        from vault_search_lib import config as config_mod
        encoder = embedding.load_encoder(config_mod.from_dict({"model_dir": os.environ["VAULT_SEARCH_MODEL_DIR"]}))
        self.assertEqual(encoder.dimension, 384)
        self.assertEqual(encoder.encode(["passage: drip line"]).shape, (1, 384))


if __name__ == "__main__":
    unittest.main()
