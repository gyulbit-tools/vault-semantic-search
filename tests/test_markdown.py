"""Front matter, chunks and embedding windows (no files, no model)."""
from __future__ import annotations

import re
import time
import unittest
from pathlib import Path

import support  # noqa: F401  (puts the project folder on sys.path)

from vault_search_lib import markdown as md


def page(body, meta=None, name="Synthetic"):
    return md.Page(Path(name + ".md"), name + ".md", name, [], "", body, meta or {})


def chunks(body):
    return md.make_chunks(page(body))


class FrontMatter(unittest.TestCase):
    def test_split_and_parse(self):
        front, body = md.split_frontmatter("---\ntitle: A\ntags: [x, \"y\"]\naliases:\n  - One\n  - Two\n"
                                           "summary: >\n  first\n  second\nempty:\n---\nBody\n")
        self.assertEqual(body, "Body\n")
        meta = md.parse_front(front)
        self.assertEqual(meta["title"], "A")
        self.assertEqual(meta["tags"], ["x", "y"])
        self.assertEqual(meta["aliases"], ["One", "Two"])
        self.assertEqual(meta["summary"], "first second")
        self.assertEqual(meta["empty"], [])

    def test_no_front_matter(self):
        self.assertEqual(md.split_frontmatter("# Title\n---\ntext\n"), ("", "# Title\n---\ntext\n"))
        self.assertEqual(md.split_frontmatter("---\nnot closed\n"), ("", "---\nnot closed\n"))

    def test_yaml_comment_quoted_key_and_indented_marker(self):
        self.assertEqual(md.parse_front("sensitivity: high # why it is high\n")["sensitivity"], "high")
        self.assertEqual(md.parse_front('"sensitivity": high\n')["sensitivity"], "high")
        front, body = md.split_frontmatter("---\nnote: |\n  first\n  ---\n  second\nsensitivity: high\n---\nBody\n")
        self.assertEqual(body, "Body\n")
        meta = md.parse_front(front)
        self.assertEqual(meta["sensitivity"], "high")
        self.assertEqual(meta["note"], "first --- second")

    def test_scalars_stay_as_written(self):
        meta = md.parse_front("private: true\nversion: 1.10\nupdated: 2026-01-02\nnothing: null\n"
                              "items: [a, 'b c']\nliteral: |\n  one\n  two\n")
        self.assertEqual((meta["private"], meta["version"], meta["updated"], meta["nothing"]),
                         ("true", "1.10", "2026-01-02", "null"))
        self.assertEqual(meta["items"], ["a", "b c"])
        self.assertEqual(meta["literal"], "one two")

    def test_unreadable_front_matter_is_reported(self):
        for front in ("title: a: b\n", "- a list\n- not a mapping\n", "key: [unclosed\n", "\tkey: tab\n"):
            with self.subTest(front=front):
                with self.assertRaises(md.FrontMatterError):
                    md.read_front(front)
        meta, unread = md.read_front("sensitivity:\n  level: high\ntitle: T\ntags: [a, [b]]\n")
        self.assertEqual(meta, {"title": "T"})
        self.assertEqual(unread, {"sensitivity", "tags"})

    def test_only_a_dash_line_closes_front_matter(self):
        text = "---\nsensitivity: low\n...\nBody\n"
        self.assertEqual(md.split_frontmatter(text), ("", text))
        self.assertEqual(md.frontmatter_parts(text), ("", text, True))
        self.assertEqual(md.frontmatter_parts("---\nnot closed\n"), ("", "---\nnot closed\n", True))
        self.assertEqual(md.frontmatter_parts("# Title\n---\ntext\n"), ("", "# Title\n---\ntext\n", False))
        self.assertEqual(md.frontmatter_parts("---\na: b\n---\nBody\n"), ("a: b\n", "Body\n", False))
        self.assertEqual(md.frontmatter_parts("---\n---\nBody\n"), ("", "Body\n", False))

    def test_key_given_twice_is_unread(self):
        for front in ("sensitivity: low\nsensitivity: high\ntitle: T\n",
                      '"sensitivity ": low\nsensitivity: high\ntitle: T\n',
                      "sensitivité: low\nsensitivité: high\ntitle: T\n"):
            with self.subTest(front=front):
                meta, unread = md.read_front(front)
                self.assertEqual(meta, {"title": "T"})
                self.assertEqual(len(unread), 1)

    def test_one_line_is_the_form_values_compare_in(self):
        for value, expected in ((" one\n  two ", "one two"), ("one", "one"), ("a\n\n  b\nc", "a b c"),
                                ("  ", ""), ("a  b", "a  b")):
            with self.subTest(value=value):
                self.assertEqual(md.one_line(value), expected)

    def test_field_helpers(self):
        p = page("", {"a": ["one", "two"], "b": "true", "c": ""})
        self.assertEqual(md.field_value(p, "a"), "one")
        self.assertEqual(md.field_values(p, "a"), ["one", "two"])
        self.assertTrue(md.is_true(md.field_value(p, "b")))
        self.assertEqual(md.field_values(p, "c"), [])
        self.assertEqual(md.field_values(p, "missing"), [])


class PlainYaml:
    """Only plain YAML is read (no tag, anchor, alias or merge key), and lines end only at "\\n".

    Run twice: as installed (with libyaml where PyYAML has it) and with libyaml
    hidden; the results must be the same.
    """
    REFUSED = {
        "null tag": "level: !!null sample\n",
        "null tag on an allowed value": "level: !!null low\n",
        "binary tag": "level: !!binary '????'\n",
        "valid int tag": "a: !!int 10\n",
        "valid str tag": "e: !!str 007\n",
        "valid bool tag": "b: !!bool true\n",
        "tag that does not fit": "a: !!bool maybe\n",
        "local tag": "level: !custom x\n",
        "non-specific tag": "level: ! x\n",
        "tag in a list": "f: [!!int 1, two]\n",
        "tag on the whole mapping": "!!seq {level: low}\n",
        "string tag on a mapping": "other: !!str {a: 1}\ntitle: T\n",
        "merge key": "<<: {level: high}\ntitle: T\n",
        "merge key from an anchor": "base: &b {level: high}\n<<: *b\n",
        "nested merge key": "outer:\n  inner:\n    <<: {level: high}\ntitle: T\n",
        "unused anchor": "level: &x low\n",
        "alias": "first: &f low\nlevel: *f\n",
        "control character": "level: low\x01\n",
        "parsed only by libyaml": "k: [foo?bar]\n",
    }

    def test_tags_anchors_aliases_and_merge_keys_make_the_front_matter_unreadable(self):
        for name, front in self.REFUSED.items():
            with self.subTest(case=name):
                with self.assertRaises(md.FrontMatterError):
                    md.read_front(front)

    def test_plain_lookalikes_are_read_as_text(self):
        meta, unread = md.read_front("'<<': kept\nlevel: low # a << b, &c, *d, !e\n"
                                     'note: "!!int 10 &a *b <<"\nplain: a&b and c!d\n')
        self.assertEqual(meta, {"<<": "kept", "level": "low", "note": "!!int 10 &a *b <<", "plain": "a&b and c!d"})
        self.assertEqual(unread, frozenset())

    def test_text_only_the_python_parser_reads_is_one_unread_field(self):
        self.assertEqual(md.read_front("k: {x:}\ntitle: T\n"), ({"title": "T"}, frozenset({"k"})))

    def test_closing_line_ends_only_at_lf(self):
        for separator in ("\u2028", "\u2029", "\x85", "\r"):
            for where, text in (("before", f"---\nlevel: low{separator}---\nlevel: high\n---\nBody\n"),
                                ("after", f"---\nlevel: low\n---{separator}level: high\n---\nBody\n")):
                with self.subTest(separator=hex(ord(separator)), where=where):
                    front, body, unclosed = md.frontmatter_parts(text)
                    self.assertEqual((body, unclosed), ("Body\n", False))
                    self.assertIn("level: high", front)
                    with self.assertRaises(md.FrontMatterError):  # two documents
                        md.read_front(front)

    def test_first_line_that_is_dashes_only_after_other_white_space_is_unclear(self):
        for first in ("---\u2028", "---\u2029", "---\x85", "---\u00a0", " ---", "\t---"):
            with self.subTest(first=repr(first)):
                text = first + "\nsensitivity: low\n---\nBody\n"
                self.assertTrue(md.opening_unclear(text))
                self.assertEqual(md.frontmatter_parts(text), ("sensitivity: low\n", "Body\n", False))
        for text in ("---\na: b\n---\n", "--- \t\na: b\n---\n", "---", "# Title\n", "", "----\n", "x ---\n"):
            with self.subTest(text=repr(text)):
                self.assertFalse(md.opening_unclear(text))

    def test_line_separators_and_control_characters_make_the_front_matter_unreadable(self):
        # YAML would end a line at the first four, so `sensitivity` would become a field of its own.
        for char in ("\u2028", "\u2029", "\x85", "\r", "\x0b", "\x0c", "\x1b", "\x7f", "\x9f"):
            with self.subTest(char=hex(ord(char))):
                with self.assertRaises(md.FrontMatterError):
                    md.read_front(f"title: clipped{char}sensitivity: low\n")

    def test_white_space_only_front_matter_is_unreadable(self):
        for front in ("\t\n", "  \n", "\n", " \t \n\n", "\u00a0\n"):
            with self.subTest(front=repr(front)):
                with self.assertRaises(md.FrontMatterError):
                    md.read_front(front)

    def test_empty_front_matter_and_tabs_in_values_are_read(self):
        self.assertEqual(md.read_front(""), ({}, frozenset()))
        front, body, unclosed = md.frontmatter_parts("---\n---\nBody\n")
        self.assertEqual((md.read_front(front), body, unclosed), (({}, frozenset()), "Body\n", False))
        self.assertEqual(md.read_front('a: "b\tc"\nd: e  # note\n'), ({"a": "b\tc", "d": "e"}, frozenset()))
        self.assertEqual(md.read_front('a: "one\\ntwo"\n')[0], {"a": "one two"})

    def test_dash_lines_that_close(self):
        self.assertEqual(md.frontmatter_parts("---\na: b\n--- \t\nBody\n"), ("a: b\n", "Body\n", False))
        self.assertEqual(md.frontmatter_parts("---\na: b\n---"), ("a: b\n", "", False))
        self.assertEqual(md.frontmatter_parts("---\na: b\n ---\nBody\n"), ("", "---\na: b\n ---\nBody\n", True))
        self.assertEqual(md.read_front("a: b\nc: [d, e]\n"), ({"a": "b", "c": ["d", "e"]}, frozenset()))


class PlainYamlAsInstalled(PlainYaml, unittest.TestCase):
    pass


class PlainYamlWithoutLibyaml(support.WithoutLibyaml, PlainYaml, unittest.TestCase):
    pass


class Chunks(unittest.TestCase):
    def test_one_chunk_per_heading_section(self):
        result = chunks("Intro line here.\n# Title\n## First\nAlpha text.\n## Second\nBeta text.\n")
        self.assertEqual([(c.heading, c.text) for c in result],
                         [("", "Intro line here."), ("First", "Alpha text."), ("Second", "Beta text.")])

    def test_thematic_break_only_sections_have_no_chunk(self):
        for line in ("---", "***", "___", "- - -", " * * * ", "   _\t_\t_", "-" * 1200):
            with self.subTest(line=line[:20]):
                self.assertEqual(chunks("## Empty\n" + line + "\n"), [])

    def test_formatting_only_body_has_no_chunk(self):
        for body in ("> > > > > >", "| :-- | --: |", "###\n>\n| | |\n- [ ]", "\n\t\n"):
            with self.subTest(body=body):
                self.assertEqual(chunks(body), [])

    def test_short_and_symbolic_bodies_are_kept(self):
        for body in ("Return: +0%.", "0 1 2 3 4 5", "--- a real claim stays.", "∅ ⊂ ∞ ⇒ ∅ ⊆ ∞", "∞" * 12):
            with self.subTest(body=body):
                self.assertEqual([c.text for c in chunks(body)], [body])

    def test_separator_does_not_carry_an_empty_heading(self):
        result = chunks("## Empty heading\n---\n## Real heading\nReturn: +0%.\n")
        self.assertEqual([(c.heading, c.text) for c in result], [("Real heading", "Return: +0%.")])

    def test_code_and_links(self):
        body = ("## Wrapper\nValue `QZX-7731` stays.\n```\nFENCED_TOKEN_9Q\n```\n"
                "See [[Folder/Other page#Part|the other page]] and [[Folder/Plain page]].\n"
                "Read [the guide](https://example.com/guide) ![img](pic.png) ![[embed.png]].\n"
                "- [[Catalog only]]\n- [[Catalog with note]] - short note\n")
        text = "\n".join(c.text for c in chunks(body))
        self.assertIn("QZX-7731", text)
        self.assertNotIn("FENCED_TOKEN_9Q", text)
        self.assertIn("the other page", text)
        self.assertIn("Plain page", text)
        self.assertNotIn("Folder/", text)
        self.assertIn("the guide", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("pic.png", text)
        self.assertNotIn("embed", text)
        self.assertNotIn("Catalog", text)

    def test_code_blocks_follow_commonmark(self):
        cases = {
            "indented": "Intro line.\n\n    INDENTED_TOKEN\n\nAfter kept.\n",
            "longer closing fence": "Intro line.\n```\nFENCE_TOKEN\n````\nAfter kept.\n",
            "tilde fence": "Intro line.\n~~~python\nTILDE_TOKEN\n~~~\nAfter kept.\n",
            "fence in a list": "- item\n\n  ```\n  LIST_TOKEN\n  ```\n\nAfter kept.\n",
            "fence in a quote": "> ```\n> QUOTE_TOKEN\n> ```\n\nAfter kept.\n",
            "code in a list item": "- item\n\n      LIST_CODE_TOKEN\n\nAfter kept.\n",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                text = "\n".join(c.text for c in chunks(body))
                self.assertNotIn("_TOKEN", text)
                self.assertIn("After kept.", text)

    def test_indented_lines_that_are_not_code_are_kept(self):
        body = ("Para line\n    lazy continuation stays\n\n- parent\n    - nested item stays\n"
                "Value `INLINE_7` and [[Folder/Linked page]].\n")
        text = "\n".join(c.text for c in chunks(body))
        for kept in ("lazy continuation stays", "nested item stays", "INLINE_7", "Linked page"):
            self.assertIn(kept, text)


class Headings(unittest.TestCase):
    # The rule the titles must keep; heading_title replaces it because it backtracks cubically.
    ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")

    def test_titles_keep_the_atx_rule(self):
        for line in ("# Title", "## Title ##", "# C#", "#", "# ", "####### x", "  ### Spaced   ###  ",
                     "#\tTab", "#notitle", "# a # b #", "    # four spaces", "###### six", "# a##  ##"):
            with self.subTest(line=line):
                match = self.ATX.match(line)
                self.assertEqual(md.heading_title(line), match.group(2) if match else None)

    def test_long_space_runs_stay_fast(self):
        start = time.perf_counter()
        md.make_chunks(page("intro\n# a" + " " * 1600 + "b\nbody text\n"))
        self.assertLess(time.perf_counter() - start, 1.0, "heading line")
        start = time.perf_counter()
        md.is_markdown_scaffold("- " + " " * 40000 + "x")
        self.assertLess(time.perf_counter() - start, 1.0, "list line")
        self.assertTrue(md.is_markdown_scaffold("-   [ ]  "))
        self.assertFalse(md.is_markdown_scaffold("- [ ] task"))


class MarkdownLinks(unittest.TestCase):
    # The rule the links must keep; drop_markdown_links replaces it because it is quadratic on a line of `[`.
    LINK = re.compile(r"(!?)\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))+\)")

    def reference(self, text):
        return self.LINK.sub(lambda m: "" if m.group(1) else m.group(2), text)

    def test_same_result_as_the_link_rule(self):
        cases = ("Read [the guide](https://example.com/guide) now.", "![img](pic.png) gone", "[a](b) and [c](d)",
                 "[wiki](https://en.wikipedia.org/wiki/A_(b)) kept", "[a [b](c) inner", "[a](b]c) bracket",
                 "[x]() empty", "[x] (y) apart", "![a](b) [c](d) ![e](f)", "!![a](b)", "[a](b(c)d)e)",
                 "[a](b(c)", "[a](b)(c)", "[[a]](b)", "]( [x](y)", "text [x](y\nz) next", "- [x](y) list")
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(md.drop_markdown_links(text), self.reference(text))

    def test_same_result_on_random_lines(self):
        import random
        rng = random.Random(7)
        for _ in range(4000):
            text = "".join(rng.choice("[]()!ab \n") for _ in range(rng.randint(0, 30)))
            self.assertEqual(md.drop_markdown_links(text), self.reference(text), repr(text))

    def test_long_unfinished_links_stay_fast(self):
        for line in ("[" * 40000, "[" * 40000 + "]", "[" * 40000 + "](", "[a" * 20000 + "](b",
                     "x](" * 13000 + "[" * 1000):
            with self.subTest(line=line[:6] + "..." + line[-4:]):
                start = time.perf_counter()
                md.make_chunks(page("intro\n" + line + "\n"))
                self.assertLess(time.perf_counter() - start, 1.0)


class CharTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(text) + ([0, 0] if add_special_tokens else [])


class Windows(unittest.TestCase):
    def test_windows_overlap_and_cover_the_text(self):
        text = "".join(chr(97 + i % 26) for i in range(1234))
        windows = list(md.retrieval_windows(text))
        self.assertTrue(all(len(w) <= 500 for w in windows))
        self.assertEqual(windows[0], text[:500])
        self.assertEqual(windows[1][:100], text[400:500])
        self.assertTrue(text.endswith(windows[-1]))

    def test_token_budget_cuts_windows(self):
        text = "x" * 300
        windows = list(md.retrieval_windows(text, tokenizer=CharTokenizer(), prefix="h" * 20, max_tokens=64))
        self.assertTrue(all(len("h" * 20 + w) + 2 <= 64 for w in windows))
        self.assertTrue(text.endswith(windows[-1]))
        with self.assertRaises(ValueError):
            list(md.retrieval_windows(text, tokenizer=CharTokenizer(), prefix="h" * 70, max_tokens=64))

    def test_inputs_carry_prefix_and_title(self):
        class Model:
            tokenizer = CharTokenizer()
            max_seq_length = 512
        result = md.make_chunks(page("## A\nalpha text\n", name="T" * 50))
        texts, owners, views = md.retrieval_inputs(result, Model(), "passage: ")
        self.assertEqual(texts, ["passage: " + "T" * 40 + "\nalpha text"])
        self.assertEqual(owners, [0])
        self.assertEqual(views, ["alpha text"])


if __name__ == "__main__":
    unittest.main()
