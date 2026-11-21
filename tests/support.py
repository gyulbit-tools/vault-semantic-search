"""Shared helpers: a synthetic vault and a fake encoder (no model, no download)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vault_search_lib import config as config_mod  # noqa: E402

_spec = importlib.util.spec_from_file_location("vault_search_cli", ROOT / "vault_search.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(text) + ([0, 0] if add_special_tokens else [])


class FakeEncoder:
    """Character-bigram hashing: texts that share words get a high cosine similarity."""
    tokenizer = FakeTokenizer()
    max_seq_length = 512
    dimension = 256  # declared like the ONNX encoder's, so a fully cached run embeds no passage

    def __init__(self):
        self.batches = []

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True):
        texts = list(texts)
        self.batches.append(texts)
        out = np.zeros((len(texts), 256), dtype="float32")
        for row, text in enumerate(texts):
            body = text.split("\n", 1)[-1].casefold()
            for a, b in zip(body, body[1:]):
                if a.isspace() or b.isspace():
                    continue
                out[row, zlib.crc32((a + b).encode("utf-8")) % 256] += 1
            if not out[row].any():  # no character pair (a one-letter text): still a unit vector, as a model gives
                out[row, zlib.crc32(body.encode("utf-8")) % 256] = 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class FakeLoader:
    def __init__(self):
        self.loads = 0
        self.encoder = FakeEncoder()

    def __call__(self, cfg):
        self.loads += 1
        return self.encoder

    def passage_batches(self):
        return [b for b in self.encoder.batches if b and not b[0].startswith("query: ")]


FRONT = "---\ntype: note\n---\n"

STANDARD_FILES = {
    "Projects/Garden irrigation.md": (
        "---\ntype: project\nkeywords: [drip, sprinkler]\n---\n"
        "# Garden irrigation\n\n## Timer\nThe drip line runs for twenty minutes every morning before sunrise.\n"
        "## Parts\nPressure reducer, filter, quarter inch tubing and emitters.\n"),
    "Projects/Bicycle repair.md": FRONT + "## Brakes\nReplace worn brake pads and adjust the cable tension.\n",
    "Journal/Spring trip.md": FRONT + "## Day one\nWe took the early train to the coast and walked along the cliffs.\n",
    "Recipes/Kimchi stew.md": FRONT + "## 끓이는 법\n김치와 돼지고기를 볶은 뒤 물을 붓고 두부를 넣어 끓인다.\n",
    "Private/Health log.md": FRONT + "## Sleep\nSlept badly, woke up at three, drip of the tap kept me awake.\n",
    "Notes/Hidden by field.md": "---\nsensitivity: high\n---\n## Plan\nThe drip watering schedule for the greenhouse.\n",
    "Notes/Shown by field.md": "---\nsensitivity: low\n---\n## Plan\nA watering can for the balcony plants.\n",
    "Archive/Old irrigation.md": FRONT + "## Timer\nThe old drip line ran for twenty minutes every morning.\n",
    ".obsidian/workspace.md": "drip line timer sunrise\n",
    "Projects/Empty.md": "",
}


def write_vault(root, files):
    for rel, text in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")


class VaultCase(unittest.TestCase):
    files = STANDARD_FILES
    settings = {"exclude_dirs": ["Archive"], "private_dirs": ["Private"],
                "hide_snippets_when": [{"field": "sensitivity", "values": ["high"]}]}

    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="vss-test-"))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.vault = self.temp / "vault"
        self.vault.mkdir()
        write_vault(self.vault, self.files)
        self.cache = self.temp / "cache"
        self.loader = FakeLoader()

    def config_file(self, extra=None, **overrides):
        data = dict(self.settings)
        data.update(overrides)
        if extra:
            data.update(extra)
        data.setdefault("cache_dir", str(self.cache))
        path = self.temp / "config.json"
        import json
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def cfg(self, **overrides):
        data = dict(self.settings)
        data.update(overrides)
        data.setdefault("cache_dir", str(self.cache))
        return config_mod.from_dict(data)

    def run_cli(self, *argv, config=None, loader=None):
        config = config if config is not None else self.config_file()
        out = io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAULT_SEARCH_")}
        with contextlib.redirect_stdout(out), _environ(env):
            code = cli.main([*argv[:1], *argv[1:], "--config", str(config)], loader=loader or self.loader)
        return code, out.getvalue()


class WithoutLibyaml:
    """Mixin: the test class runs with PyYAML's C extension hidden, as on a machine without libyaml.

    The markdown module is loaded again with `yaml.CSafeLoader` missing, and
    loaded again as installed after the class.
    """
    @classmethod
    def setUpClass(cls):
        import importlib
        import yaml
        from vault_search_lib import markdown
        cls._libyaml = yaml.__dict__.pop("CSafeLoader", None)
        importlib.reload(markdown)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        import importlib
        import yaml
        from vault_search_lib import markdown
        super().tearDownClass()
        if cls._libyaml is not None:
            yaml.CSafeLoader = cls._libyaml
        importlib.reload(markdown)


@contextlib.contextmanager
def _environ(values):
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
