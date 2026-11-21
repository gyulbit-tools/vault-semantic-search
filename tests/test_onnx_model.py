"""Smoke test of the real ONNX encoder. Skipped unless VAULT_SEARCH_MODEL_DIR names the model folder."""
from __future__ import annotations

import os
import unittest

import support  # noqa: F401

from vault_search_lib import config as config_mod
from vault_search_lib import embedding

MODEL_DIR = os.environ.get("VAULT_SEARCH_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR, "set VAULT_SEARCH_MODEL_DIR to the folder with model.onnx and tokenizer.json")
class OnnxModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = embedding.load_encoder(config_mod.from_dict({"model_dir": MODEL_DIR}))

    def test_vectors_are_normalised_and_meaningful(self):
        import numpy as np
        vectors = self.encoder.encode([
            "query: how do I water the garden automatically",
            "passage: A drip irrigation timer waters the plants every morning.",
            "passage: 정원에 물을 자동으로 주는 점적 관개 타이머",
            "passage: The quarterly tax return is due at the end of the month."])
        self.assertEqual(vectors.shape[0], 4)
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4))
        similar, korean, unrelated = vectors[1:] @ vectors[0]
        self.assertGreater(similar, unrelated)
        self.assertGreater(korean, unrelated)

    def test_trailing_whitespace_does_not_change_tokens(self):
        tok = self.encoder.tokenizer
        self.assertEqual(tok.encode("drip line"), tok.encode("drip line \n"))
        self.assertEqual(tok.encode("drip line"), tok.encode("drip line​"))

    def test_long_input_is_cut_to_the_budget(self):
        vectors = self.encoder.encode(["passage: " + "word " * 2000])
        self.assertEqual(vectors.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
