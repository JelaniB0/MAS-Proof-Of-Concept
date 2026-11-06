"""Offline checks for audited inputs, model selection and the controller feature contract."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import MPNetModel

from .audit import audit_prepared
from .data import GROUPS, prepare
from .evaluation import summarize, gain_diagnostics, utility_diagnostics, paired_interval, safe_spearman
from .frozen import FrozenMPNet, PoolHead, FrozenEnsemblePredictor, export_ensemble
from .inputs import question_text, input_text, contract_id, controller_features, GSM_DEMO_HEADER
from .model import DifficultyPredictor
from .research import run_suite
from .test_transformer import tiny_encoder, tiny_tokenizer


def local_encoder(path):
    config = tiny_encoder().config
    config.max_position_embeddings = 514
    MPNetModel(config).save_pretrained(path, safe_serialization=True)
    tiny_tokenizer().save_pretrained(path)


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_recognized_demonstrations_are_removed(self):
        prompt = GSM_DEMO_HEADER + "\nQuestion: example\nAnswer: 9\nA new question?"
        self.assertEqual(question_text(prompt), "A new question?")
        self.assertEqual(question_text("A user question\nAnswer: explain this"), "A user question\nAnswer: explain this")
        self.assertEqual(question_text("Write a function to add lists."), "Write a function to add lists.")
        self.assertEqual(question_text(prompt + "\nAnswer: part of user question"), "A new question?\nAnswer: part of user question")
        self.assertEqual(input_text(prompt, "full"), prompt)
        with self.assertRaises(ValueError):
            input_text(prompt, "unknown")

    def test_controller_contract_has_exact_order_and_signed_gains(self):
        class Predictor:
            manifest = {"groups": {"weak": ["a"], "intermediate": ["b"], "strong": ["c"]}}
            def predict(self, texts):
                return [{"strong": 0.5, "weak": 0.25, "intermediate": 0.0} for text in texts]
        predictor = Predictor()
        result = controller_features(predictor, ["question"])[0]
        self.assertEqual(result["difficulty_vector"], [0.25, 0, 0.5])
        self.assertEqual(result["estimated_quality"]["strong"], 0.5)
        self.assertEqual(result["estimated_quality_gains"]["weak_to_strong"], -0.25)
        modified = {"groups": {"weak": ["a"], "intermediate": ["c"], "strong": ["b"]}}
        self.assertNotEqual(contract_id(predictor.manifest), contract_id(modified))
        with self.assertRaises(ValueError):
            controller_features(predictor, ["question"], expected_contract=contract_id(modified))

    def test_missing_labels_and_no_evidence_metrics_remain_finite_or_null(self):
        rows = [{"eval_name": "a"}, {"eval_name": "b"}]
        truth = np.array([[np.nan, 0, 1], [np.nan, 1, 0]], dtype=np.float32)
        prediction = np.full((2, 3), 0.5)
        report = summarize(prediction, truth, rows)
        self.assertIsNone(report["weak"]["mae"])
        self.assertIsNone(report["weak"]["prediction_std"])
        self.assertEqual(gain_diagnostics(prediction, truth, rows)["all"]["n"], 0)
        self.assertEqual(utility_diagnostics(prediction, truth)["points"], [])
        json.dumps(report, allow_nan=False)

    def test_paired_bootstrap_and_utility_oracle(self):
        rows = [{"eval_name": "a"}] * 5 + [{"eval_name": "b"}] * 5
        truth = np.tile([0.8, 0.5, 0.1], (10, 1))
        better = truth.copy()
        worse = np.full_like(truth, 0.5)
        interval = paired_interval(worse, better, truth, rows, repetitions=100)
        self.assertGreater(interval["ci95"][0], 0)
        report = utility_diagnostics(truth, truth)
        self.assertTrue(all(abs(point["oracle_regret"]) < 1e-9 for point in report["points"]))

    def test_constant_float32_correlations_are_null_not_nan(self):
        constant = np.full(1156, 0.357953, dtype=np.float32)
        self.assertIsNone(safe_spearman(constant, np.arange(1156)))
        self.assertIsNone(safe_spearman(np.arange(1156), constant))

    def test_frozen_cache_and_self_contained_ensemble(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "encoder"
            local_encoder(source)
            extractor = FrozenMPNet(source, device="cpu", max_length=32, threads=1)
            rows = [{"prompt": "question add"}, {"prompt": "question divide"}]
            values, _ = extractor.cached_encode(rows, root / "cache", "question")
            cached, _ = extractor.cached_encode(rows, root / "cache", "question")
            np.testing.assert_array_equal(values, cached)
            self.assertTrue(all(not p.requires_grad for p in extractor.encoder.parameters()))
            head_files = []
            for seed in [1, 2]:
                torch.manual_seed(seed)
                head = PoolHead(values.shape[1], hidden_dim=8)
                path = root / f"head_{seed}.pt"
                torch.save(head.state_dict(), path)
                head_files.append(path)
            manifest = {"groups": {"weak": ["a"], "intermediate": ["b"], "strong": ["c"]}}
            metadata = {"input_mode": "question", "input_dim": 16, "hidden_dim": 8, "dropout": 0.2,
                        "max_length": 32, "encoder_weights_sha256": extractor.weights_sha256,
                        "encoder_configuration_sha256": extractor.configuration_sha256,
                        "tokenizer_sha256": extractor.tokenizer_sha256}
            destination = root / "exported"
            export_ensemble(source, head_files, destination, metadata, manifest)
            model = DifficultyPredictor.load(destination)
            predictions = model.predict(["question add"])
            self.assertEqual(set(predictions[0]), set(GROUPS))
            self.assertEqual(predictions, model.predict(["question add"]))
            self.assertEqual(model.features(["question add"])[0]["difficulty"], predictions[0])
            self.assertEqual(model.predict([]), [])
            # A different valid tensor checkpoint must not keep the same predictor identity.
            original_head = (destination / "head_0.pt").read_bytes()
            changed = torch.load(destination / "head_0.pt", weights_only=True)
            changed["layers.0.bias"] = changed["layers.0.bias"] + 0.1
            torch.save(changed, destination / "head_0.pt")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                FrozenEnsemblePredictor.load(destination)
            (destination / "head_0.pt").write_bytes(original_head)
            # Detect altered input processing rather than quietly using stale artifacts.
            configuration = destination / "encoder" / "tokenizer_config.json"
            value = json.loads(configuration.read_text(encoding="utf-8"))
            value["test_revision"] = "changed"
            configuration.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                FrozenEnsemblePredictor.load(destination)

    def test_small_end_to_end_study_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, prepared, encoder, output = root / "source.jsonl", root / "data", root / "encoder", root / "study"
            local_encoder(encoder)
            frame = pd.DataFrame([{"sample_id": str(i), "eval_name": "math" if i % 3 else "code",
                                   "prompt": f"question {'add' if i % 2 else 'divide'} {i}",
                                   "a": i % 2, "b": 0.5, "c": 1} for i in range(120)])
            frame.to_json(source, orient="records", lines=True)
            prepare(source, prepared)
            self.assertEqual(audit_prepared(prepared)["question_only_cross_split_overlaps"], 0)
            report = run_suite(prepared, encoder, output, seeds=(42,), epochs=1, patience=1, device="cpu")
            self.assertEqual(len(report["protocol"]["configurations"]), 4)
            self.assertTrue((output / "selection.json").exists())
            self.assertTrue((output / "selected_model" / "test_predictions.jsonl").exists())
            self.assertIsNotNone(report["test"]["selected_frozen_ensemble"]["prediction"]["macro_mse"])
            with patch("BAMAS.difficulty.research.fit_head", side_effect=AssertionError("Should reuse finished heads")), \
                 patch.object(FrozenMPNet, "encode", side_effect=AssertionError("Should reuse valid cached embeddings")):
                resumed = run_suite(prepared, encoder, output, seeds=(42,), epochs=1, patience=1, device="cpu")
                self.assertEqual(resumed["selected_configuration"], report["selected_configuration"])
            with self.assertRaises(ValueError):
                run_suite(prepared, encoder, output, seeds=(42,), epochs=2, patience=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
