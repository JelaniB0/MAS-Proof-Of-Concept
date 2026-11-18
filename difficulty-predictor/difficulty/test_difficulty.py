"""Regression checks: python -m unittest difficulty.test_difficulty -v."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from .data import (GROUPS, add_targets, assign_splits, normalize, prepare,
                   prompt_text, rank_groups, score, validate_groups)
from .model import DifficultyPredictor, load_splits, masked_mse, train
from .safe_pickle import Node, parse_data, read_routerbench


class DifficultyTests(unittest.TestCase):
    def test_three_pool_example(self):
        groups = {"weak": ["w1", "w2", "w3", "w4"],
                  "intermediate": ["m1", "m2", "m3", "m4"],
                  "strong": ["s1", "s2", "s3", "s4"]}
        scores = dict.fromkeys(groups["weak"], 0)
        scores.update(dict(zip(groups["intermediate"], [0, 0, 1, 1])))
        scores.update(dict.fromkeys(groups["strong"], 1))
        row = add_targets([{"scores": scores}], groups)[0]
        self.assertEqual(list(row["difficulty"].values()), [1, 0.5, 0])

    def test_missing_is_not_failure_and_no_ordering_forced(self):
        groups = {"weak": ["a", "b"], "intermediate": ["c"], "strong": ["d"]}
        row = add_targets([{"scores": {"a": 1, "c": 1, "d": 0}}], groups)[0]
        self.assertIsNone(row["difficulty"]["weak"])
        self.assertEqual(row["coverage"]["weak"], 1)
        self.assertEqual(row["difficulty"]["strong"], 1)
        self.assertEqual(row["difficulty"]["intermediate"], 0)

    def test_repeated_trials_average_per_model(self):
        frame = pd.DataFrame([{"sample_id": "q", "eval_name": "d", "prompt": "question",
                               "model_name": model, "performance": quality}
                              for model, quality in [("a", 0), ("a", 1), ("b", 1)]])
        row = normalize(frame)[0]
        self.assertEqual(row["scores"], {"a": 0.5, "b": 1})
        self.assertEqual(row["trials"], {"a": 2, "b": 1})

    def test_metadata_and_response_not_targets(self):
        frame = pd.DataFrame([{"sample_id": "q", "eval_name": "d", "prompt": "question",
                               "a": True, "a|model_response": "gold leak", "a|total_cost": 0.5,
                               "gold_response": "answer", "oracle_model_to_route_to": "a"}])
        self.assertEqual(normalize(frame)[0]["scores"], {"a": 1})

    def test_duplicate_wide_rows_and_mixed_prompts_rejected(self):
        row = {"sample_id": "q", "eval_name": "d", "prompt": "question", "a": 1}
        with self.assertRaises(ValueError):
            normalize(pd.DataFrame([row, row]))
        with self.assertRaises(ValueError):
            normalize(pd.DataFrame([row, {**row, "prompt": "different shot setting"}]))

    def test_duplicate_id_and_prompt_components_share_split(self):
        rows = [{"sample_id": str(i), "prompt": f"Question {i}"} for i in range(120)]
        rows += [{"sample_id": "0", "prompt": "Another formatting of question zero"},
                 {"sample_id": "another-id", "prompt": "Question   0"}]
        rows = assign_splits(rows)
        self.assertEqual(rows[0]["split"], rows[-1]["split"])
        self.assertEqual(rows[0]["split"], rows[-2]["split"])
        reordered = assign_splits(list(reversed(rows)))
        self.assertEqual({r["prompt"]: r["split"] for r in rows},
                         {r["prompt"]: r["split"] for r in reordered})

    def test_group_ranking_uses_training_only(self):
        training = [{"split": "train", "eval_name": "d", "scores": {"a": 0, "b": 0.5, "c": 1}}]
        groups, _ = rank_groups(training + [{"split": "test", "eval_name": "d",
                                             "scores": {"a": 1, "b": 0.5, "c": 0}}])
        self.assertEqual(groups, {"weak": ["a"], "intermediate": ["b"], "strong": ["c"]})

    def test_overlapping_or_incomplete_groups_rejected(self):
        with self.assertRaises(ValueError):
            validate_groups({"weak": ["a"], "intermediate": ["a"], "strong": ["c"]}, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            validate_groups({"weak": ["a"], "intermediate": ["b"], "strong": ["c"]}, ["a", "b", "c", "d"])

    def test_bounded_scores_and_chat_formats(self):
        self.assertIsNone(score(np.nan))
        for value in [-0.1, 1.1, float("inf")]:
            with self.assertRaises(ValueError):
                score(value)
        self.assertEqual(prompt_text('[{"role":"user","content":"Q"}]'), "Q")
        self.assertEqual(prompt_text("['line one\\nline two']"), "line one\nline two")

    def test_missing_targets_have_no_gradient(self):
        import torch
        prediction = torch.tensor([[0.25, 0.5, 0.75]], requires_grad=True)
        truth = torch.tensor([[0, float("nan"), 1]])
        masked_mse(prediction, truth).backward()
        self.assertEqual(prediction.grad[0, 1].item(), 0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_data_only_reader_does_not_invoke_reducers(self):
        # Encodes builtins.eval("1/0"). A real unpickler would execute the call.
        payload = b'\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x031/0\x85R.'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inert.pkl"
            path.write_bytes(payload)
            result = parse_data(path)
            self.assertIsInstance(result, Node)
            self.assertEqual(result.name, "builtins.eval")
            with self.assertRaises(ValueError):
                read_routerbench(path)

    def test_data_only_reader_table_values(self):
        # Match the NumPy-backed layout used in the official 2024 export.
        with pd.option_context("future.infer_string", False):
            frame = pd.DataFrame({"sample_id": ["q1", "q2"], "prompt": ["hello", "world"],
                                  "eval_name": ["d", "d"], "model": [0.25, 1.0]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.pkl"
            frame.to_pickle(path, protocol=5)
            recovered = read_routerbench(path)
            self.assertEqual(list(frame.columns), list(recovered.columns))
            self.assertEqual(frame.to_dict("list"), recovered.to_dict("list"))

    def test_training_inference_reload_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            frame = pd.DataFrame([{"sample_id": str(i), "eval_name": "toy", "prompt": f"Compute example {i} with {'addition' if i % 2 else 'division'}",
                                   "a": i % 2, "b": 0.5, "c": 1} for i in range(120)])
            frame.to_json(source, orient="records", lines=True)
            prepared, output = root / "prepared", root / "model"
            prepare(source, prepared)
            report = train(prepared, output, epochs=2, patience=2, hidden_dim=8)
            self.assertIsNotNone(report["test"]["macro_mse"])
            first = DifficultyPredictor.load(output).predict(["Compute addition"])
            second = DifficultyPredictor.load(output).predict(["Compute addition"])
            self.assertEqual(first, second)
            self.assertEqual(set(first[0]), set(GROUPS))
            self.assertTrue(all(0 <= v <= 1 for v in first[0].values()))
            # Deliberately contaminate held-out data with a training question.
            train_line = (prepared / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            with (prepared / "test.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(train_line + "\n")
            with self.assertRaises(ValueError):
                load_splits(prepared)


if __name__ == "__main__":
    unittest.main()
