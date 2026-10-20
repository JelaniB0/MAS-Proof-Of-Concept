import csv
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.adaptive.pools import construct_pools
from agentbalance.adaptive.routerbench import convert_routerbench, difficulty_rows, prompt_text
from agentbalance.config import load_config
from agentbalance.cli import main
from agentbalance.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]


class ThreeModelTests(unittest.TestCase):
    def setUp(self):
        base = load_config(ROOT / "agentbalance/demo.json")
        models = tuple(replace(base.models[0], id=name, quality=q, latency_seconds=t)
                       for name, q, t in (("cheap", .4, 3), ("medium", .65, 6), ("strong", .9, 10)))
        self.config = replace(base, models=models, medium_model="medium", pool_count=2, pool_min_size=2)

    def test_medium_routes_every_role_and_requires_configured_model(self):
        plan = make_plan(self.config, "Question", policy="medium")
        self.assertEqual(set(plan.assignments.values()), {"medium"})
        with self.assertRaisesRegex(ValueError, "medium_model"):
            make_plan(replace(self.config, medium_model=None), "Question", policy="medium")

    def test_overlapping_pools_keep_matching_choices(self):
        pools, removed = construct_pools(self.config.models, count=2, min_size=2)
        self.assertEqual(pools, [("cheap", "medium"), ("medium", "strong")])
        self.assertEqual(removed, [])
        dominated = replace(self.config.models[0], id="dominated", quality=.1, latency_seconds=30)
        pools, removed = construct_pools((*self.config.models, dominated), count=2, min_size=2)
        self.assertEqual(removed, ["dominated"])
        self.assertNotIn("dominated", set().union(*map(set, pools)))
        with self.assertRaisesRegex(ValueError, "Minimum pool size"):
            construct_pools(self.config.models, count=1, min_size=4)

    def test_prices_resolve_from_environment_and_fail_on_unknown_rates(self):
        data = asdict(self.config)
        data["models"][0]["input_usd_per_million"] = "${TEST_INPUT_RATE}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch.dict(os.environ, {"TEST_INPUT_RATE": "1.25"}):
                self.assertEqual(load_config(path).models[0].input_usd_per_million, 1.25)
            for invalid in ("", "private-invalid-value", "nan", "inf", "-1"):
                with self.subTest(invalid=invalid), patch.dict(os.environ, {"TEST_INPUT_RATE": invalid}):
                    with self.assertRaises(ValueError) as error:
                        load_config(path)
                    self.assertNotIn("private-invalid-value", str(error.exception))


class RouterBenchTests(unittest.TestCase):
    def test_native_csv_uses_correctness_not_cost_and_groups_before_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "scores.csv", Path(directory) / "difficulty.jsonl"
            columns = ["prompt", "eval_name", "a", "b", "a|model_response", "b|model_response", "a|total_cost"]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows([
                    dict(zip(columns, ["Question", "gsm8k", 1, "", "secret answer", "response", 99])),
                    dict(zip(columns, ["Question", "gsm8k", 0, 0, "another answer", "response", 99])),
                    dict(zip(columns, ["Excluded", "MMLU:physics", 1, 1, "answer", "response", 99])),
                ])
            with redirect_stdout(StringIO()):
                status = main(["research", "convert-routerbench", "--input", str(source),
                               "--out", str(output), "--exclude-eval", "mmlu"])
            self.assertEqual(status, 0)
            report = json.loads(output.with_suffix(".source.json").read_text())
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["difficulty"], .75)
            self.assertEqual(rows[0]["model_scores"], {"a": .5, "b": 0})
            self.assertNotIn("secret answer", output.read_text())
            self.assertEqual(report["excluded_rows"], 1)
            self.assertEqual(len(report["sha256"]), 64)
            self.assertTrue(output.with_suffix(".source.json").exists())
            with self.assertRaisesRegex(ValueError, "exists"):
                convert_routerbench(source, output)

    def test_prompts_and_missing_outcomes(self):
        self.assertEqual(prompt_text("['first', 'second']"), "first\nsecond")
        self.assertEqual(prompt_text([{"role": "user", "content": "Hello"}]), "user: Hello")
        rows, _ = difficulty_rows([{"prompt": "Q", "a": float("nan"), "b": 1}], ["a", "b"])
        self.assertEqual(rows[0]["difficulty"], 0)
        for value in (float("inf"), -1, 1.1):
            with self.assertRaises(ValueError):
                difficulty_rows([{"prompt": "Q", "a": value}], ["a"])
        with self.assertRaisesRegex(ValueError, "no valid"):
            difficulty_rows([{"prompt": "Q", "a": None}], ["a"])
        with self.assertRaisesRegex(ValueError, "eval_name"):
            difficulty_rows([{"prompt": "Q", "a": 1}], ["a"], ["mmlu"])

    def test_pickle_requires_explicit_trust_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "trust-pickle"):
                convert_routerbench(Path(directory) / "unread.pkl", Path(directory) / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
