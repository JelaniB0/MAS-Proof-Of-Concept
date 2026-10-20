import json
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.adaptive.benchmarks import fetch_dataset, normalize, prepare
from agentbalance.adaptive.data import load_records
from agentbalance.adaptive.scoring import Scorer, boxed
from agentbalance.config import load_config
from agentbalance.planner import make_plan
from agentbalance.roles import AGENT_SETS, templates, with_agent_set

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_rosters_follow_task_cycles_and_preserve_final_role(self):
        self.assertEqual([len(templates(name)) for name in AGENT_SETS], [6, 10, 6, 6, 18])
        for name in AGENT_SETS:
            config = with_agent_set(load_config(ROOT / "agentbalance/demo.json"), name)
            self.assertEqual(config.min_workers, 2)
            self.assertEqual(list(config.agent_templates)[-1], "synthesizer")
            for topology in ("star", "linear"):
                plan = make_plan(config, "Question", topology, "cheap", latency_budget_seconds=100)
                self.assertEqual(len(plan.assignments), len(config.agent_templates))
                self.assertEqual(len(plan.edges), len(config.agent_templates) - 1)

    def test_native_mmlu_csv_import_and_disjoint_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "subject_dev.csv"
            source.write_text("Question 1?,a,b,c,d,A\nQuestion 2?,a,b,c,d,D\n", encoding="utf-8")
            first, second = Path(folder) / "train.jsonl", Path(folder) / "test.jsonl"
            prepare(source, "mmlu", first, limit=1)
            prepare(source, "mmlu", second, offset=1)
            a, b = load_records(first)[0], load_records(second)[0]
            self.assertNotEqual(a["key"], b["key"])
            self.assertEqual(a["answers"], ["A"])
            self.assertNotIn("answer: A", a["task"])
            self.assertTrue(first.with_suffix(".manifest.json").exists())

    def test_math_nested_box_and_numeric_equivalence_without_execution(self):
        self.assertEqual(boxed(r"Earlier \boxed{1}. Final \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        row = normalize({"problem": "Find half", "solution": r"Answer \boxed{\frac{1}{2}}"}, "math", "fixture")
        self.assertEqual(Scorer().score(r"Thus \boxed{0.5}", row), 1)
        self.assertEqual(Scorer().score("1 or 0.5", row), 0)
        self.assertNotIn(r"\frac{1}{2}", row["task"])

    def test_mmlu_scorer_does_not_reward_arbitrary_letter_mentions(self):
        row = {"scorer": "mmlu", "answers": ["B"]}
        self.assertEqual(Scorer().score("B\nExplanation", row), 1)
        self.assertEqual(Scorer().score("A or B", row), 0)
        self.assertEqual(Scorer().score("Because option B looks right", row), 0)

    def code_row(self):
        return normalize({"task_id": "fixture/1", "prompt": "def f(x):\n", "test": "def check(f):\n    assert f(1) == 2",
                          "entry_point": "f", "canonical_solution": "SECRET_REFERENCE"}, "humaneval", "fixture")

    def test_humaneval_requires_functional_grader_and_hides_tests(self):
        row = self.code_row()
        self.assertNotIn("assert", row["task"])
        self.assertNotIn("SECRET_REFERENCE", json.dumps(row))
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "functional tests"):
                Scorer().preflight([row])
            run.assert_not_called()

    def test_humaneval_container_constraints_and_timeout_cleanup(self):
        with patch("subprocess.run", side_effect=[subprocess.CompletedProcess([], 0, "AGENTBALANCE_TESTS_PASSED", ""),
                                                 subprocess.CompletedProcess([], 0)]) as run:
            self.assertEqual(Scorer("python:test").score("    return x + 1\n", self.code_row()), 1)
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[0:2], ["docker", "run"])
            self.assertIn("--read-only", command)
            self.assertIn("none", command)
            self.assertIn("never", command)
            self.assertNotIn("-v", command)
            self.assertNotIn("shell", run.call_args_list[0].kwargs)
            payload = json.loads(run.call_args_list[0].kwargs["input"])
            self.assertEqual(payload["program"], "def f(x):\n    return x + 1\n")
        with patch("subprocess.run", side_effect=[subprocess.TimeoutExpired("docker", 10), subprocess.CompletedProcess([], 0)]) as run:
            self.assertEqual(Scorer("python:test").score("    while True: pass", self.code_row()), 0)
            self.assertEqual(run.call_args_list[-1].args[0][:3], ["docker", "rm", "-f"])

    def test_invalid_native_labels_and_duplicate_prompts_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize({"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 99}, "mmlu", "fixture")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "data.jsonl"
            row = {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": "A"}
            source.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare(source, "mmlu", Path(folder) / "out.jsonl")

    def test_dataset_fetch_paginates_and_records_source(self):
        pages = [BytesIO(json.dumps({"rows": [{"row": {"id": i}, "truncated_cells": []} for i in range(100)],
                                     "num_rows_total": 101}).encode()),
                 BytesIO(json.dumps({"rows": [{"row": {"id": 100}, "truncated_cells": []}], "num_rows_total": 101}).encode())]
        with tempfile.TemporaryDirectory() as folder, patch("agentbalance.adaptive.benchmarks.urlopen", side_effect=pages) as fetch:
            output = Path(folder) / "raw.jsonl"
            self.assertEqual(fetch_dataset("owner/dataset", "all", "dev", output, 101), 101)
            self.assertEqual(len(output.read_text().splitlines()), 101)
            self.assertIn("offset=100", fetch.call_args_list[1].args[0])
            self.assertEqual(json.loads(output.with_suffix(".download.json").read_text())["split"], "dev")

    def test_dataset_fetch_refuses_truncation_before_writing(self):
        page = BytesIO(json.dumps({"rows": [{"row": {"prompt": "partial"}, "truncated_cells": ["prompt"]}]}).encode())
        with tempfile.TemporaryDirectory() as folder, patch("agentbalance.adaptive.benchmarks.urlopen", return_value=page):
            output = Path(folder) / "raw.jsonl"
            with self.assertRaisesRegex(ValueError, "truncated"):
                fetch_dataset("owner/dataset", "all", "test", output, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
