from pathlib import Path
import csv
import json
import tempfile
import unittest

from agentbalance.files import write_json
from agentbalance.reporting import export_directory, export_evaluation, export_training


def graph(status="completed", calls=3):
    return {"status": status, "mode": "live", "answer": "42",
            "plan": {"topology": "star", "assignments": {"worker": "cheap", "synthesizer": "strong"},
                     "graph_edges": [["worker", "synthesizer"]],
                     "selection": {"model_probabilities": {"worker": {"cheap": .8, "strong": .2}}}},
            "totals": {"calls": calls, "input_tokens": 100, "output_tokens": 20,
                       "cost_usd": 0., "cost_complete": status == "completed", "latency_seconds": 4.},
            "settings": {"min_request_interval_seconds": 65.},
            "model_profiles": [{"id": "cheap", "input_usd_per_million": 0, "output_usd_per_million": 0}]}


class ReportingTests(unittest.TestCase):
    def test_partial_evaluation_is_labeled_and_probabilities_are_preserved(self):
        report = {"status": "provider_or_budget_failure", "accuracy": 0., "evaluated": 1,
                  "requested": 3, "calls": 1, "accounted_usd": 0.,
                  "runs": [{"key": "test", "score": 0, "query_seconds": 4., "report": graph("error", 1)}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = export_evaluation(output / "eval.json", output, report)
            self.assertFalse(result["complete"])
            self.assertFalse(result["valid_live_evaluation"])
            self.assertFalse(result["cost_usage_complete"])
            self.assertFalse(result["nonzero_configured_prices"])
            policy = json.loads((output / "policy.json").read_text())
            self.assertEqual(policy["questions"][0]["model_probabilities"]["worker"]["cheap"], .8)
            with (output / "questions.csv").open(newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["status"], "error")

    def test_resume_chain_keeps_failed_usage_and_does_not_invent_old_update_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "model.failed-001.training.json"
            current = root / "model.training.json"
            old = {"status": "provider_or_budget_failure", "calls": 4, "accounted_usd": .1,
                   "training": {"policy_steps": 1}, "runs": [{"key": "a", "accuracy": 1, "agents": 2}]}
            new = {"status": "completed", "calls": 3, "accounted_usd": .2,
                   "resumed_from": str(root / "model.failed-001.optimizer.pt"),
                   "training": {"policy_steps": 2, "policy_train_keys": ["a", "b"]},
                   "runs": [{"key": "b", "accuracy": 1, "agents": 2}]}
            write_json(previous, old)
            write_json(current, new)
            (root / "model.failed-001.rollouts.jsonl").write_text(
                json.dumps({"key": "a", "report": graph()}) + "\n" +
                json.dumps({"key": "b", "report": graph("error", 1)}) + "\n")
            (root / "model.rollouts.jsonl").write_text(json.dumps({"key": "b", "report": graph()}) + "\n")
            output = root / "export"
            summary = export_training(current, output)
            self.assertEqual(summary["calls_all_attempts"], 7)
            self.assertEqual(summary["successful_rollouts"], 2)
            self.assertEqual(summary["failed_graphs"], 1)
            history = json.loads((output / "policy-history.json").read_text())
            self.assertIsNone(history["rollouts"][0]["policy_steps_before"])
            self.assertEqual(history["rollouts"][1]["calls"], 3)
            self.assertEqual(history["fixed_probes"], [])

    def test_directory_does_not_treat_configs_or_index_as_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "config.json", {"models": []})
            write_json(root / "index.json", {"evaluations": []})
            write_json(root / "eval.json", {"status": "completed", "evaluated": 1, "requested": 1,
                "accuracy": 1., "calls": 3, "runs": [{"key": "x", "score": 1., "report": graph()}]})
            rows = export_directory(root, root / "export")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "evaluation")
            self.assertTrue((root / "export" / "README.md").exists())

    def test_missing_resume_report_cannot_silently_undercount_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.training.json"
            write_json(source, {"resumed_from": str(root / "missing.optimizer.pt"), "runs": []})
            with self.assertRaisesRegex(ValueError, "Missing resumed training report"):
                export_training(source, root / "export")


if __name__ == "__main__":
    unittest.main()
