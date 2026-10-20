from dataclasses import replace
import json
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agentbalance.cli import main, describe
from agentbalance.config import load_config
from agentbalance.graph import messages, planning_prompt, token_reserve
from agentbalance.planner import InfeasiblePlan, make_plan
from agentbalance.providers import Completion, ProviderError
from agentbalance.runner import execute

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "agentbalance/demo.json")

    def plan(self, **kwargs):
        return make_plan(self.config, "Find a solution", **kwargs)

    def test_zero_budget_selects_only_free_profiles(self):
        plan = self.plan(budget_usd=0)
        self.assertEqual(set(plan.assignments.values()), {"cheap"})
        with self.assertRaises(InfeasiblePlan):
            self.plan(budget_usd=0, policy="strong")

    def test_balanced_prioritizes_final_answer_and_obeys_cap(self):
        plan = self.plan()
        self.assertEqual(plan.assignments, {"analyst": "cheap", "critic": "cheap", "synthesizer": "strong"})
        self.assertEqual(set(self.plan(max_strong_agents=0).assignments.values()), {"cheap"})

    def test_easy_task_does_not_buy_unneeded_quality(self):
        self.assertEqual(set(self.plan(difficulty=0).assignments.values()), {"cheap"})

    def test_latency_uses_graph_critical_path(self):
        self.assertAlmostEqual(self.plan(policy="strong").estimated_latency_seconds, 0.16)
        self.assertAlmostEqual(self.plan(policy="strong", topology="linear").estimated_latency_seconds, 0.24)
        with self.assertRaises(InfeasiblePlan):
            self.plan(policy="strong", topology="linear", latency_budget_seconds=0.2)

    def test_nonfinite_and_negative_budgets_rejected(self):
        for budget in (-1, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.plan(budget_usd=budget)

    def test_star_is_concurrent_and_synthesizer_receives_both(self):
        barrier = threading.Barrier(2)

        def fake(model, prompt, limit, timeout):
            role = prompt[0]["content"]
            if not role.startswith("Produce"):
                barrier.wait(timeout=2)  # Would fail if leaves were accidentally serialized.
            else:
                self.assertIn("Contribution from analyst", prompt[1]["content"])
                self.assertIn("Contribution from critic", prompt[1]["content"])
            return Completion("work", 10, 10, "provider", "stop")

        with patch("agentbalance.providers.complete", side_effect=fake):
            result = execute(self.config, self.plan(), "task")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["totals"]["calls"], 3)
        self.assertEqual(result["agents"][0]["parents"], [])
        self.assertEqual(result["agents"][1]["parents"], [])

    def test_linear_only_passes_direct_predecessor(self):
        def fake(model, prompt, limit, timeout):
            return Completion("work", 10, 10, "provider", "stop")
        with patch("agentbalance.providers.complete", side_effect=fake):
            result = execute(replace(self.config, trace_level="full"), self.plan(topology="linear"), "task")
        traces = result["agents"]
        self.assertEqual([t["parents"] for t in traces], [[], ["analyst"], ["critic"]])
        self.assertNotIn("Contribution from analyst", traces[-1]["messages"][-1]["content"])
        for previous, current in zip(traces, traces[1:]):
            self.assertGreaterEqual(current["started_seconds"], previous["started_seconds"] + previous["latency_seconds"])

    def test_upstream_context_is_bounded_and_unicode_reserve_covers_it(self):
        context = {"analyst": "\U0001f680" * 10000, "critic": "x" * 10000}
        prompt = messages("task", "synthesizer", context, 20)
        reserve = planning_prompt("task", "synthesizer", "star", 20)
        self.assertLessEqual(token_reserve(prompt), token_reserve(reserve))
        self.assertEqual(prompt[1]["content"].count("\U0001f680"), 20)

    def test_compact_trace_keeps_accounting_and_final_answer_without_duplicate_prompts(self):
        with patch("agentbalance.providers.complete", return_value=Completion("42", 10, 10, "provider", "stop")):
            result = execute(self.config, self.plan(topology="linear"), "task")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["totals"]["calls"], 3)
        self.assertNotIn("instructions", result["plan"])
        for trace in result["agents"]:
            self.assertNotIn("messages", trace)
            self.assertNotIn("text", trace)
            self.assertEqual(trace["input_tokens"], 10)

    def test_failed_call_stops_downstream_and_preserves_unknown_usage(self):
        with patch("agentbalance.providers.complete", side_effect=ProviderError("network error")) as call:
            result = execute(self.config, self.plan(topology="linear", policy="strong"), "task")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["totals"]["cost_complete"])
        self.assertGreater(result["totals"]["unknown_usage_reserved_usd"], 0)

    def test_reported_cost_overrun_stops_next_wave(self):
        with patch("agentbalance.providers.complete", return_value=Completion("work", 1000000, 1000000, "provider", "stop")) as call:
            result = execute(self.config, self.plan(topology="linear", policy="strong"), "task")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["status"], "budget_exceeded")
        self.assertAlmostEqual(result["totals"]["cost_usd"], 4)

    def test_empty_output_retains_billed_usage(self):
        with patch("agentbalance.providers.complete", return_value=Completion("", 100, 200, "provider", "length")):
            result = execute(self.config, self.plan(topology="linear", policy="strong"), "task")
        self.assertEqual(result["status"], "error")
        self.assertAlmostEqual(result["totals"]["cost_usd"], 0.0007)
        self.assertIn("completion-token limit", result["error"])
        self.assertIn("configured cap: 256", result["error"])
        self.assertIn("strong", result["error"])

    def test_failed_response_status_preserves_usage_but_stops_graph(self):
        with patch("agentbalance.providers.complete", return_value=Completion("partial", 100, 200, "provider", "failed")) as call:
            result = execute(self.config, self.plan(topology="linear", policy="strong"), "task")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["status"], "error")
        self.assertAlmostEqual(result["totals"]["cost_usd"], 0.0007)

    def test_star_failure_retains_other_parallel_calls_usage(self):
        def fake(model, prompt, limit, timeout):
            if prompt[0]["content"].startswith("Analyze"):
                raise ProviderError("network error")
            return Completion("work", 100, 200, "provider", "stop")
        with patch("agentbalance.providers.complete", side_effect=fake) as call:
            result = execute(self.config, self.plan(policy="strong"), "task")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["status"], "error")
        self.assertAlmostEqual(result["totals"]["cost_usd"], 0.0007)
        self.assertFalse(result["totals"]["cost_complete"])

    def test_admission_reserves_entire_parallel_wave(self):
        # Simulate a stale/manually tightened plan to exercise runtime admission.
        plan = replace(self.plan(policy="strong"), budget_usd=0)
        with patch("agentbalance.providers.complete") as call:
            result = execute(self.config, plan, "task")
        call.assert_not_called()
        self.assertEqual(result["status"], "budget_stopped")

    def test_latency_overrun_is_reported(self):
        plan = replace(self.plan(topology="linear"), latency_budget_seconds=0.001)
        def slow(*args):
            time.sleep(0.005)
            return Completion("work", 10, 10, "provider", "stop")
        with patch("agentbalance.providers.complete", side_effect=slow) as call:
            result = execute(self.config, plan, "task")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["status"], "budget_stopped")

    def test_config_rejects_missing_prices_and_duplicate_ids(self):
        for change in (lambda d: d["models"][1].update(input_usd_per_million=None),
                       lambda d: d["models"][1].update(id="cheap"),
                       lambda d: d["models"][1].update(extra_body={"max_tokens": 10000})):
            data = json.loads((ROOT / "agentbalance/demo.json").read_text())
            change(data)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_comparison_cli_writes_six_runs_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            with patch("builtins.print"):
                code = main(["compare", "--no-env", "--task", "task", "--out", str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(data["runs"]), 6)
            self.assertTrue(output.with_suffix(".csv").is_file())

    def test_terminal_summary_includes_agent_provider_errors(self):
        with patch("agentbalance.providers.complete", side_effect=ProviderError("HTTP 403; client signature 1010")):
            report = execute(self.config, self.plan(), "task")
        with patch("builtins.print") as output:
            describe(report)
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("analyst: HTTP 403; client signature 1010", printed)
        self.assertIn("critic: HTTP 403; client signature 1010", printed)

    def test_cli_prints_unicode_answer_on_ascii_terminal(self):
        with patch("agentbalance.providers.complete", return_value=Completion("A non\u2011ASCII answer", 10, 10, "provider", "stop")):
            report = execute(self.config, self.plan(), "task")
        buffer = io.BytesIO()
        terminal = io.TextIOWrapper(buffer, encoding="ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode.json"
            with patch("sys.stdout", terminal), patch("agentbalance.cli.execute", return_value=report):
                code = main(["run", "--no-env", "--task", "task", "--out", str(path)])
            terminal.flush()
            self.assertEqual(code, 0)
            self.assertIn(b"A non\\u2011ASCII answer", buffer.getvalue())
            self.assertIn(b"Report:", buffer.getvalue())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["answer"], "A non\u2011ASCII answer")
        terminal.close()


if __name__ == "__main__":
    unittest.main()
