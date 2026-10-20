from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.cli import main
from agentbalance.providers import Completion

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ["--config", str(ROOT / "agentbalance/demo.json"), "--policy", "balanced", "--topology", "star"]


class InteractiveCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = []
        self.stdout, self.stderr = StringIO(), StringIO()

        def report_path(command):
            path = self.directory / f"{command}-{len(self.paths)}.json"
            self.paths.append(path)
            return path

        for context in (
            patch.dict(os.environ, {
                "GROQ_API_KEY": "test-key", "GROQ_CHEAP_MODEL": "test-cheap", "GROQ_STRONG_MODEL": "test-strong",
                "AZURE_API_KEY": "test-key", "AZURE_CHAT_ENDPOINT": "https://example.test/chat/completions",
                "AZURE_CHEAP_DEPLOYMENT": "test-cheap", "AZURE_STRONG_DEPLOYMENT": "test-strong",
            }, clear=True),
            patch("agentbalance.cli.Path.cwd", return_value=self.directory),
            patch("agentbalance.cli.report_path", side_effect=report_path),
            patch("agentbalance.cli.load_env"),
            patch("agentbalance.providers.complete", return_value=Completion("Test answer", 10, 10, "provider", "stop")),
            redirect_stdout(self.stdout), redirect_stderr(self.stderr),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def reports(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.paths if path.exists()]

    def test_explicit_config_runs_repeated_independent_queries(self):
        with patch("builtins.input", side_effect=["First query", "Second query", "/exit"]):
            self.assertEqual(main(OFFLINE), 0)
        reports = self.reports()
        self.assertEqual([r["task"] for r in reports], ["First query", "Second query"])
        self.assertTrue(all(r["status"] == "completed" for r in reports))
        self.assertEqual(self.stdout.getvalue().count("\nTest answer\n"), 2)
        self.assertIn("mock models", self.stdout.getvalue())
        self.assertNotIn("First query", json.dumps(reports[1]))

    def test_file_command_preserves_multiline_bom_and_rereads_changes(self):
        path = self.directory / "my tasks.txt"
        path.write_text("Question one?\nQuestion two?", encoding="utf-8-sig")

        def inputs():
            yield f'/file "{path}"'
            path.write_text("Updated question", encoding="utf-8")
            yield f"/file {path}"
            yield "/quit"

        with patch("builtins.input", side_effect=inputs()):
            self.assertEqual(main(["chat", "--no-env", *OFFLINE]), 0)
        self.assertEqual([r["task"] for r in self.reports()],
                         ["Question one?\nQuestion two?", "Updated question"])

    def test_bad_files_blank_input_and_help_do_not_execute(self):
        empty = self.directory / "empty.txt"
        empty.write_text(" \n", encoding="utf-8")
        binary = self.directory / "binary.dat"
        binary.write_bytes(b"\xff\xff")
        lines = ["", "/help", "/file", f"/file {empty}", f"/file {binary}",
                 f"/file {self.directory / 'missing.txt'}", "Valid query", "/exit"]
        with patch("builtins.input", side_effect=lines):
            self.assertEqual(main(["chat", *OFFLINE]), 0)
        self.assertEqual([r["task"] for r in self.reports()], ["Valid query"])
        self.assertIn("Task file is empty", self.stderr.getvalue())

    def test_chat_accepts_initial_file_and_commandless_options(self):
        path = self.directory / "task.txt"
        path.write_text("Initial task", encoding="utf-8")
        with patch("builtins.input", side_effect=EOFError):
            self.assertEqual(main(["--task-file", str(path), *OFFLINE, "--topology", "linear"]), 0)
        self.assertEqual(self.reports()[0]["task"], "Initial task")
        self.assertEqual(self.reports()[0]["plan"]["topology"], "linear")

    def test_run_prompts_once_and_writes_requested_report(self):
        path = self.directory / "run.json"
        with patch("builtins.input", return_value="One task") as prompt:
            self.assertEqual(main(["run", "--out", str(path), *OFFLINE]), 0)
        prompt.assert_called_once()
        self.assertEqual(json.loads(path.read_text())["task"], "One task")

    def test_eof_and_interrupt_do_not_execute(self):
        for command in ([], ["run"]):
            with patch("builtins.input", side_effect=EOFError):
                self.assertEqual(main(command), 0)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(main(["chat"]), 130)
        self.assertEqual(self.paths, [])

    def test_failed_query_allows_next_query(self):
        with patch("builtins.input", side_effect=["First", "Second", "/exit"]), \
                patch("agentbalance.cli.make_plan", side_effect=ValueError("No plan fits")):
            self.assertEqual(main(["chat", *OFFLINE]), 0)
        self.assertEqual(self.stderr.getvalue().count("No plan fits"), 2)

    def test_provider_uses_matching_local_config_but_explicit_config_wins(self):
        demo = ROOT / "agentbalance" / "demo.json"
        local = self.directory / "config.local.json"
        local.write_text((ROOT / "configs/groq.example.json").read_text(encoding="utf-8"), encoding="utf-8")
        from agentbalance.config import load_config
        for arguments, expected in ((["--provider", "groq", "--policy", "cheap", "--topology", "star"], local), (OFFLINE, demo)):
            with self.subTest(arguments=arguments), patch("builtins.input", side_effect=EOFError), \
                    patch("agentbalance.cli.load_config", wraps=load_config) as loader:
                self.assertEqual(main(arguments), 0)
                loader.assert_called_with(expected)

    def test_invalid_local_config_does_not_fall_back_to_demo(self):
        (self.directory / "config.local.json").write_text("{}", encoding="utf-8")
        with patch("builtins.input") as prompt:
            self.assertEqual(main(["--provider", "groq", "--policy", "balanced", "--topology", "star"]), 2)
        prompt.assert_not_called()

    def write_provider_config(self, provider):
        data = json.loads((ROOT / "configs" / f"{provider}.example.json").read_text(encoding="utf-8"))
        for model in data["models"]:
            model.update(input_usd_per_million=0, output_usd_per_million=0)
        path = self.directory / "configs" / f"{provider}.local.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_startup_routes_both_providers_and_all_policies(self):
        for provider in ("groq", "azure"):
            self.write_provider_config(provider)
            for policy in ("cheap", "strong", "balanced"):
                with self.subTest(provider=provider, policy=policy), \
                        patch("builtins.input", side_effect=[provider, policy, "star", "My query", "/exit"]) as prompt:
                    self.assertEqual(main([]), 0)
                    self.assertEqual([call.args[0].split(">")[0] for call in prompt.call_args_list],
                                     ["Provider", "Policy [balanced]", "Topology [star]", "You", "You"])
                report = self.reports()[-1]
                self.assertEqual(report["plan"]["policy"], policy)
                self.assertEqual(report["plan"]["topology"], "star")
                self.assertTrue(all(agent["provider"] == provider for agent in report["agents"]))
                assignments = report["plan"]["assignments"]
                if policy in {"cheap", "strong"}:
                    self.assertEqual(set(assignments.values()), {policy})

    def test_invalid_choices_reprompt_and_accept_numbers_or_default(self):
        with patch("builtins.input", side_effect=["wrong", "1", "wrong", "", "wrong", "", "Question", "/exit"]):
            self.assertEqual(main([]), 0)
        self.assertEqual(self.reports()[-1]["plan"]["policy"], "balanced")
        self.assertEqual(self.reports()[-1]["plan"]["topology"], "star")
        self.assertIn("Choose groq, azure", self.stdout.getvalue())
        self.assertIn("Choose star, linear", self.stdout.getvalue())
        with patch("builtins.input", side_effect=["GROQ", "2", "2", "Question", "/exit"]):
            self.assertEqual(main([]), 0)
        self.assertEqual(self.reports()[-1]["plan"]["policy"], "strong")
        self.assertEqual(self.reports()[-1]["plan"]["topology"], "linear")
        self.assertEqual([agent["parents"] for agent in self.reports()[-1]["agents"]],
                         [[], ["analyst"], ["critic"]])

    def test_quit_at_any_menu_makes_no_calls(self):
        for answers in (["/exit"], ["groq", "/quit"], ["groq", EOFError],
                        ["groq", "balanced", "/exit"], ["groq", "balanced", EOFError]):
            with self.subTest(answers=answers), patch("builtins.input", side_effect=answers):
                self.assertEqual(main([]), 0)
        self.assertEqual(self.paths, [])

    def test_run_without_task_prompts_for_provider_policy_topology_then_query(self):
        with patch("builtins.input", side_effect=["groq", "cheap", "LINEAR", "One question"]):
            self.assertEqual(main(["run"]), 0)
        self.assertEqual(self.reports()[-1]["plan"]["policy"], "cheap")
        self.assertEqual(self.reports()[-1]["plan"]["topology"], "linear")

    def test_azure_does_not_reuse_groq_config(self):
        local = self.directory / "config.local.json"
        local.write_text((ROOT / "configs/groq.example.json").read_text(encoding="utf-8"), encoding="utf-8")
        azure = self.write_provider_config("azure")
        with patch("builtins.input", side_effect=["azure", "strong", "linear", "Question", "/exit"]):
            self.assertEqual(main([]), 0)
        self.assertIn(str(azure), self.stdout.getvalue())
        self.assertTrue(all(agent["provider"] == "azure" for agent in self.reports()[-1]["agents"]))

    def test_azure_template_requires_prices_before_execution(self):
        with patch("builtins.input", side_effect=["azure", "balanced", "star"]):
            self.assertEqual(main([]), 2)
        self.assertIn("configs/azure.local.json", self.stderr.getvalue())
        self.assertEqual(self.paths, [])


if __name__ == "__main__":
    unittest.main()
