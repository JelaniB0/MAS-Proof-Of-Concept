from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from agentbalance.adaptive.controller import Controller, load_controller, transfer_controller
from agentbalance.adaptive.data import task_key
from agentbalance.adaptive.encoder import TextEncoder
from agentbalance.adaptive.training import RunAllowance, evaluate, fit_difficulty, train_policy, policy_snapshot
from agentbalance.config import load_config
from agentbalance.graph import dag_waves, INSTRUCTIONS
from agentbalance.providers import Completion, ProviderError
from agentbalance.runner import execute
from agentbalance.cli import main

ROOT = Path(__file__).resolve().parents[2]


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        base = load_config(ROOT / "agentbalance/demo.json")
        model = base.models[0]
        models = tuple(replace(model, id=f"m{i}", quality=.2 + i*.2, latency_seconds=.01 + i*.01,
                               input_usd_per_million=.1*(i+1), output_usd_per_million=.2*(i+1))
                       for i in range(4))
        roles = {"analyst": INSTRUCTIONS["analyst"], "researcher": "Find supporting evidence.",
                 "critic": INSTRUCTIONS["critic"], "synthesizer": INSTRUCTIONS["synthesizer"]}
        self.config = replace(base, models=models, cheap_model="m0", strong_model="m3", agent_templates=roles)
        self.controller = Controller(self.config, TextEncoder("hash-test", 32), hidden=16, pool_count=2)

    def row(self, text, **kwargs):
        return {"task": text, "answers": ["42"], "key": task_key(text), **kwargs}

    def test_sampled_graphs_are_acyclic_connected_and_pool_consistent(self):
        shapes = set()
        for _ in range(30):
            plan, logp, entropy, penalty = self.controller.select("Solve a math problem", sample=True)
            dag_waves(plan.assignments, plan.edges)
            self.assertGreaterEqual(len(plan.assignments), 2)
            self.assertTrue(set(plan.assignments.values()) <= set(self.controller.pools[plan.selection["selected_pool"]]))
            self.assertTrue(torch.isfinite(logp))
            self.assertTrue(torch.isfinite(entropy))
            self.assertGreaterEqual(float(penalty.detach()), 0)
            shapes.add(tuple(plan.edges))
        self.assertGreater(len(shapes), 1)

    def test_forced_pruning_and_hop_limit_control_execution(self):
        with torch.no_grad():
            self.controller.gate.weight.zero_()
            self.controller.gate.bias.fill_(-50)
        plan = self.controller.plan("Question")
        self.assertEqual(len(plan.assignments), 2)
        with patch("agentbalance.providers.complete", return_value=Completion("42", 10, 10, "provider", "stop")) as call:
            report = execute(self.config, plan, "Question")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["agents"][-1]["parents"], [plan.selection["forced_worker"]])
        with torch.no_grad():
            self.controller.gate.bias.fill_(50)
            self.controller.edge_source.weight.zero_()
            self.controller.edge_source.bias.fill_(10)
            self.controller.edge_target.weight.zero_()
            self.controller.edge_target.bias.fill_(10)
            self.controller.hops.weight.zero_()
            self.controller.hops.bias.copy_(torch.tensor([50., -50., -50.]))
        plan = self.controller.plan("Question")
        self.assertEqual(len(dag_waves(plan.assignments, plan.edges)), 2)
        self.assertTrue(plan.selection["hop_pruned_edges"])

    def test_gradients_reach_matching_representations_gating_and_topology(self):
        with torch.no_grad():
            self.controller.gate.bias.fill_(2)
        for _ in range(100):
            plan, logp, entropy, penalty = self.controller.select("Review code", sample=True)
            if len(plan.assignments) >= 3 and len(self.controller.pools[plan.selection["selected_pool"]]) > 1:
                break
        (-logp - .01 * entropy + .01 * penalty).backward()
        for layer in (self.controller.role_match, self.controller.context_output, self.controller.gate,
                      self.controller.edge_source, self.controller.hops, self.controller.pool_score):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_fixed_topology_ablation_preserves_all_roles(self):
        for topology in ("star", "linear"):
            plan = self.controller.plan("Question", topology=topology)
            self.assertEqual(len(plan.assignments), 4)
            self.assertEqual(plan.selection["pruned_agents"], [])
            self.assertEqual(len(plan.edges), 3)

    def test_three_model_overlap_trains_difficulty_and_matching_and_roundtrips(self):
        config = replace(self.config, models=self.config.models[:3], strong_model="m2",
                         medium_model="m1", pool_count=2, pool_min_size=2)
        controller = Controller(config, TextEncoder("hash-test", 32), hidden=16)
        self.assertEqual(len(controller.pools), 2)
        self.assertTrue(all(len(pool) >= 2 for pool in controller.pools))
        _, logp, entropy, _ = controller.select("Solve the question", sample=True, topology="star")
        (-logp - .01 * entropy).backward()
        for module in (controller.difficulty, controller.role_match, controller.pool_score):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.pt"
            controller.save(path)
            loaded = load_controller(path, config, allow_untrained=True)
            self.assertEqual(loaded.pools, controller.pools)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_controller(path, replace(config, pool_min_size=1), allow_untrained=True)

    def test_checkpoint_guard_identity_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "controller.pt"
            self.controller.save(checkpoint)
            with self.assertRaisesRegex(ValueError, "untrained"):
                load_controller(checkpoint, self.config)
            loaded = load_controller(checkpoint, self.config, allow_untrained=True)
            self.assertEqual(loaded.plan("Question").to_dict(), self.controller.plan("Question").to_dict())
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_controller(checkpoint, replace(self.config, max_output_tokens=999), True)

    def test_supervised_fit_reduces_error_without_training_encoder(self):
        train = [self.row("easy addition", difficulty=0), self.row("hard theorem", difficulty=1)]
        validation = [self.row("addition easy", difficulty=0), self.row("theorem hard", difficulty=1)]
        result = fit_difficulty(self.controller, train, validation, epochs=50, learning_rate=.02)
        self.assertLess(result["epochs"][-1]["validation_mse"], result["epochs"][0]["validation_mse"])
        self.assertEqual(self.controller.training_info["difficulty_steps"], 50)
        self.assertFalse(self.controller.encoder.encode(["easy addition"]).requires_grad)

    def test_training_updates_weights_and_obeys_call_cap(self):
        self.controller = Controller(self.config, TextEncoder("hash-test", 32), hidden=16, pool_count=1)
        allowance = RunAllowance(self.config, max_calls=4, total_budget=1)
        before = self.controller.role_match.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch("agentbalance.providers.complete", return_value=Completion("42", 10, 10, "provider", "stop")) as call:
            path = Path(directory) / "trained.pt"
            result = train_policy(self.controller, [self.row("Question one"), self.row("Question two")],
                                  path, allowance, epochs=4, topology="star")
            self.assertEqual(result["status"], "limit_reached")
            self.assertEqual(call.call_count, 4)
            self.assertEqual(self.controller.training_info["policy_steps"], 1)
            self.assertFalse(torch.equal(before, self.controller.role_match.weight))
            self.assertTrue(path.with_suffix(".rollouts.jsonl").exists())
            self.assertEqual([s["policy_steps"] for s in result["policy_snapshots"]], [0, 1])
            self.assertEqual(result["runs"][0]["policy_steps_after"], 1)

    def test_policy_probes_preserve_rng_mode_and_make_no_provider_calls(self):
        self.controller.train()
        state = torch.get_rng_state().clone()
        with patch("agentbalance.providers.complete") as complete:
            first = policy_snapshot(self.controller, [self.row("Probe")], .02, 120, "star")
            second = policy_snapshot(self.controller, [self.row("Probe")], .02, 120, "star")
            complete.assert_not_called()
        self.assertTrue(self.controller.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(first, second)

    def test_provider_error_stops_training_without_update_or_retry(self):
        allowance = RunAllowance(self.config, max_calls=100, total_budget=1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch("agentbalance.providers.complete", side_effect=ProviderError("HTTP 429")) as call:
            result = train_policy(self.controller, [self.row("Question")], Path(directory) / "error.pt",
                                  allowance, epochs=10, topology="linear")
        self.assertEqual(result["status"], "provider_or_budget_failure")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.controller.training_info["policy_steps"], 0)

    def test_evaluation_rejects_training_leakage_before_calls(self):
        row = self.row("Question")
        self.controller.training_info["policy_train_keys"] = [row["key"]]
        with tempfile.TemporaryDirectory() as directory, patch("agentbalance.providers.complete") as call:
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate(self.controller, [row], Path(directory) / "eval.json", RunAllowance(self.config, 10, 1))
            call.assert_not_called()

    def test_infeasible_graph_and_exhausted_budget_make_no_calls(self):
        with self.assertRaises(ValueError):
            self.controller.plan("Question", latency_budget_seconds=.0001)
        plan = self.controller.plan("Question", topology="star")
        with patch("agentbalance.providers.complete") as call:
            self.assertIsNone(RunAllowance(self.config, 1, 1).run(plan, "Question"))
            call.assert_not_called()

    def test_cli_learned_plan_requires_explicit_fixture_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            folder = Path(directory)
            config = folder / "config.json"
            config.write_text(json.dumps(asdict(self.config)), encoding="utf-8")
            checkpoint = folder / "fixture.pt"
            self.controller.save(checkpoint)
            args = ["plan", "--no-env", "--config", str(config), "--policy", "learned", "--task", "Question",
                    "--checkpoint", str(checkpoint), "--out", str(folder / "plan.json")]
            with patch("sys.stderr", StringIO()), patch("agentbalance.providers.complete") as call:
                self.assertEqual(main(args), 2)
                self.assertEqual(main([*args, "--allow-untrained"]), 0)
                call.assert_not_called()
            report = json.loads((folder / "plan.json").read_text())
            self.assertEqual(report["topology"], "adaptive")
            self.assertEqual(report["policy"], "learned")

    def test_baseline_evaluation_uses_identical_exact_match_scoring(self):
        from agentbalance.adaptive.training import evaluate_baseline
        with tempfile.TemporaryDirectory() as directory, \
                patch("agentbalance.providers.complete", return_value=Completion("42", 10, 10, "provider", "stop")):
            result = evaluate_baseline(self.config, [self.row("Unseen question")], Path(directory) / "baseline.json",
                                       RunAllowance(self.config, 4, 1))
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["calls"], 4)
        self.assertEqual(result["status"], "completed")

    def test_transfer_rebuilds_profiles_and_preserves_weights_and_leakage(self):
        target_models = tuple(replace(m, id="azure_" + m.id, model="deployment_" + m.id,
                                      provider="azure", input_usd_per_million=2*m.input_usd_per_million)
                              for m in self.config.models)
        target_config = replace(self.config, models=target_models, cheap_model="azure_m0", strong_model="azure_m3")
        self.controller.training_info["policy_train_keys"] = ["previous-training-prompt"]
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / "source.pt"
            self.controller.save(source_path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_controller(source_path, target_config, True)
            target = transfer_controller(source_path, self.config, target_config, pool_count=2)
            self.assertEqual(target.training_info["target_policy_steps"], 0)
            self.assertEqual(target.training_info["policy_train_keys"], ["previous-training-prompt"])
            self.assertTrue(torch.equal(target.role_match.weight, self.controller.role_match.weight))
            plan = target.plan("Unseen question")
            self.assertTrue(all(mid.startswith("azure_") for mid in plan.assignments.values()))
            self.assertEqual(plan.selection["pools"], target.pools)
            target.save(Path(folder) / "target.pt")
            load_controller(Path(folder) / "target.pt", target_config, True)
            with self.assertRaisesRegex(ValueError, "ordered role"):
                transfer_controller(source_path, self.config, replace(target_config, agent_templates=INSTRUCTIONS))

    def test_minimum_two_workers_and_non_star_graph(self):
        config = replace(self.config, min_workers=2)
        controller = Controller(config, TextEncoder("hash-test", 32), hidden=16, pool_count=2)
        with torch.no_grad():
            controller.gate.weight.zero_()
            controller.gate.bias.fill_(-50)
        plan = controller.plan("Question")
        self.assertEqual(len(plan.assignments), 3)
        self.assertEqual(len(plan.selection["forced_workers"]), 2)
        with torch.no_grad():
            controller.gate.bias.fill_(50)
            controller.edge_source.weight.zero_()
            controller.edge_target.weight.zero_()
            controller.edge_source.bias.fill_(10)
            controller.edge_target.bias.fill_(10)
            controller.hops.weight.zero_()
            controller.hops.bias.copy_(torch.tensor([-50., -50., 50.]))
        plan = controller.plan("Question")
        self.assertGreater(len(plan.edges), len(plan.assignments)-1)
        self.assertTrue(any(target != "synthesizer" for _, target in plan.edges))
        self.assertAlmostEqual(plan.selection["joint_log_probability"],
                               sum(a["log_probability"] for a in plan.selection["actions"]), places=5)

    def test_minibatch_training_and_target_finetuning_counters(self):
        self.controller.training_info["target_policy_steps"] = 0
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()), \
                patch("agentbalance.providers.complete", return_value=Completion("42", 10, 10, "provider", "stop")):
            result = train_policy(self.controller, [self.row("Question one"), self.row("Question two"), self.row("Question three")],
                                  Path(folder) / "batch.pt", RunAllowance(self.config, 12, 1), topology="star",
                                  batch_size=2, reward_units="raw")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(self.controller.training_info["policy_steps"], 2)
        self.assertEqual(self.controller.training_info["target_policy_steps"], 2)

    def test_humaneval_missing_grader_fails_before_live_calls(self):
        row = self.row("Code question", scorer="humaneval")
        with tempfile.TemporaryDirectory() as folder, patch("agentbalance.providers.complete") as call:
            with self.assertRaisesRegex(ValueError, "functional tests"):
                train_policy(self.controller, [row], Path(folder) / "unused.pt", RunAllowance(self.config, 10, 1))
            call.assert_not_called()

    def test_model_environment_change_cannot_silently_reuse_checkpoint(self):
        model = replace(self.config.models[0], model="${TEST_MODEL_NAME}")
        config = replace(self.config, models=(model, *self.config.models[1:]))
        controller = Controller(config, TextEncoder("hash-test", 32), hidden=16, pool_count=2)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"TEST_MODEL_NAME": "original"}):
            path = Path(folder) / "bound.pt"
            controller.save(path)
            os.environ["TEST_MODEL_NAME"] = "replacement"
            with self.assertRaisesRegex(ValueError, "Resolved model names"):
                load_controller(path, config, True)

    def test_cli_prepare_configure_train_and_transfer_workflow(self):
        from agentbalance.research import main as research_main
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            root = Path(folder)
            config_path = root / "config.json"
            self.assertEqual(research_main(["configure", "--no-env", "--config", str(ROOT / "agentbalance/demo.json"),
                                           "--agent-set", "mmlu", "--out", str(config_path)]), 0)
            data = root / "native.jsonl"
            data.write_text(json.dumps({"question": "Fixture question", "choices": ["a", "b", "c", "d"], "answer": 0}) + "\n", encoding="utf-8")
            prepared = root / "data.jsonl"
            self.assertEqual(research_main(["prepare-benchmark", "--benchmark", "mmlu", "--source", str(data),
                                           "--out", str(prepared)]), 0)
            checkpoint = root / "source.pt"
            with patch("agentbalance.providers.complete", return_value=Completion("A", 10, 10, "provider", "stop")):
                self.assertEqual(research_main(["train", "--no-env", "--config", str(config_path), "--data", str(prepared),
                                               "--encoder", "hash-test", "--out", str(checkpoint), "--max-calls", "6",
                                               "--total-budget", "1", "--batch-size", "2"]), 0)
            self.assertEqual(research_main(["transfer", "--no-env", "--source-config", str(config_path),
                                           "--config", str(config_path), "--checkpoint", str(checkpoint),
                                           "--out", str(root / "target.pt")]), 0)


if __name__ == "__main__":
    unittest.main()
