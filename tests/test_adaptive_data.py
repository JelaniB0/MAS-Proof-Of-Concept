from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from agentbalance.adaptive.data import import_scores, load_records, split_records, exact_match
from agentbalance.adaptive.pools import construct_pools, pareto_models
from agentbalance.config import load_config
from agentbalance.graph import dag_waves

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveDataTests(unittest.TestCase):
    def test_dag_validation_and_parallel_waves(self):
        nodes = ("a", "b", "c", "synthesizer")
        edges = (("a", "c"), ("b", "c"), ("c", "synthesizer"))
        self.assertEqual(dag_waves(nodes, edges), [("a", "b"), ("c",), ("synthesizer",)])
        for bad in (edges + (("c", "a"),), edges[:-1], edges + (("unknown", "c"),), edges + (edges[0],)):
            with self.assertRaises(ValueError):
                dag_waves(nodes, bad)

    def test_pareto_preserves_tradeoffs_and_filters_dominated(self):
        m = load_config(ROOT / "agentbalance/demo.json").models[0]
        models = (replace(m, id="fast", quality=.4, latency_seconds=1),
                  replace(m, id="slow", quality=.4, latency_seconds=2),
                  replace(m, id="strong", quality=.9, latency_seconds=3))
        self.assertEqual([m.id for m in pareto_models(models)], ["fast", "strong"])
        pools, filtered = construct_pools(models, 2)
        self.assertEqual(pools, [("fast",), ("strong",)])
        self.assertEqual(filtered, ["slow"])

    def test_equal_profiles_and_zero_prices_have_nonempty_clusters(self):
        m = load_config(ROOT / "agentbalance/demo.json").models[0]
        models = [replace(m, id=f"m{i}", latency_seconds=0) for i in range(4)]
        first = construct_pools(models, 3)
        self.assertEqual(first, construct_pools(models, 3))
        self.assertTrue(all(first[0]))
        self.assertEqual(sum(map(len, first[0])), 4)

    def test_csv_import_groups_models_and_inverts_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "scores.csv", Path(directory) / "difficulty.jsonl"
            source.write_text("prompt,model,score\nHard,A,0\nHard,B,0\nEasy,A,1\nEasy,B,1\n", encoding="utf-8")
            self.assertEqual(import_scores(source, target, "prompt", ["score"], "model"), 2)
            rows = load_records(target, "difficulty")
            self.assertEqual([r["difficulty"] for r in rows], [1, 0])
            source.write_text("prompt,A,B\nMixed,0,1\n", encoding="utf-8")
            import_scores(source, target, "prompt", ["A", "B"])
            self.assertEqual(load_records(target, "difficulty")[0]["difficulty"], .5)

    def test_duplicates_invalid_scores_and_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "scores.csv", Path(directory) / "data.jsonl"
            for rows in ("prompt,A\nX,nan\n", "prompt,A\nX,2\n", "prompt,A\nX,1\nX,0\n"):
                source.write_text(rows, encoding="utf-8")
                with self.assertRaises(ValueError):
                    import_scores(source, target, "prompt", ["A"])
            target.write_text(json.dumps({"task": "Question", "answers": ["a"]}) + "\n" +
                              json.dumps({"task": " question ", "answers": ["b"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(target)
        with self.assertRaises(ValueError):
            split_records([{"key": "same"}], [{"key": "same"}])

    def test_exact_match_does_not_reward_answer_substrings(self):
        self.assertEqual(exact_match("  PARIS \n", ["Paris"]), 1)
        self.assertEqual(exact_match("Paris or London", ["Paris"]), 0)
        self.assertEqual(exact_match(None, ["Paris"]), 0)


if __name__ == "__main__":
    unittest.main()
