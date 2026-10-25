"""Evaluation integrity: external overlap, reference isolation and pool contracts."""

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from .benchmark import external_rows, identity, reject_overlap, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_external_identity_is_question_text_not_just_dataset_id(self):
        rows=[{"sample_id":"external/1","prompt":"Old   question"},
              {"sample_id":"external/2","prompt":"New question"},
              {"sample_id":"external/3","prompt":"New  question"}]
        selected,excluded=reject_overlap(rows,{identity("Old question"):{"train"}})
        self.assertEqual([r["sample_id"] for r in selected],["external/2"])
        self.assertEqual(excluded[0]["development_splits"],["train"])
        self.assertTrue(excluded[1]["duplicate_within_external_set"])

    def test_reference_solutions_and_tests_are_not_predictor_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            math=root/"math/algebra/test.parquet"
            math.parent.mkdir(parents=True)
            pd.DataFrame([{"problem":f"Math question {i}","level":"Level 1","type":"Algebra",
                           "solution":"SENSITIVE_REFERENCE_ANSWER"} for i in range(5000)]).to_parquet(math)
            he=root/"HumanEval.jsonl.gz"
            payload="\n".join(json.dumps({"task_id":f"HumanEval/{i}","prompt":f"Code problem {i}",
                                          "canonical_solution":"SENSITIVE_REFERENCE_ANSWER",
                                          "test":"raise RuntimeError('MUST_NOT_EXECUTE')"}) for i in range(164))
            he.write_bytes(gzip.compress(payload.encode()))
            gsm=root/"gsm.jsonl"
            gsm.write_text("\n".join(json.dumps({"question":f"GSM problem {i}","answer":"SENSITIVE_REFERENCE_ANSWER"})
                                     for i in range(1319)),encoding="utf-8")
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            manifest={"math":{"files":[{"path":"math/algebra/test.parquet","sha256":digest(math)}]},
                      "humaneval":{"path":he.name,"sha256":digest(he)}}
            (root/"source_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            datasets,_=external_rows(root,gsm)
            for rows in datasets.values():
                for row in rows:
                    self.assertFalse({"solution","canonical_solution","test","answer"}&row.keys())
                    self.assertNotIn("SENSITIVE_REFERENCE_ANSWER",row["prompt"])
                    self.assertNotIn("MUST_NOT_EXECUTE",row["prompt"])
            he.write_bytes(he.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError,"changed since download"):
                external_rows(root,gsm)

    def test_wrong_model_pool_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prepared=root/"prepared/main"
            prepared.mkdir(parents=True)
            groups={"weak":["a"],"intermediate":["b"],"strong":["c"]}
            (prepared/"manifest.json").write_text(json.dumps({"groups":groups}),encoding="utf-8")
            (root/"recommendation.json").write_text(json.dumps({"model_relative_path":"main/selected_model","predictor_id":"fixed"}),encoding="utf-8")
            wrong=SimpleNamespace(manifest={"groups":{**groups,"strong":["another-model"]}},predict_with_support=Mock())
            splits={name:[{"prompt":name}] for name in ("train","validation","test")}
            with patch("BAMAS.difficulty.benchmark.load_splits",return_value=splits), \
                 patch("BAMAS.difficulty.benchmark.external_rows",return_value=({},{})), \
                 patch("BAMAS.difficulty.benchmark.DifficultyPredictor.load",return_value=wrong):
                with self.assertRaisesRegex(ValueError,"pool/target contract"):
                    run_benchmark(root,root,root,root/"evaluation",device="cpu")
            wrong.predict_with_support.assert_not_called()


if __name__ == "__main__":
    unittest.main()
