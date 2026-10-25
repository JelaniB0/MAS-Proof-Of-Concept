"""Adversarial parser checks and split/target invariants for correctness regrading."""

from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .gsm_correctness import (canonical_question, decode_response, extract_answer, numeric_value,
                              reference_value, regrade_rows, prepare_correctness)
from .data import prepare
from .inputs import contract_id


class CorrectnessTests(unittest.TestCase):
    def test_exact_numeric_reference_and_fraction_comparison(self):
        self.assertEqual(reference_value("Work\n#### 1,200"), 1200)
        self.assertEqual(numeric_value("0.75"), Fraction(3, 4))
        self.assertEqual(numeric_value("-1/2"), Fraction(-1, 2))
        for value in ["Work without a final marker", "#### 2 or 3"]:
            with self.assertRaises(ValueError):
                reference_value(value)

    def test_serialized_responses_and_ambiguous_trials(self):
        value = repr(["Work\\nAnswer: 72"])
        self.assertEqual(decode_response(value)[0], "Work\nAnswer: 72")
        self.assertEqual(extract_answer(value)["value"], "72")
        for value in [repr(["Answer 72", "Answer 73"]), "[invalid", None, "", "[]", "Error 404"]:
            self.assertEqual(extract_answer(value)["status"], "unknown")

    def test_final_numbers_markers_and_ambiguous_conclusions(self):
        examples = {"48 + 24 = 72.": "72", "The answer is $1,200.00.": "1200",
                    r"Therefore, $\boxed{\frac{3}{4}}$.": "3/4", "#### -2": "-2",
                    "Answer: 25%": "25", "Work 48 24\nTherefore 72 clips.": "72"}
        for response, expected in examples.items():
            with self.subTest(response=response):
                self.assertEqual(extract_answer(response)["value"], expected)
        for response in ["The answer is 72 or 73.", r"\boxed{72} but \boxed{73}", "No answer.",
                         "Maybe 48 in April and 24 in May.", "```python\nprint(72)\n```", "#### 1/0",
                         "First, find how many elephants left during the 4-hour exodus:",
                         "First, we know a steel bar weighs 20 kg.", "What is the total after 20 days?"]:
            self.assertEqual(extract_answer(response)["status"], "unknown")

    def test_regrading_does_not_select_answer_by_reference_or_fill_missing(self):
        groups = {"weak": ["a"], "intermediate": ["b"], "strong": ["c"]}
        rows = [{"sample_id": str(i), "split": split, "eval_name": "grade-school-math", "prompt": f"Question {i}?",
                 "scores": {m: .75 for m in "abc"}} for i, split in enumerate(["train", "validation", "test"])]
        references = [{"question": r["prompt"], "answer": "Work\n#### 72"} for r in rows]
        source = pd.DataFrame([{**r, "a|model_response": repr(["Answer 72"]), "b|model_response": repr(["Answer 73"]),
                                "c|model_response": repr(["Answer 72" if r["sample_id"] != "1" else "72 or 73"])} for r in rows])
        complete, observations, quarantine = regrade_rows(source, references, rows, groups)
        self.assertEqual([r["split"] for r in complete], ["train", "test"])
        self.assertEqual(complete[0]["difficulty"], {"weak": 0, "intermediate": 1, "strong": 0})
        self.assertEqual(quarantine[0]["sample_id"], "1")
        self.assertIsNone(next(r for r in observations if r["model"] == "c" and r["sample_id"] == "1")["correct"])
        self.assertEqual(rows[0]["scores"]["a"], .75)

    def test_question_identity_and_target_semantics_are_explicit(self):
        self.assertEqual(canonical_question("What\u2019s  1 + 1?"), "What's 1 + 1?")
        base = {"groups": {"weak": ["a"], "intermediate": ["b"], "strong": ["c"]}}
        self.assertNotEqual(contract_id(base), contract_id({**base, "target_definition": "binary numeric correctness"}))
        self.assertNotEqual(contract_id({**base, "target_definition": "a"}), contract_id({**base, "target_definition": "b"}))

    def test_preparation_handles_unicode_jsonl_and_refuses_changed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame([{"sample_id": str(i), "eval_name": "grade-school-math", "prompt": f"Question {i}?",
                                   **{m: .75 for m in "abc"}, **{m+"|model_response": repr(["Answer 72"]) for m in "abc"}}
                                  for i in range(120)])
            source = root / "original.jsonl"
            frame.to_json(source,orient="records",lines=True)
            prepare(source,root / "prepared")
            references = root / "references.jsonl"
            references.write_text("".join(json.dumps({"question":f"Question {i}?", "answer":"Work\u2028more work\n#### 72"},ensure_ascii=False)+"\n" for i in range(120)),encoding="utf-8")
            with patch("BAMAS.difficulty.safe_pickle.read_routerbench",return_value=frame):
                stats=prepare_correctness(source,references,root / "prepared",root / "audit")
            self.assertEqual(stats["questions_complete"],120)
            self.assertEqual(stats["questions_quarantined"],0)
            training=root / "audit/correctness/train.jsonl"
            before=training.read_bytes()
            with references.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"question":"extra", "answer":"#### 1"})+"\n")
            with self.assertRaisesRegex(ValueError,"protocol"):
                prepare_correctness(source,references,root / "prepared",root / "audit")
            self.assertEqual(before,training.read_bytes())


if __name__ == "__main__":
    unittest.main()
