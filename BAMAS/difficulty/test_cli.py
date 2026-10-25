"""Default inference selects the current verified study, without legacy fallbacks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from BAMAS.scripts import try_difficulty


class DefaultCheckpointTests(unittest.TestCase):
    def test_selects_separate_main_and_rag_recommendations(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            for cohort, filename in (("main", "recommendation.json"), ("rag", "rag_recommendation.json")):
                relative = f"{cohort}/selected_model"
                (study / filename).write_text(json.dumps({"model_relative_path": relative}), encoding="utf-8")
                with patch.object(try_difficulty, "ALL_STUDY", study):
                    self.assertEqual(try_difficulty.default_model(cohort), (study / relative).resolve())

    def test_missing_recommendation_requires_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(try_difficulty, "ALL_STUDY", Path(temporary)):
                for cohort in ("main", "rag"):
                    with self.assertRaisesRegex(FileNotFoundError, "Train and finalize"):
                        try_difficulty.default_model(cohort)

    def test_recommendation_cannot_escape_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            (study / "recommendation.json").write_text(
                json.dumps({"model_relative_path": "../other/selected_model"}), encoding="utf-8")
            with patch.object(try_difficulty, "ALL_STUDY", study):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    try_difficulty.default_model()


if __name__ == "__main__":
    unittest.main()
