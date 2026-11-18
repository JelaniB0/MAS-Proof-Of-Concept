"""Input-support calibration, artifact integrity and feature propagation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from .data import prepare
from .inputs import controller_features
from .model import DifficultyPredictor
from .research import run_suite
from .support import InputSupport, attach_support, unit_vectors
from .test_research import local_encoder


class SupportTests(unittest.TestCase):
    def test_distances_bounds_and_invalid_features(self):
        guard=InputSupport(np.array([[1.,0.],[0.,1.]]),.2)
        assessments=guard.assess(np.array([[2.,0.],[-1.,-1.]]))
        self.assertTrue(assessments[0]["within_support"])
        self.assertFalse(assessments[1]["within_support"])
        self.assertEqual(assessments[0]["distance"],0)
        for values in [np.zeros((1,2)), np.array([[np.nan,1]]), np.empty((0,2))]:
            with self.assertRaises(ValueError):
                unit_vectors(values)
        with self.assertRaises(ValueError):
            InputSupport([[1,0]],float('nan'))

    def test_controller_preserves_raw_scores_and_support_assessment(self):
        class Predictor:
            manifest={"groups":{"weak":["a"],"intermediate":["b"],"strong":["c"]}}
            metadata={"input_support":{"version":"test"}}
            def predict_with_support(self,texts):
                return [{"prediction":{"weak":.6,"intermediate":.5,"strong":.4},"input_support":{"within_support":False}} for text in texts]
        result=controller_features(Predictor(),["question"])[0]
        self.assertFalse(result["input_support"]["within_support"])
        self.assertEqual(result["difficulty_vector"],[.6,.5,.4])

    def test_frozen_support_export_reload_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            local_encoder(root/"encoder")
            frame=pd.DataFrame([{"sample_id":str(i),"eval_name":"math","prompt":f"question add {i}",
                                 "a":i%2,"b":.5,"c":1} for i in range(80)])
            frame.to_json(root/"source.jsonl",orient="records",lines=True)
            prepare(root/"source.jsonl",root/"data")
            run_suite(root/"data",root/"encoder",root/"run",device="cpu",seeds=(42,),epochs=1,patience=1,
                      configurations=[{"name":"one","input_mode":"question","balanced":False,"loss":"mse"}],compare_legacy=False)
            directory=root/"run/selected_model"
            before=DifficultyPredictor.load(directory).predict(["question add 1"])
            support=attach_support(directory,root/"data",device="cpu")
            after=DifficultyPredictor.load(directory)
            self.assertEqual(before,after.predict(["question add 1"]))
            self.assertIn("input_support",controller_features(after,["question add 1"])[0])
            self.assertEqual(attach_support(directory,root/"data",device="cpu"),support)
            path=directory/support["file"]
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"checksum"):
                DifficultyPredictor.load(directory)


if __name__=="__main__":
    unittest.main()
