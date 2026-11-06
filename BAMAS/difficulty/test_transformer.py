"""Offline MPNet regression tests, using a tiny random encoder, not downloaded weights."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import MPNetConfig, MPNetModel, PreTrainedTokenizerFast

from .data import GROUPS, prepare
from .model import DifficultyPredictor
from .transformer import (MPNetDifficultyNetwork, combined_loss, make_batch, optimizer_for,
                          sample_indices, tokenize_rows, train_transformer)


def tiny_tokenizer():
    vocabulary = {word: i for i, word in enumerate(
        ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "question", "add", "divide", "tail", "prefix"])}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A </s>",
                                                           special_tokens=[("<s>", 0), ("</s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                   cls_token="<s>", sep_token="</s>", unk_token="<unk>",
                                   pad_token="<pad>", mask_token="<mask>")


def tiny_encoder():
    return MPNetModel(MPNetConfig(vocab_size=10, hidden_size=16, num_hidden_layers=6,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64))


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_question_only_normalized_mse_checkpoint_reload(self):
        from .inputs import GSM_DEMO_HEADER
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoder = root / "encoder"
            tiny_encoder().save_pretrained(encoder, safe_serialization=True)
            tiny_tokenizer().save_pretrained(encoder)
            source = root / "source.jsonl"
            frame = pd.DataFrame([{"sample_id":str(i),"eval_name":"math",
                                   "prompt":GSM_DEMO_HEADER+f"\nQuestion: prefix\nAnswer: 9\nquestion add {i}",
                                   "a":i%2,"b":.5,"c":1} for i in range(80)])
            frame.to_json(source,orient="records",lines=True)
            prepare(source,root / "data")
            result = train_transformer(root / "data", root / "model", encoder=str(encoder), epochs=1,
                                       hidden_dim=8,batch_size=16,unfreeze_last=1,max_length=32,device="cpu",
                                       local_files_only=True,threads=1,input_mode="question",loss_kind="mse",
                                       normalize_embeddings=True,dropout=.2)
            predictor=DifficultyPredictor.load(root / "model")
            self.assertTrue(predictor.network.normalize_embeddings)
            self.assertEqual(predictor.metadata["input_mode"],"question")
            self.assertEqual(result["loss"],"masked MSE")
            self.assertEqual(predictor.predict(["question add 1"]),predictor.predict([GSM_DEMO_HEADER+"\nQuestion: prefix\nAnswer: 9\nquestion add 1"]))
            from .support import attach_support
            from .inputs import controller_features
            attach_support(root/"model",root/"data",device="cpu")
            supported=DifficultyPredictor.load(root/"model")
            direct=supported.predict(["question add 1"])[0]
            assessed=controller_features(supported,["question add 1"])[0]
            np.testing.assert_allclose(list(direct.values()),list(assessed["difficulty"].values()),atol=1e-6)
            self.assertIn("input_support",assessed)
            self.assertTrue(supported.metadata["predictor_id"])

    def test_only_final_four_layers_and_head_update(self):
        torch.manual_seed(42)
        network = MPNetDifficultyNetwork(tiny_encoder(), hidden_dim=8, unfreeze_last=4)
        optimizer = optimizer_for(network)
        tokenizer = tiny_tokenizer()
        items, _ = tokenize_rows(tokenizer, [{"prompt": "question add"}], 32)
        prefix = network.encoder.encoder.layer[0].attention.attn.q.weight
        adapted = network.encoder.encoder.layer[-1].attention.attn.q.weight
        before_prefix, before_adapted = prefix.detach().clone(), adapted.detach().clone()
        network.train()
        self.assertFalse(network.encoder.embeddings.training)
        self.assertFalse(network.encoder.encoder.layer[0].training)
        self.assertTrue(network.encoder.encoder.layer[-1].training)
        logits = network(**make_batch(tokenizer, items, [0], torch.device("cpu")))
        combined_loss(logits, torch.tensor([[0.0, 0.5, 1.0]])).backward()
        self.assertIsNone(prefix.grad)
        self.assertIsNotNone(adapted.grad)
        self.assertGreater(float(adapted.grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(torch.equal(before_prefix, prefix))
        self.assertFalse(torch.equal(before_adapted, adapted))
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 5e-4])
        self.assertTrue(all(group["weight_decay"] == 1e-3 for group in optimizer.param_groups))

    def test_masked_combined_loss_matches_reference_and_is_stable(self):
        logits = torch.tensor([[100.0, 0.5, -100.0]], requires_grad=True)
        truth = torch.tensor([[0.0, float("nan"), 1.0]])
        loss = combined_loss(logits, truth)
        reference = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(logits[:, [0, 2]], truth[:, [0, 2]])
                           + torch.nn.functional.mse_loss(logits[:, [0, 2]].sigmoid(), truth[:, [0, 2]]))
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_truncation_preserves_tail_special_tokens_and_batch_masks(self):
        tokenizer = tiny_tokenizer()
        tokenizer.truncation_side = "left"
        rows = [{"prompt": "prefix " * 20 + "tail"}, {"prompt": "question"}]
        items, stats = tokenize_rows(tokenizer, rows, 8)
        self.assertEqual(stats["truncated"], 1)
        self.assertEqual(items[0]["input_ids"][0], tokenizer.bos_token_id)
        self.assertEqual(items[0]["input_ids"][-1], tokenizer.eos_token_id)
        self.assertIn(tokenizer.convert_tokens_to_ids("tail"), items[0]["input_ids"])
        batch = make_batch(tokenizer, items, [0, 1], torch.device("cpu"))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(int(batch["attention_mask"][1].sum()), 3)

    def test_frozen_encoder_and_invalid_layer_counts(self):
        network = MPNetDifficultyNetwork(tiny_encoder(), unfreeze_last=0)
        network.train()
        self.assertFalse(any(p.requires_grad for p in network.encoder.parameters()))
        self.assertFalse(any(layer.training for layer in network.encoder.encoder.layer))
        for count in (-1, 7):
            with self.assertRaises(ValueError):
                MPNetDifficultyNetwork(tiny_encoder(), unfreeze_last=count)

    def test_balancing_is_reproducible_and_changes_expected_domain_mix(self):
        rows = [{"eval_name": "math"}] * 900 + [{"eval_name": "code"}] * 100
        one = sample_indices(rows, torch.Generator().manual_seed(42), True)
        two = sample_indices(rows, torch.Generator().manual_seed(42), True)
        self.assertEqual(one, two)
        code_fraction = sum(rows[i]["eval_name"] == "code" for i in one) / len(one)
        self.assertTrue(0.4 < code_fraction < 0.6)

    def test_local_training_and_offline_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, prepared, output, backbone = root / "source.jsonl", root / "data", root / "run", root / "encoder"
            tiny_encoder().save_pretrained(backbone, safe_serialization=True)
            tiny_tokenizer().save_pretrained(backbone)
            frame = pd.DataFrame([{"sample_id": str(i), "eval_name": "toy",
                                   "prompt": f"question {'add' if i % 2 else 'divide'} {i}",
                                   "a": i % 2, "b": 0.5, "c": 1.0} for i in range(90)])
            frame.to_json(source, orient="records", lines=True)
            prepare(source, prepared)
            report = train_transformer(prepared, output, epochs=1, patience=1, encoder=str(backbone),
                                       hidden_dim=8, batch_size=4, max_length=16, threads=1,
                                       accumulation_steps=3, local_files_only=True,
                                       device="cpu",
                                       max_train_samples=13, max_eval_samples=5)
            self.assertTrue(report["smoke_test"])
            self.assertEqual(report["training"]["counts"]["train"], 13)
            self.assertTrue(np.isfinite(report["test"]["macro_mse"]))
            predictor = DifficultyPredictor.load(output)
            first = predictor.predict(["question add"])
            second = DifficultyPredictor.load(output).predict(["question add"])
            self.assertEqual(first, second)
            self.assertEqual(set(first[0]), set(GROUPS))
            self.assertTrue(all(0 <= value <= 1 for value in first[0].values()))
            self.assertTrue((output / "model.safetensors").is_file())
            # The checkpoint is self-contained; inference does not read backbone files.
            self.assertEqual(predictor.predict([]), [])
            config = json.loads((output / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(config["unfreeze_last"], 4)


if __name__ == "__main__":
    unittest.main()
