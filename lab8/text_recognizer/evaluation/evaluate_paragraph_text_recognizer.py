"""Run validation test for paragraph_text_recognizer module."""
import os
import argparse
import time
import unittest
import torch
import pytorch_lightning as pl
from text_recognizer.data import IAMParagraphs
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer


_TEST_CHARACTER_ERROR_RATE = 0.17


class TestEvaluateParagraphTextRecognizer(unittest.TestCase):
    """Evaluate ParagraphTextRecognizer on the IAMParagraphs test dataset."""

    @torch.no_grad()
    def test_evaluate(self):
        dataset = IAMParagraphs(argparse.Namespace(batch_size=16, num_workers=10))
        dataset.prepare_data()
        dataset.setup()

        text_recog = ParagraphTextRecognizer()
        trainer = pl.Trainer(gpus=1)

        start_time = time.time()
        metrics = trainer.test(text_recog.lit_model, datamodule=dataset)
        end_time = time.time()

        test_cer = round(metrics[0]["test_cer"], 2)
        time_taken = round((end_time - start_time) / 60, 2)

        print(f"Character error rate: {test_cer}, time_taken: {time_taken} m")
        self.assertEqual(test_cer, _TEST_CHARACTER_ERROR_RATE)
        self.assertLess(time_taken, 45)
