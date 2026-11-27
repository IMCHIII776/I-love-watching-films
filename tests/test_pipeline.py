"""Meaningful data-leakage checks plus training/inference serialization round trip."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from predict import predict
from sentiment import clean_text, read_reviews
from train import split_indices


class PipelineTests(unittest.TestCase):
    def write_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "label", "rating"])
            writer.writerows(rows)

    def test_cleaning_dedup_and_conflicting_labels(self):
        self.assertEqual(clean_text("<br />Not &amp; BAD!"), "not & bad!")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            rows = [(f"unique review number {i}", i % 2, 10 if i % 2 else 1) for i in range(40)]
            self.write_csv(path, rows + [rows[0], ("<br>", 0, 1)])
            texts, labels, info = read_reviews(path)
            self.assertEqual(len(texts), 40)
            self.assertEqual(info["dropped_duplicates"], 1)
            self.assertEqual(info["dropped_empty"], 1)
            self.assertEqual(info["input_columns"], ["text"])
            self.write_csv(path, rows + [(rows[0][0], 1, 1)])
            with self.assertRaisesRegex(ValueError, "conflicting"):
                read_reviews(path)

    def test_splits_are_disjoint_and_repeatable(self):
        labels = [0, 1] * 250
        a = split_indices(labels, 42)
        b = split_indices(labels, 42)
        self.assertEqual([len(x) for x in a], [350, 75, 75])
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(set(a[0]) & set(a[1]))
        self.assertFalse(set(a[0]) & set(a[2]))
        self.assertFalse(set(a[1]) & set(a[2]))

    def test_train_save_reload_and_vocabulary_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            data, model_dir = Path(temp) / "reviews.csv", Path(temp) / "model"
            # Synthetic examples exist only in this test; they are not assignment data.
            rows = [(f"{'excellent wonderful' if i % 2 else 'terrible boring'} movie uniqueword{i}", i % 2, i % 2)
                    for i in range(80)]
            self.write_csv(data, rows)
            result = subprocess.run([sys.executable, "train.py", "--data", str(data), "--output", str(model_dir),
                "--epochs", "2", "--hidden-dim", "8", "--max-features", "200"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            from predict import load_model
            _, vectorizer, config = load_model(model_dir)
            tr, va, te = split_indices([r[1] for r in rows], 42)
            for i in np.concatenate([va, te]):
                self.assertNotIn(f"uniqueword{i}", vectorizer.vocabulary_)
            predictions = predict(["excellent wonderful movie", "terrible boring movie"], model_dir)
            self.assertEqual(predictions, predict(["excellent wonderful movie", "terrible boring movie"], model_dir))
            self.assertTrue(all(0 <= p["positive_score"] <= 1 for p in predictions))
            self.assertEqual(config["training_origin"], "github_actions" if __import__('os').getenv('GITHUB_ACTIONS') == 'true' else "local_validation")
            hashes = json.loads((model_dir / "split_hashes.json").read_text())
            self.assertFalse(set(hashes["train"]) & set(hashes["test"]))
            with self.assertRaises(ValueError):
                predict(["<br />"], model_dir)
            unknown = predict(["qzxvnjqzxvnj"], model_dir)[0]
            self.assertFalse(unknown["has_known_terms"])


if __name__ == "__main__":
    unittest.main()
