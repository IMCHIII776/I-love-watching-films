"""Train a TF-IDF + PyTorch MLP and save reproducible inference artifacts."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from sentiment import SentimentMLP, read_reviews


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_indices(labels, seed):
    indices = np.arange(len(labels))
    train, held = train_test_split(indices, test_size=0.30, stratify=labels, random_state=seed)
    valid, test = train_test_split(held, test_size=0.5, stratify=np.asarray(labels)[held], random_state=seed)
    return train, valid, test


def probabilities(model, matrix, batch_size=64):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            batch = torch.from_numpy(matrix[start:start + batch_size].toarray())
            outputs.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(outputs)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_features < 10 or args.hidden_dim < 1:
        raise ValueError("Epochs, batch size, hidden dimension, and vocabulary size must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    texts, labels, data_info = read_reviews(args.data)
    y = np.asarray(labels, dtype=np.float32)
    tr, va, te = split_indices(labels, args.seed)
    # Fit vocabulary and IDF exclusively on the training split.
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True, dtype=np.float32)
    x_train = vectorizer.fit_transform([texts[i] for i in tr])
    x_valid = vectorizer.transform([texts[i] for i in va])
    x_test = vectorizer.transform([texts[i] for i in te])
    model = SentimentMLP(x_train.shape[1], args.hidden_dim, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(len(tr))
        total_loss = 0.0
        for start in range(0, len(tr), args.batch_size):
            ids = permutation[start:start + args.batch_size]
            batch = torch.from_numpy(x_train[ids].toarray())
            targets = torch.from_numpy(y[tr][ids])
            optimizer.zero_grad()
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
        model.eval()
        with torch.inference_mode():
            valid_logits = model(torch.from_numpy(x_valid.toarray()))
            valid_loss = criterion(valid_logits, torch.from_numpy(y[va])).item()
            valid_accuracy = accuracy_score(y[va], (torch.sigmoid(valid_logits).numpy() >= 0.5))
        record = {"epoch": epoch, "train_loss": total_loss / len(tr),
                  "validation_loss": valid_loss, "validation_accuracy": float(valid_accuracy)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if valid_loss < best_loss - 1e-5:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), valid_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    # Evaluate the untouched test split only after validation-based model selection.
    predicted = (probabilities(model, x_test) >= 0.5).astype(int)
    majority = int(np.bincount(y[tr].astype(int)).argmax())
    origin = "github_actions" if os.getenv("GITHUB_ACTIONS") == "true" else "local_validation"
    config = {
        "architecture": "TF-IDF -> Linear -> ReLU -> Dropout -> Linear -> sigmoid",
        "input_dim": int(x_train.shape[1]), "hidden_dim": args.hidden_dim,
        "dropout": args.dropout, "threshold": 0.5,
        "id2label": {"0": "negative", "1": "positive"}, "seed": args.seed,
        "vectorizer_fit_split": "train_only", "split_fractions": [0.70, 0.15, 0.15],
        "data": data_info, "training_origin": origin,
        "github_commit": os.getenv("GITHUB_SHA"), "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"torch": torch.__version__, "scikit_learn": sklearn.__version__, "numpy": np.__version__},
    }
    metrics = {
        "test_accuracy": float(accuracy_score(y[te], predicted)),
        "test_macro_f1": float(f1_score(y[te], predicted, average="macro", zero_division=0)),
        "majority_baseline_accuracy": float(accuracy_score(y[te], np.full(len(te), majority))),
        "confusion_matrix_labels": ["negative", "positive"],
        "confusion_matrix": confusion_matrix(y[te], predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y[te], predicted, labels=[0, 1],
            target_names=["negative", "positive"], output_dict=True, zero_division=0),
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
        "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
        "selection_criterion": "minimum validation BCE; fixed threshold 0.5",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_origin": origin,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    with (output / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    json_save(output / "config.json", config)
    json_save(output / "metrics.json", metrics)
    split_hashes = {name: [hashlib.sha256(texts[i].encode()).hexdigest() for i in ids]
                    for name, ids in [("train", tr), ("validation", va), ("test", te)]}
    json_save(output / "split_hashes.json", split_hashes)
    for filename in ["sentiment.py", "predict.py", "requirements.txt"]:
        shutil.copyfile(Path(__file__).parent / filename, output / filename)
    card = f'''---
language: en
tags:
- sentiment-analysis
- text-classification
- pytorch
- tfidf
library_name: pytorch
---
# IMDb Sentiment Classification: TF-IDF + Neural Network

An educational classifier trained on the user-provided course file `imdb_top_500.csv`.
Only `text` is used as input. `label` is the target; `rating` is excluded.

## Results from this training run
- Training origin: `{origin}`
- Test accuracy: {metrics['test_accuracy']:.4f}
- Test macro F1: {metrics['test_macro_f1']:.4f}
- Train / validation / test sizes: {len(tr)} / {len(va)} / {len(te)}
- Best epoch selected by validation loss: {best_epoch}
- Parameters: {metrics['parameter_count']:,}
- GitHub commit: {config['github_commit'] or 'not a GitHub Actions run'}

## Usage
Download this repository's files, install `requirements.txt`, then run:
```bash
python predict.py "The acting was excellent and I loved the story." --model-dir .
```
The artifact folder includes inference code. It is a Hub model repository, not a hosted API.
`vectorizer.pkl` is a Python pickle: load files only from a trusted source.

## Evaluation and limitations
The vocabulary is fit on training data only. Model selection uses validation loss;
the test set is used once after selection. Exact normalized duplicates are removed
before stratified splitting. The test set is small, and its score is not a broad
generalization guarantee. Sigmoid output is not a calibrated confidence estimate.
This model is intended for English movie reviews, not safety-critical decisions.

## Data provenance
Course file: `imdb_top_500.csv`; SHA-256: `{data_info['sha256']}`.
Source supplied by the student from the course repository; no new data license is asserted.
'''
    (output / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps({"saved_to": str(output), "test_accuracy": metrics["test_accuracy"],
                      "test_macro_f1": metrics["test_macro_f1"]}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/imdb_top_500.csv")
    parser.add_argument("--output", default="model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
