# IMDb Sentiment Neural Network with GitHub CI/CD

An educational **TF-IDF + PyTorch feedforward neural network** for positive/negative
English movie-review classification. This implements Option A of the course assignment.

## What is automated

```text
Push code to main/master
    -> install CPU dependencies
    -> run pipeline tests
    -> train on the course CSV
    -> save weights, vectorizer, configuration, metrics
    -> verify inference and retain downloadable artifacts
    -> create/update a Hugging Face model repository
```

No Codex/GitHub integration is required. GitHub Actions runs inside your own repository.
The upload script refuses local validation artifacts. Future model updates are made
by pushing code, not by manually editing the Hugging Face repository.

## Project structure

```text
data/imdb_top_500.csv       # exact course file provided by the student
data/README.md             # source and checksum
model/.gitkeep             # CI writes model outputs here
sentiment.py               # cleaning, validated CSV reader, MLP
train.py                   # training, validation-based selection, final test evaluation
predict.py                 # inference from saved artifacts
upload_to_hf.py            # GitHub Actions-only model deployment
requirements.txt          # pinned core dependencies
tests/test_pipeline.py     # leakage, split, and serialization checks
.github/workflows/train-and-upload.yml
SETUP_CN.md                # Chinese setup instructions without connecting Codex
```

## Dataset and model

- Course file: `imdb_top_500.csv`, with 500 reviews: 250 negative and 250 positive.
- Input: `text`. Target: `label` (0 = negative, 1 = positive).
- `rating` is deliberately ignored: a rating could leak information about sentiment.
- HTML tags are removed, HTML entities decoded, case normalized, and whitespace collapsed.
- Exact normalized duplicate reviews are removed before stratified splitting.
- Train / validation / test proportions: 70% / 15% / 15%, seed 42.
- TF-IDF: up to 5,000 unigram/bigram features, fitted **only on training text**.
- Network: `input -> Linear(64) -> ReLU -> Dropout(0.3) -> Linear(1)`.
- Loss: `BCEWithLogitsLoss`; sigmoid is applied at inference; threshold is fixed at 0.5.
- Adam optimizer; up to 40 epochs; early stopping after 5 unimproved validation losses.
- The checkpoint with lowest validation loss is evaluated on the test set once.

The usable split sizes and final metrics are written to `model/metrics.json` by each
actual training run. No fixed accuracy target is assumed or hard-coded. With just
75 held-out test examples, results are a small-sample estimate, not evidence of
general performance across domains. Scores are not calibrated confidence estimates.

## Set up deployment

1. Create a GitHub repository, for example `imdb-sentiment-nn`.
2. Put this project's **contents** at its root, including `.github/` and `data/`.
3. Add an HF write token as a GitHub Actions **Secret** named `HF_TOKEN`.
   Never put the token in a file, commit, screenshot, or chat.
4. Optional: add an Actions **Variable** named `HF_REPO_ID`, such as
   `your-hf-username/imdb-sentiment-nn`. If omitted, the script obtains the token
   owner's name and uses the GitHub repository name.
5. Push to `main`/`master`, or use Actions -> Train and Upload to Hugging Face -> Run workflow.
6. Open the run summary for test results, the artifact download, and the deployed HF URL.

The HF token must have write access to the target repository, and permission to
create it if it does not exist. For a narrowly scoped token, create the empty model
repository first and grant write access to it. The script then uploads automatically.
The target must be a **model repository**, not a Space or dataset repository.

If `HF_TOKEN` is missing, training and artifact retention still run, but the upload
step fails with a setup message. Add the Secret and rerun the workflow. A training
artifact alone is not a completed deployment.

## Local development / validation

```bash
python -m venv .venv
# Activate the environment using your operating system's normal command.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py
python predict.py "An excellent movie with wonderful acting." "A dull and disappointing story."
```

Python 3.11 is used in CI; the core dependencies also support Python 3.12. Local
training is for code validation only. Do not manually deploy its weights.

## Saved model artifacts

| Artifact | Contents |
|---|---|
| `model.pt` | PyTorch state dictionary |
| `vectorizer.pkl` | Training-fitted scikit-learn TF-IDF vectorizer |
| `config.json` | Dimensions, labels, preprocessing provenance, versions, commit |
| `metrics.json` | Accuracy, macro F1, majority baseline, confusion matrix, training history |
| `split_hashes.json` | Normalized-text hashes for auditing split overlap |
| `README.md` | Generated HF model card with this run's actual results |
| Inference source + requirements | Code needed to reload the model from the HF repository |

The model repository hosts files, not a live inference API. Download its files,
install the requirements, and use `python predict.py "review" --model-dir .` there.
Only load `vectorizer.pkl` from a source you trust, because it is a Python pickle.

## Final submission

After the entire Actions run succeeds, send these two URLs to the course WeChat group:

- Your GitHub repository URL.
- The Hugging Face model repository URL shown in the successful run summary.

## Sources

- [Course assignment](https://gitee.com/lundechen/data_engineering_master_2026_fall/blob/master/session-3/practice-huggingface-with-github-ci-cd.md)
- [PyTorch BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Hugging Face Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload)

No independent redistribution license is asserted for the course CSV. See its provenance
in `data/README.md`. Training code and documentation were prepared with AI assistance.
