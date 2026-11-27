"""Publish only models trained in GitHub Actions, never local validation artifacts."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    if os.getenv("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Deployment must run in GitHub Actions, as required by the assignment.")
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Add HF_TOKEN in GitHub Settings > Secrets and variables > Actions, then rerun the workflow.")
    root = Path("model")
    required = ["model.pt", "vectorizer.pkl", "config.json", "metrics.json", "README.md",
                "split_hashes.json", "sentiment.py", "predict.py", "requirements.txt"]
    for filename in required:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Missing model artifact: {filename}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("training_origin") != "github_actions" or config.get("github_commit") != os.getenv("GITHUB_SHA"):
        raise RuntimeError("Artifacts must be trained in this GitHub Actions commit.")
    api = HfApi(token=token)
    repo_id = os.getenv("HF_REPO_ID", "").strip()
    if not repo_id:
        username = api.whoami()["name"]
        repo_name = os.environ["GITHUB_REPOSITORY"].split("/")[-1]
        repo_id = f"{username}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(root),
        allow_patterns=required, commit_message=f"Model trained by GitHub Actions: {os.environ['GITHUB_SHA'][:12]}")
    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded model: {url}")
    print(f"Hugging Face commit: {commit.oid}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"\n## Deployment\n- Model repository: {url}\n- HF commit: `{commit.oid}`\n")


if __name__ == "__main__":
    main()
