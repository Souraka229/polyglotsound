"""Create / update the Hugging Face Space (free CPU hardware). Requires `hf auth login` first.

python scripts/deploy_space.py [space_name]
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parents[1]
IGNORE = [".venv/*", ".git/*", "third_party/*", "models/nllb*", "models/*.bin", "datasets/projects/*",
          "datasets/registry.db", "datasets/synthetic/*", "benchmarks/results/svc/*", "benchmarks/results/_*",
          "**/__pycache__/*", "*.pyc", ".pytest_cache/*", "space/*", "README.md", "requirements.txt",
          "**/*.wav", "**/*.mp3", "**/*.db"]


def main(name: str = "polyglotsound") -> None:
    user = whoami()["name"]
    repo_id = f"{user}/{name}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="PolyglotSound : code")
    for f in ("space_app.py", "requirements.txt", "packages.txt", "README.md"):
        api.upload_file(path_or_fileobj=str(ROOT / "space" / f), path_in_repo=f, repo_id=repo_id,
                        repo_type="space", commit_message=f"Space : {f}")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main(*sys.argv[1:])
