#!/usr/bin/env python3
"""
upload_model_to_hub.py
───────────────────────
One-time helper: pushes your locally-trained BERT weights to a private (or
public) Hugging Face Hub repository so Render can download them at build time.

Usage
-----
1. Install the HF hub client (already in requirements-dev.txt):
       pip install huggingface-hub

2. Log in once (this stores a token in ~/.cache/huggingface/):
       huggingface-cli login

3. Run this script from the project root:
       python scripts/upload_model_to_hub.py --repo YOUR_HF_USERNAME/exam-anxiety-bert

The script will create the repo if it doesn't exist and upload every file
inside  model/bert_anxiety_model/.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="HF Hub repo id, e.g. your-username/exam-anxiety-bert",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "model", "bert_anxiety_model"),
        help="Local path to the saved model directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create a private repository (default: True)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface-hub is not installed.\n"
            "Run:  pip install huggingface-hub"
        )

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(
            f"Model directory not found: {model_dir}\n"
            "Train the model first with:  python training/train_model.py"
        )

    api = HfApi()
    print(f"Creating / verifying repo: {args.repo}")
    api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)

    print(f"Uploading files from: {model_dir}")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        commit_message="Upload trained BERT anxiety model",
    )
    print(f"\n✅ Model uploaded to: https://huggingface.co/{args.repo}")
    print(
        "\nNext step — set this env var in your Render backend service:\n"
        f"   HF_MODEL_REPO = {args.repo}"
    )


if __name__ == "__main__":
    main()
