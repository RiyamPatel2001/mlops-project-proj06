#!/usr/bin/env python3
"""
Create models/tfidf_logreg_onnx/model.onnx without Jupyter.

Requires: pandas, scikit-learn, skl2onnx, onnx (same stack as eval notebook).

Usage (from repo root):
    python3 scripts/export_tfidf_onnx_surrogate.py
    python3 scripts/export_tfidf_onnx_surrogate.py --repo /path/to/repo
"""

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = os.path.dirname(_SCRIPTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TF-IDF surrogate ONNX for Triton")
    parser.add_argument(
        "--repo",
        default=_DEFAULT_REPO,
        help="Project root (directory containing data/ and models/)",
    )
    args = parser.parse_args()
    sys.path.insert(0, _SCRIPTS_DIR)
    from tfidf_onnx_helpers import write_surrogate_onnx

    path = write_surrogate_onnx(args.repo)
    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"Wrote {path} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
