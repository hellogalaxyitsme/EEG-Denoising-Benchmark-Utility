#!/usr/bin/env python3
"""Inspect a controlled DSConv U-Net checkpoint without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_denoise_benchmark.checkpoints import checkpoint_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt/.pth checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(checkpoint_summary(args.checkpoint), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
