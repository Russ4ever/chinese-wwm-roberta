"""Compatibility CLI for the canonical single-position loss-recovery stage.

The former script contained a separate 12-layer joint intervention path.  That
path was methodologically incomparable with the Layer-6 residual intervention,
so this entry point now delegates to the same audited implementation as Cell 10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activation_rank import load_activation_rank_config
from src.activation_rank_loss_recovery import run_loss_recovery_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "activation_rank.yaml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs" / "activation_rank_loss_recovery.yaml",
    )
    args = parser.parse_args()
    config = load_activation_rank_config(args.config)
    output = run_loss_recovery_stage(config, args.policy)
    summary = pd.read_parquet(output / "site_summary.parquet")
    print(summary.to_string(index=False))
    print(f"\nCanonical output: {output}")


if __name__ == "__main__":
    main()
