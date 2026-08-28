"""Pocket sequence diversity: 1 - mean pairwise identity."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    group_by_complex,
    load_sample_rows,
    mean_std,
    pairwise_identity,
    resolve_repo_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute generated pocket sequence diversity.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = resolve_repo_path(args.sample_dir)
    rows = load_sample_rows(sample_dir)
    if not rows:
        raise SystemExit(f"No samples found in {sample_dir}")

    per_complex = []
    diversities: List[float] = []
    for cid, items in group_by_complex(rows).items():
        seqs = [item.get("gen_pocket_seq") or "" for item in items]
        identity = pairwise_identity(seqs)
        diversity = None if identity is None else 1.0 - identity
        if diversity is not None:
            diversities.append(diversity)
        per_complex.append(
            {
                "complex_id": cid,
                "n_sequences": len([s for s in seqs if s]),
                "mean_pairwise_identity": identity,
                "diversity": diversity,
            }
        )
    summary = {"diversity": mean_std(diversities), "n_complexes": len(per_complex)}
    payload = {"summary": summary, "per_complex": per_complex}
    out_json = args.out_json or os.path.join(sample_dir, "diversity.json")
    write_json(out_json, payload)
    import json

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
