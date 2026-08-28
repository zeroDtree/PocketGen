"""PoseBusters ligand validity checks on PocketGen-updated ligands."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    load_sample_rows,
    mean_std,
    resolve_repo_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PoseBusters on generated ligands.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--config", type=str, default="mol", choices=("mol", "dock"))
    return parser.parse_args()


def row_passes(series) -> Dict[str, Any]:
    import numpy as np

    values = {}
    all_ok = True
    for key, value in series.items():
        if key in {"file", "molecule", "mol_pred", "mol_true", "mol_cond"}:
            continue
        if isinstance(value, (bool, np.bool_)):
            values[str(key)] = bool(value)
            all_ok = all_ok and bool(value)
    values["passes_all"] = all_ok
    return values


def main() -> None:
    args = parse_args()
    try:
        from posebusters import PoseBusters
    except ImportError as exc:
        raise SystemExit(
            "posebusters is not installed. Install it in the pocketgen env with: pip install posebusters"
        ) from exc

    sample_dir = resolve_repo_path(args.sample_dir)
    rows = load_sample_rows(sample_dir)
    if not rows:
        raise SystemExit(f"No samples found in {sample_dir}")

    buster = PoseBusters(config=args.config)
    per_sample: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="PoseBusters"):
        ligand = row["paths"]["ligand_sdf"]
        protein = row["paths"].get("pocket_relaxed")
        if not os.path.exists(ligand):
            per_sample.append({"complex_id": row["complex_id"], "sample_id": row["sample_id"], "error": "missing sdf"})
            continue
        try:
            if args.config == "dock" and protein and os.path.exists(protein):
                df = buster.bust(ligand, None, protein)
            else:
                df = buster.bust(ligand)
            series = df.iloc[0]
            result = row_passes(series)
        except Exception as exc:
            result = {"passes_all": False, "error": str(exc)}
        result.update({"complex_id": row["complex_id"], "sample_id": row["sample_id"]})
        per_sample.append(result)

    pass_rate = mean_std([float(item.get("passes_all", False)) for item in per_sample if "error" not in item or item.get("passes_all") is not None])
    test_keys = sorted({k for item in per_sample for k in item.keys() if k not in {"complex_id", "sample_id", "error", "passes_all"}})
    per_test = {
        key: mean_std([float(item[key]) for item in per_sample if isinstance(item.get(key), bool)])
        for key in test_keys
    }
    summary = {"pass_rate": pass_rate["mean"], "n": len(per_sample), "per_test": per_test}
    payload = {"summary": summary, "per_sample": per_sample}
    out_json = args.out_json or os.path.join(sample_dir, "ligand_posebusters.json")
    write_json(out_json, payload)
    import json

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
