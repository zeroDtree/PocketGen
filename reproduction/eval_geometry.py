"""Substructure geometry KL: backbone bonds and dihedral angles."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    histogram_kl,
    load_jsonl,
    parse_pdb_residues,
    resolve_repo_path,
    samples_path,
    write_json,
)

CHI_ATOMS = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute geometry KL divergence vs reference pockets.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--bins", type=int, default=36)
    return parser.parse_args()


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def collect_geometry(pdb_path: str) -> Dict[str, List[float]]:
    residues = parse_pdb_residues(pdb_path)
    geom: Dict[str, List[float]] = {
        "bond_CN": [],
        "bond_CO": [],
        "bond_CC": [],
        "phi": [],
        "psi": [],
        "omega": [],
        "chi1": [],
        "chi2": [],
        "chi3": [],
        "chi4": [],
    }
    for i, res in enumerate(residues):
        atoms = res["atoms"]
        if "CA" in atoms and "C" in atoms:
            geom["bond_CC"].append(dist(atoms["CA"], atoms["C"]))
        if "C" in atoms and "O" in atoms:
            geom["bond_CO"].append(dist(atoms["C"], atoms["O"]))
        if i + 1 < len(residues) and "C" in atoms and "N" in residues[i + 1]["atoms"]:
            geom["bond_CN"].append(dist(atoms["C"], residues[i + 1]["atoms"]["N"]))
        if i > 0:
            prev_atoms = residues[i - 1]["atoms"]
            if all(k in prev_atoms for k in ("C",)) and all(k in atoms for k in ("N", "CA", "C")):
                geom["phi"].append(dihedral(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"]))
        if i + 1 < len(residues):
            nxt = residues[i + 1]["atoms"]
            if all(k in atoms for k in ("N", "CA", "C")) and "N" in nxt:
                geom["psi"].append(dihedral(atoms["N"], atoms["CA"], atoms["C"], nxt["N"]))
            if all(k in atoms for k in ("CA", "C")) and all(k in nxt for k in ("N", "CA")):
                geom["omega"].append(dihedral(atoms["CA"], atoms["C"], nxt["N"], nxt["CA"]))
        chi_list = CHI_ATOMS.get(res["resname"], [])
        for chi_idx, names in enumerate(chi_list, start=1):
            if all(name in atoms for name in names):
                geom[f"chi{chi_idx}"].append(dihedral(*[atoms[name] for name in names]))
    return geom


def merge_geom(store: Dict[str, List[float]], extra: Dict[str, List[float]]) -> None:
    for key, values in extra.items():
        store.setdefault(key, []).extend(values)


def main() -> None:
    args = parse_args()
    sample_dir = resolve_repo_path(args.sample_dir)
    rows = load_jsonl(samples_path(sample_dir))
    if not rows:
        raise SystemExit(f"No samples found in {samples_path(sample_dir)}")

    generated: Dict[str, List[float]] = {}
    reference: Dict[str, List[float]] = {}
    seen_ref = set()
    for row in tqdm(rows, desc="Geometry"):
        gen_pdb = row["paths"].get("pocket_relaxed") or row["paths"]["pocket_pdb"]
        if os.path.exists(gen_pdb):
            merge_geom(generated, collect_geometry(gen_pdb))
        ref_pdb = row.get("ref_pocket_pdb")
        if ref_pdb and ref_pdb not in seen_ref and os.path.exists(ref_pdb):
            seen_ref.add(ref_pdb)
            merge_geom(reference, collect_geometry(ref_pdb))

    angle_keys = {"phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"}
    kl = {}
    counts = {}
    for key in sorted(set(generated) | set(reference)):
        gen_vals = generated.get(key, [])
        ref_vals = reference.get(key, [])
        counts[key] = {"generated": len(gen_vals), "reference": len(ref_vals)}
        if key in angle_keys:
            kl[key] = histogram_kl(gen_vals, ref_vals, bins=args.bins, range_limits=(-180.0, 180.0))
        else:
            kl[key] = histogram_kl(gen_vals, ref_vals, bins=args.bins)
    payload = {"kl_divergence": kl, "counts": counts}
    out_json = args.out_json or os.path.join(sample_dir, "geometry.json")
    write_json(out_json, payload)
    import json

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
