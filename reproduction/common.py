"""Shared helpers for PocketGen CrossDocked reproduction scripts."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENIE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "genie"))
DEFAULT_TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "crossdocked_sample")
SAMPLES_JSONL = "samples.jsonl"


def load_sample_rows(sample_dir: str) -> List[Dict[str, Any]]:
    """Load DONE sample payloads from the ResumableSaver ledger under sample_dir.

    Eval scripts must not mutate the ledger, so auto_recover is disabled.
    """
    from reproduction.utils.resumable_saver import ResumableSaver, SaveStatus

    rows: List[Dict[str, Any]] = []
    with ResumableSaver(sample_dir, auto_recover=False, retry_failed=False) as saver:
        for record in saver.list_records(SaveStatus.DONE):
            payload = saver.load(record.sample_id)
            if isinstance(payload, dict):
                rows.append(payload)
    rows.sort(key=lambda row: (int(row.get("complex_index", 0)), int(row.get("sample_id", 0))))
    return rows

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def ensure_repo_on_path() -> str:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def tensor_to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def make_complex_id(index: int, ligand_filename: str) -> str:
    stem = os.path.splitext(os.path.basename(ligand_filename))[0]
    return f"{index:03d}_{stem}"


def samples_path(sample_dir: str) -> str:
    return os.path.join(sample_dir, SAMPLES_JSONL)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def group_by_complex(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["complex_id"], []).append(row)
    return grouped


def parse_pdb_residues(pdb_path: str) -> List[Dict[str, Any]]:
    """Parse protein residues from a PDB, keeping CA coordinates and pLDDT (B-factor)."""
    residues: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    order: List[Tuple[str, int, str]] = []
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            key = (chain, resseq, res_name)
            if key not in residues:
                residues[key] = {
                    "chain": chain,
                    "resseq": resseq,
                    "resname": res_name,
                    "aa": AA3_TO_1.get(res_name, "X"),
                    "atoms": {},
                    "plddt": None,
                }
                order.append(key)
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            residues[key]["atoms"][atom_name] = coord
            if atom_name == "CA":
                try:
                    residues[key]["plddt"] = float(line[60:66])
                except ValueError:
                    residues[key]["plddt"] = None
    return [residues[key] for key in order]


def residue_sequence(residues: Sequence[Dict[str, Any]]) -> str:
    return "".join(res.get("aa", "X") for res in residues)


def ca_coords(residues: Sequence[Dict[str, Any]]) -> np.ndarray:
    coords = [res["atoms"]["CA"] for res in residues if "CA" in res["atoms"]]
    return np.stack(coords, axis=0) if coords else np.zeros((0, 3), dtype=np.float64)


def select_residues_by_resseq(
    residues: Sequence[Dict[str, Any]], resseqs: Sequence[int]
) -> List[Dict[str, Any]]:
    wanted = set(int(x) for x in resseqs)
    return [res for res in residues if res["resseq"] in wanted]


def residue_indices_by_resseq(
    residues: Sequence[Dict[str, Any]], resseqs: Sequence[int]
) -> List[int]:
    wanted = set(int(x) for x in resseqs)
    return [i for i, res in enumerate(residues) if res["resseq"] in wanted]


def select_residues_by_index(
    residues: Sequence[Dict[str, Any]], indices: Sequence[int]
) -> List[Dict[str, Any]]:
    n = len(residues)
    selected: List[Dict[str, Any]] = []
    for raw in indices:
        idx = int(raw)
        if 0 <= idx < n:
            selected.append(residues[idx])
    return selected


def sequence_recovery(pred: str, ref: str) -> Optional[float]:
    if not pred or not ref or len(pred) != len(ref):
        n = min(len(pred), len(ref))
        if n == 0:
            return None
        return sum(a == b for a, b in zip(pred[:n], ref[:n])) / float(n)
    return sum(a == b for a, b in zip(pred, ref)) / float(len(ref))


def pairwise_identity(sequences: Sequence[str]) -> Optional[float]:
    seqs = [s for s in sequences if s]
    if len(seqs) < 2:
        return None
    total = 0.0
    count = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            ident = sequence_recovery(seqs[i], seqs[j])
            if ident is None:
                continue
            total += ident
            count += 1
    if count == 0:
        return None
    return total / count


def mean_std(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not nums:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(nums, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def topk_mean(values: Sequence[float], k: int, lower_is_better: bool = True) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values, reverse=not lower_is_better)
    take = ordered[: min(k, len(ordered))]
    return float(np.mean(take))


def histogram_kl(
    values_p: Sequence[float],
    values_q: Sequence[float],
    bins: int = 50,
    range_limits: Optional[Tuple[float, float]] = None,
) -> Optional[float]:
    if len(values_p) < 2 or len(values_q) < 2:
        return None
    if range_limits is None:
        lo = min(min(values_p), min(values_q))
        hi = max(max(values_p), max(values_q))
        if lo == hi:
            hi = lo + 1e-6
        range_limits = (lo, hi)
    p_hist, _ = np.histogram(values_p, bins=bins, range=range_limits, density=False)
    q_hist, _ = np.histogram(values_q, bins=bins, range=range_limits, density=False)
    p = p_hist.astype(np.float64) + 1e-8
    q = q_hist.astype(np.float64) + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def parse_tmscore_output(text: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for line in text.splitlines():
        if line[:4] == "RMSD":
            results["rmsd"] = float(line.split("=")[1])
        elif line[:8] == "TM-score":
            results["tm"] = float(line.split("(")[0].split("=")[1])
        elif line[:6] == "Number":
            results["seqlen"] = float(line.split("=")[1])
    return results


def default_tmscore_path() -> str:
    return os.path.join(GENIE_ROOT, "packages", "TMscore", "TMscore")
