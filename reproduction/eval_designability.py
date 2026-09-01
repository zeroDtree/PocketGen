"""Self-consistency / designability evaluation using ESMFold and TMscore.

Default protocol: ProteinMPNN x8 from the generated backbone, ESMFold each
sequence, keep the lowest scRMSD (paper Fig. 2 / README designability).

--sequence_source codesign folds the model's own sequence once (Table 1 Co,
Delta scTM +co only). This folder folds with ESMFold (Table S2), not AF2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    GENIE_ROOT,
    append_jsonl,
    ca_coords,
    default_tmscore_path,
    ensure_dir,
    ensure_repo_on_path,
    load_jsonl,
    load_sample_rows,
    mean_std,
    parse_pdb_residues,
    parse_tmscore_output,
    residue_indices_by_resseq,
    residue_sequence,
    resolve_repo_path,
    select_residues_by_index,
    write_json,
)

ensure_repo_on_path()

from utils.rmsd import compute_rmsd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PocketGen designability with ESMFold/TMscore.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tmscore", type=str, default=default_tmscore_path())
    parser.add_argument(
        "--sequence_source",
        type=str,
        choices=("codesign", "proteinmpnn"),
        default="proteinmpnn",
        help="proteinmpnn: MPNN x8 main protocol. codesign: Co Delta scTM add-on.",
    )
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="ESMFold axial-attention chunk size. Default 256; 0 leaves the model default (no chunking).",
    )
    parser.add_argument("--whole_scrmsd_thresh", type=float, default=2.0)
    parser.add_argument("--pocket_scrmsd_thresh", type=float, default=1.0)
    parser.add_argument("--fold_original", action="store_true", help="Also fold the original sequence for Delta scTM.")
    parser.set_defaults(fold_original=True)
    parser.add_argument("--skip_original", dest="fold_original", action="store_false")
    return parser.parse_args()


def default_out_paths(sample_dir: str, sequence_source: str) -> Tuple[str, str]:
    if sequence_source == "codesign":
        return (
            os.path.join(sample_dir, "designability_codesign.json"),
            os.path.join(sample_dir, "designability_codesign.jsonl"),
        )
    return (
        os.path.join(sample_dir, "designability.json"),
        os.path.join(sample_dir, "designability.jsonl"),
    )


def pdb_sequence(pdb_path: str) -> str:
    return residue_sequence(parse_pdb_residues(pdb_path))


def write_pdb_chain_a(src: str, dest: str) -> str:
    """Copy a PDB with ATOM chain IDs rewritten to A for Genie ProteinMPNN."""
    with open(src, "r") as fin, open(dest, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and len(line) > 21:
                line = line[:21] + "A" + line[22:]
            fout.write(line)
    return dest


def run_tmscore(tmscore_exec: str, pdb_a: str, pdb_b: str) -> Dict[str, float]:
    if not os.path.exists(tmscore_exec):
        raise FileNotFoundError(f"TMscore binary not found: {tmscore_exec}")
    proc = subprocess.run(
        [tmscore_exec, pdb_a, pdb_b],
        check=False,
        capture_output=True,
        text=True,
    )
    parsed = parse_tmscore_output(proc.stdout)
    if "rmsd" not in parsed or "tm" not in parsed:
        raise RuntimeError(f"Failed to parse TMscore output for {pdb_a} vs {pdb_b}:\n{proc.stdout}\n{proc.stderr}")
    return parsed


def pocket_and_whole_rmsd(
    generated_pdb: str,
    predicted_pdb: str,
    designed_resseq: Sequence[int],
    full_seq_idx: Optional[Sequence[int]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    gen_res = parse_pdb_residues(generated_pdb)
    pred_res = parse_pdb_residues(predicted_pdb)
    gen_ca = ca_coords(gen_res)
    pred_ca = ca_coords(pred_res)
    n = min(len(gen_ca), len(pred_ca))
    whole = compute_rmsd(pred_ca[:n], gen_ca[:n]) if n else None
    pocket = None
    pocket_plddt = None
    if not designed_resseq:
        return whole, pocket, pocket_plddt
    gen_indices = residue_indices_by_resseq(gen_res, designed_resseq)
    gen_p = select_residues_by_index(gen_res, gen_indices)
    if len(pred_res) == len(gen_res) and gen_indices:
        pred_indices = gen_indices
    else:
        pred_indices = [int(x) for x in (full_seq_idx or [])]
    pred_p = select_residues_by_index(pred_res, pred_indices)
    gen_ca_p = ca_coords(gen_p)
    pred_ca_p = ca_coords(pred_p)
    m = min(len(gen_ca_p), len(pred_ca_p))
    if m:
        pocket = compute_rmsd(pred_ca_p[:m], gen_ca_p[:m])
        plddts = [res["plddt"] for res in pred_p if res.get("plddt") is not None]
        if plddts:
            pocket_plddt = float(np.mean(plddts))
    return whole, pocket, pocket_plddt


def _ensure_genie_on_path() -> None:
    genie_pipeline = os.path.join(GENIE_ROOT, "evaluations", "pipeline")
    genie_packages = os.path.join(GENIE_ROOT, "packages")
    for path in (genie_pipeline, genie_packages):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_esmfold(device: str, chunk_size: int = 256):
    _ensure_genie_on_path()
    from fold_models.esmfold import ESMFold  # type: ignore

    model = ESMFold()
    if hasattr(model, "model"):
        model.model = model.model.to(device)
        if chunk_size > 0 and hasattr(model.model, "set_chunk_size"):
            model.model.set_chunk_size(chunk_size)
    return model


def load_proteinmpnn():
    _ensure_genie_on_path()
    from inverse_fold_models.proteinmpnn import ProteinMPNN  # type: ignore

    rootdir = os.path.join(GENIE_ROOT, "packages", "ProteinMPNN")
    if not os.path.isdir(rootdir):
        rootdir = os.path.join(GENIE_ROOT, "evaluations", "pipeline", "packages", "ProteinMPNN")
    return ProteinMPNN(rootdir=rootdir, num_samples=8)


def sequences_from_mpnn_text(text: str) -> List[str]:
    seqs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        seqs.append(line.replace("/", ""))
    return seqs


def fold_sequence(fold_model, seq: str, out_pdb: str) -> str:
    pdb_str, pae = fold_model.predict(seq)
    with open(out_pdb, "w") as handle:
        handle.write(pdb_str)
    pae_path = out_pdb.replace(".pdb", ".pae.txt")
    np.savetxt(pae_path, pae, fmt="%.3f")
    return out_pdb


def evaluate_one_structure(
    fold_model,
    tmscore_exec: str,
    generated_pdb: str,
    sequences: Sequence[str],
    designed_resseq: Sequence[int],
    full_seq_idx: Sequence[int],
    work_dir: str,
    tag: str,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for i, seq in enumerate(sequences):
        pred_pdb = os.path.join(work_dir, f"{tag}_resample_{i}.pdb")
        fold_sequence(fold_model, seq, pred_pdb)
        tm = run_tmscore(tmscore_exec, generated_pdb, pred_pdb)
        whole, pocket, pocket_plddt = pocket_and_whole_rmsd(
            generated_pdb, pred_pdb, designed_resseq, full_seq_idx=full_seq_idx
        )
        scrmsd = pocket if pocket is not None else tm["rmsd"]
        record = {
            "seq": seq,
            "pred_pdb": pred_pdb,
            "scTM": tm["tm"],
            "scRMSD_tmscore": tm["rmsd"],
            "scRMSD_whole": whole if whole is not None else tm["rmsd"],
            "scRMSD_pocket": pocket,
            "pLDDT_pocket": pocket_plddt,
        }
        if best is None or scrmsd < (best["scRMSD_pocket"] if best["scRMSD_pocket"] is not None else best["scRMSD_whole"]):
            best = record
    return best or {}


def unique_sample_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[Any, int], Dict[str, Any]] = {}
    for row in rows:
        by_key[(row["complex_id"], int(row["sample_id"]))] = row
    return list(by_key.values())


def summarize(per_sample: Sequence[Dict[str, Any]], sequence_source: str) -> Dict[str, Any]:
    return {
        "sequence_source": sequence_source,
        "designability": mean_std([float(item["designable"]) for item in per_sample])["mean"],
        "scTM": mean_std([item.get("scTM") for item in per_sample]),
        "scRMSD_whole": mean_std([item.get("scRMSD_whole") for item in per_sample]),
        "scRMSD_pocket": mean_std([item.get("scRMSD_pocket") for item in per_sample]),
        "pLDDT_pocket": mean_std([item.get("pLDDT_pocket") for item in per_sample]),
        "delta_scTM": mean_std([item.get("delta_scTM") for item in per_sample]),
        "n": len(per_sample),
    }


def main() -> None:
    args = parse_args()
    sample_dir = resolve_repo_path(args.sample_dir)
    rows = load_sample_rows(sample_dir)
    if not rows:
        raise SystemExit(f"No samples found in {sample_dir}")
    tmscore_exec = resolve_repo_path(args.tmscore)
    default_json, _ = default_out_paths(sample_dir, args.sequence_source)
    out_json = args.out_json or default_json
    jsonl_path = os.path.splitext(out_json)[0] + ".jsonl"
    work_dir = ensure_dir(os.path.join(sample_dir, "designability_work"))

    done_rows = load_jsonl(jsonl_path)
    done_keys = {(item["complex_id"], int(item["sample_id"])) for item in done_rows}
    pending = [row for row in rows if (row["complex_id"], int(row["sample_id"])) not in done_keys]
    print(f"Designability: {len(done_keys)} done, {len(pending)} remaining ({args.sequence_source})")

    if pending:
        print("Loading ESMFold...")
        fold_model = load_esmfold(args.device, chunk_size=args.chunk_size)
        mpnn_model = load_proteinmpnn() if args.sequence_source == "proteinmpnn" else None
        original_cache: Dict[str, Dict[str, float]] = {}
        for row in tqdm(pending, desc="Designability"):
            cid = row["complex_id"]
            sid = int(row["sample_id"])
            generated_pdb = row["paths"].get("whole_relaxed") or row["paths"]["whole_pdb"]
            if not os.path.exists(generated_pdb):
                print(f"Missing generated PDB: {generated_pdb}")
                continue
            if args.sequence_source == "codesign":
                sequences = [pdb_sequence(generated_pdb)]
            else:
                mpnn_pdb = write_pdb_chain_a(
                    generated_pdb, os.path.join(work_dir, f"{cid}_{sid}_mpnn.pdb")
                )
                sequences = sequences_from_mpnn_text(mpnn_model.predict(mpnn_pdb))
            tag = f"{cid}_{sid}"
            best = evaluate_one_structure(
                fold_model,
                tmscore_exec,
                generated_pdb,
                sequences,
                row.get("designed_resseq") or [],
                row.get("full_seq_idx") or [],
                work_dir,
                tag,
            )
            delta_sctm = None
            if args.fold_original and row.get("ref_protein_pdb") and os.path.exists(row["ref_protein_pdb"]):
                if cid not in original_cache:
                    orig_seq = pdb_sequence(row["ref_protein_pdb"])
                    orig_pred = os.path.join(work_dir, f"{cid}_original_esmfold.pdb")
                    fold_sequence(fold_model, orig_seq, orig_pred)
                    original_cache[cid] = run_tmscore(tmscore_exec, row["ref_protein_pdb"], orig_pred)
                orig_tm = original_cache[cid]["tm"]
                if best.get("scTM") is not None:
                    delta_sctm = float(best["scTM"]) - float(orig_tm)
            whole = best.get("scRMSD_whole")
            pocket = best.get("scRMSD_pocket")
            designable = (
                whole is not None
                and pocket is not None
                and whole < args.whole_scrmsd_thresh
                and pocket < args.pocket_scrmsd_thresh
            )
            record = {
                "complex_id": cid,
                "sample_id": sid,
                "sequence_source": args.sequence_source,
                "scTM": best.get("scTM"),
                "scRMSD_whole": whole,
                "scRMSD_pocket": pocket,
                "pLDDT_pocket": best.get("pLDDT_pocket"),
                "delta_scTM": delta_sctm,
                "designable": designable,
            }
            append_jsonl(jsonl_path, record)

    per_sample = unique_sample_rows(load_jsonl(jsonl_path))
    summary = summarize(per_sample, args.sequence_source)
    payload = {"summary": summary, "per_sample": per_sample}
    write_json(out_json, payload)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json} and {jsonl_path}")


if __name__ == "__main__":
    main()
