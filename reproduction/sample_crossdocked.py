"""Sample PocketGen pockets on the CrossDocked test split.

Writes per-complex PDB/SDF files and stores AAR/RMSD rows in a ResumableSaver
ledger (SQLite + pickles). Vina and other metrics are computed by separate
evaluation scripts.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from typing import List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    ensure_dir,
    ensure_repo_on_path,
    make_complex_id,
    parse_pdb_residues,
    residue_sequence,
    resolve_repo_path,
    select_residues_by_resseq,
    sequence_recovery,
    tensor_to_python,
)
from reproduction.utils.resumable_saver import ResumableSaver, build_sample_id

ensure_repo_on_path()

import esm  # noqa: E402
import torch  # noqa: E402
from torch_geometric.transforms import Compose  # noqa: E402
from tqdm import tqdm  # noqa: E402

from models.PD import Pocket_Design_new  # noqa: E402
import models.PD as pd_module  # noqa: E402
from utils.data import collate_mols_block  # noqa: E402
from utils.datasets import get_dataset  # noqa: E402
from utils.misc import load_config, seed_all  # noqa: E402
from utils.transforms import FeaturizeLigandAtom, FeaturizeProteinAtom  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample PocketGen on the CrossDocked test set.")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "train_model.yml"))
    parser.add_argument("--ckpt", type=str, default=None, help="Override config.model.checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_complexes", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2089)
    return parser.parse_args()


_RELAX_FALLBACK_PATHS = set()


def _finite_positions(positions) -> bool:
    import numpy as np

    try:
        coords = np.array([[v.x, v.y, v.z] for v in positions], dtype=np.float64)
    except Exception:
        return False
    return bool(np.isfinite(coords).all())


def safe_openmm_relax(pdb, out_pdb=None, excluded_chains=None, inverse_exclude=False):
    """Relax a PDB without filling residue-number gaps (pocket fragments)."""
    from openmm import CustomExternalForce, LangevinIntegrator
    from openmm.app import ForceField, HBonds, Modeller, PDBFile, Simulation
    from pdbfixer import PDBFixer

    try:
        from openmm import unit as openmm_unit
    except ImportError:
        from simtk import unit as openmm_unit

    if out_pdb is None:
        out_pdb = pdb[:-4] + "_relaxed.pdb"
    if excluded_chains is None:
        excluded_chains = []

    try:
        tolerance_in_kj = 2.39 * openmm_unit.kilojoules_per_mole / openmm_unit.kilocalories_per_mole
        stiffness = 10.0 * openmm_unit.kilocalories_per_mole / (openmm_unit.angstroms ** 2)

        fixer = PDBFixer(pdb)
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        force_field = ForceField("amber99sb.xml")
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)
        system = force_field.createSystem(modeller.topology, constraints=HBonds)

        force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for param in ("x0", "y0", "z0"):
            force.addPerParticleParameter(param)

        for residue in modeller.topology.residues():
            if (not inverse_exclude and residue.chain.id in excluded_chains) or (
                inverse_exclude and residue.chain.id not in excluded_chains
            ):
                for atom in residue.atoms():
                    system.setParticleMass(atom.index, 0)
            for atom in residue.atoms():
                if atom.element.name != "hydrogen":
                    force.addParticle(atom.index, modeller.positions[atom.index])

        system.addForce(force)
        integrator = LangevinIntegrator(0, 0.01, 0.0)
        simulation = Simulation(modeller.topology, system, integrator)
        if not _finite_positions(modeller.positions):
            raise RuntimeError("non-finite coordinates before minimization")
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance_in_kj)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        if not _finite_positions(state.getPositions()):
            raise RuntimeError("non-finite coordinates after minimization")
        with open(out_pdb, "w") as fout:
            PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)
        return out_pdb
    except Exception as exc:
        print(f"Warning: openmm_relax failed for {pdb}: {exc}; copying unrelaxed PDB to {out_pdb}")
        if os.path.exists(pdb):
            shutil.copyfile(pdb, out_pdb)
        _RELAX_FALLBACK_PATHS.add(os.path.abspath(out_pdb))
        return out_pdb


def patch_openmm_relax() -> None:
    """Upstream openmm_relax misses `unit` and fills pocket residue-number gaps."""
    from utils import relax as relax_mod

    try:
        from openmm import unit as openmm_unit
    except ImportError:
        from simtk import unit as openmm_unit

    relax_mod.unit = openmm_unit
    relax_mod.openmm_relax = safe_openmm_relax
    pd_module.openmm_relax = safe_openmm_relax


def patch_sampling_temperature(temperature: float) -> None:
    original = pd_module.sample_from_categorical

    def sample_from_categorical(logits=None, temperature_arg=None):
        used = temperature if temperature_arg is None else temperature_arg
        return original(logits=logits, temperature=used)

    pd_module.sample_from_categorical = sample_from_categorical


def absolutize_example(data, orig_data_path: str, pocket10_path: str):
    example = dict(data)
    for key, root in (
        ("whole_protein_name", orig_data_path),
        ("protein_filename", pocket10_path),
        ("ligand_filename", pocket10_path),
    ):
        value = example[key]
        if not os.path.isabs(value):
            example[key] = os.path.join(root, value)
    return example


def designed_resseqs(example) -> list:
    mask = example["protein_edit_residue"]
    res_idx = example["res_idx"]
    if hasattr(mask, "bool"):
        mask = mask.bool()
    selected = res_idx[mask]
    return [int(x) for x in tensor_to_python(selected)]


def consecutive_runs(ids: List[int]) -> List[List[int]]:
    """Split integer ids into maximal consecutive runs."""
    if not ids:
        return []
    runs: List[List[int]] = []
    current = [ids[0]]
    for value in ids[1:]:
        if value == current[-1] + 1:
            current.append(value)
        else:
            runs.append(current)
            current = [value]
    runs.append(current)
    return runs


def sample_paths(complex_dir: str, sample_id: int) -> dict:
    return {
        "dir": complex_dir,
        "pocket_pdb": os.path.join(complex_dir, f"{sample_id}.pdb"),
        "pocket_relaxed": os.path.join(complex_dir, f"{sample_id}_relaxed.pdb"),
        "whole_pdb": os.path.join(complex_dir, f"{sample_id}_whole.pdb"),
        "whole_relaxed": os.path.join(complex_dir, f"{sample_id}_whole_relaxed.pdb"),
        "ligand_sdf": os.path.join(complex_dir, f"{sample_id}.sdf"),
        "orig_pocket_pdb": os.path.join(complex_dir, f"{sample_id}_orig.pdb"),
    }


def record_sample(example, complex_id: str, complex_index: int, sample_id: int, paths: dict, batch_aar, batch_rmsd) -> dict:
    designed = designed_resseqs(example)
    orig_seq = ""
    gen_seq = ""
    orig_full = ""
    gen_full = ""
    if os.path.exists(paths["orig_pocket_pdb"]):
        orig_res = parse_pdb_residues(paths["orig_pocket_pdb"])
        orig_full = residue_sequence(orig_res)
        orig_seq = residue_sequence(select_residues_by_resseq(orig_res, designed)) if designed else orig_full
    pocket_for_seq = paths["pocket_relaxed"] if os.path.exists(paths["pocket_relaxed"]) else paths["pocket_pdb"]
    if os.path.exists(pocket_for_seq):
        gen_res = parse_pdb_residues(pocket_for_seq)
        gen_full = residue_sequence(gen_res)
        gen_seq = residue_sequence(select_residues_by_resseq(gen_res, designed)) if designed else gen_full
    aar = sequence_recovery(gen_seq, orig_seq)
    return {
        "complex_id": complex_id,
        "complex_index": complex_index,
        "sample_id": sample_id,
        "aar": aar,
        "batch_aar": tensor_to_python(batch_aar),
        "rmsd": tensor_to_python(batch_rmsd),
        "orig_pocket_seq": orig_seq,
        "gen_pocket_seq": gen_seq,
        "orig_r10_seq": orig_full,
        "gen_r10_seq": gen_full,
        "full_seq": example["seq"] if isinstance(example["seq"], str) else str(example["seq"]),
        "designed_resseq": designed,
        "full_seq_idx": [int(x) for x in tensor_to_python(example["full_seq_idx"])],
        "r10_idx": [int(x) for x in tensor_to_python(example["r10_idx"])],
        "ref_protein_pdb": example["whole_protein_name"],
        "ref_pocket_pdb": example["protein_filename"],
        "ref_ligand_sdf": example["ligand_filename"],
        "paths": paths,
        "relax_fallback": any(
            os.path.abspath(paths[key]) in _RELAX_FALLBACK_PATHS
            for key in ("pocket_relaxed", "whole_relaxed")
        ),
    }


def job_id(complex_id: str, sample_id: int) -> str:
    return build_sample_id(complex_id, sample_id)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    seed_all(args.seed)
    patch_openmm_relax()
    patch_sampling_temperature(args.temperature)

    orig_data_path = resolve_repo_path(config.model.orig_data_path)
    pocket10_path = resolve_repo_path(config.dataset.path)
    config.dataset.path = pocket10_path
    config.dataset.split = resolve_repo_path(config.dataset.split)
    config.model.orig_data_path = orig_data_path
    config.model.pocket10_path = pocket10_path
    ckpt_path = resolve_repo_path(args.ckpt or config.model.checkpoint)
    config.model.checkpoint = ckpt_path

    out_dir = resolve_repo_path(args.out_dir)
    ensure_dir(out_dir)

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    name = "esm2_t33_650M_UR50D"
    _, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loading CrossDocked test split...")
    _, subsets = get_dataset(config=config.dataset, transform=transform)
    test_set = subsets["test"]
    n_complexes = len(test_set)
    end_index = n_complexes if args.max_complexes is None else min(n_complexes, args.start_index + args.max_complexes)
    print(f"Test complexes: {n_complexes}; sampling [{args.start_index}, {end_index}) x {args.num_samples}")

    with ResumableSaver(out_dir, retry_failed=True) as saver:
        for complex_index in tqdm(range(args.start_index, end_index), desc="Complexes"):
            example = absolutize_example(test_set[complex_index], orig_data_path, pocket10_path)
            for required in ("whole_protein_name", "protein_filename", "ligand_filename"):
                if not os.path.exists(example[required]):
                    raise FileNotFoundError(f"Missing {required}: {example[required]}")

            complex_id = make_complex_id(complex_index, example["ligand_filename"])
            complex_dir = ensure_dir(os.path.join(out_dir, complex_id))

            for sample_id in range(args.num_samples):
                saver.register_pending(
                    job_id(complex_id, sample_id),
                    meta={
                        "complex_id": complex_id,
                        "complex_index": complex_index,
                        "sample_id": sample_id,
                    },
                )

            remaining = [
                sample_id
                for sample_id in range(args.num_samples)
                if not saver.is_done(job_id(complex_id, sample_id))
            ]
            if not remaining:
                print(f"Skipping {complex_id}: already sampled")
                continue

            with torch.no_grad():
                for run in consecutive_runs(remaining):
                    for start in range(0, len(run), args.batch_size):
                        sample_ids = run[start : start + args.batch_size]
                        assert sample_ids == list(range(sample_ids[0], sample_ids[0] + len(sample_ids)))

                        model.generate_id = sample_ids[0]
                        model.generate_id1 = sample_ids[0]
                        batch = collate_mols_block(
                            [copy.deepcopy(example) for _ in sample_ids],
                            batch_converter=batch_converter,
                        )
                        for key in batch:
                            if torch.is_tensor(batch[key]):
                                batch[key] = batch[key].to(args.device)

                        try:
                            aar, rmsd, _ = model.generate(batch, complex_dir)
                        except Exception as exc:
                            for sample_id in sample_ids:
                                saver.save_failure(job_id(complex_id, sample_id), exc)
                            print(f"Failed {complex_id} samples {sample_ids}: {exc}")
                            continue

                        for sample_id in sample_ids:
                            paths = sample_paths(complex_dir, sample_id)
                            row = record_sample(
                                example, complex_id, complex_index, sample_id, paths, aar, rmsd
                            )
                            saver.save_success(
                                job_id(complex_id, sample_id),
                                row,
                                meta={
                                    "complex_id": complex_id,
                                    "complex_index": complex_index,
                                    "sample_id": sample_id,
                                },
                            )
                        print(
                            f"{complex_id} samples {sample_ids}: "
                            f"batch_aar={tensor_to_python(aar):.4f} rmsd={tensor_to_python(rmsd):.4f}"
                        )

        print(f"Saver stats: {saver.stats()}")

    print(f"Wrote sample ledger under {out_dir}")


if __name__ == "__main__":
    main()
