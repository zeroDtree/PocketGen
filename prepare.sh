#!/usr/bin/env bash
# Prepare PocketGen micromamba env for GPU training (PyTorch 1.13.1+cu117).
# Usage: bash prepare.sh [--skip-esm] [--process-data] [--env-name NAME]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="pocketgen"
SKIP_ESM=0
PROCESS_DATA=0
MIN_FREE_GB=20
PYG_WHL_INDEX="https://data.pyg.org/whl/torch-1.13.0+cu117.html"

info() { printf '[prepare] %s\n' "$*"; }
warn() { printf '[prepare] WARNING: %s\n' "$*" >&2; }
die() { printf '[prepare] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash prepare.sh [options]

Options:
  --skip-esm          Do not prefetch ESM2 weights
  --process-data      Run CrossDocked extract/split if local data files exist
  --env-name NAME     micromamba env name (default: pocketgen)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-esm) SKIP_ESM=1; shift ;;
    --process-data) PROCESS_DATA=1; shift ;;
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done

mamba_run() {
  micromamba run -n "$ENV_NAME" "$@"
}

free_gb_home() {
  df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0
}

preflight() {
  command -v micromamba >/dev/null 2>&1 || die "micromamba not found on PATH"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (NVIDIA driver required)"
  command -v wget >/dev/null 2>&1 || die "wget not found on PATH"

  local free_gb
  free_gb="$(free_gb_home)"
  if [[ -z "$free_gb" || "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    die "Need at least ${MIN_FREE_GB}GB free on \$HOME (have ${free_gb:-unknown}GB). Free space or run: micromamba clean -a"
  fi
  info "Disk free on \$HOME: ${free_gb}GB"
  info "Using driver + conda pytorch-cuda=11.7 runtime (no local CUDA toolkit required)"
  nvidia-smi -L || die "nvidia-smi -L failed"
}

env_exists() {
  micromamba env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"
}

create_env() {
  if env_exists; then
    info "Environment '$ENV_NAME' already exists; reusing"
  else
    info "Creating environment '$ENV_NAME' (python=3.8)"
    micromamba create -n "$ENV_NAME" python=3.8 pip -y
  fi
}

install_git() {
  # micromamba run does not inherit base-env PATH; pip git+https needs git
  # inside this env (AutoDockTools_py3).
  info "Installing git into '$ENV_NAME'"
  micromamba install -n "$ENV_NAME" -c conda-forge git -y
}

install_cuda_pytorch() {
  # Pin MKL 2023.*: MKL 2024+/2026 with pytorch 1.13 often breaks import with
  #   libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
  info "Installing CUDA PyTorch 1.13.1 + pytorch-cuda=11.7 (MKL 2023.* pinned)"
  micromamba install -n "$ENV_NAME" -y --channel-priority flexible \
    -c pytorch -c nvidia -c conda-forge \
    'pytorch=1.13.1=py3.8_cuda11.7*' \
    'pytorch-cuda=11.7' \
    'pytorch-mutex=1.0=cuda' \
    'mkl=2023.*' \
    'mkl-include=2023.*'
}

assert_cuda() {
  local label="$1"
  info "Checking CUDA availability ($label)"
  mamba_run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(1)
print("device", torch.cuda.get_device_name(0))
PY
}

torch_import_ok() {
  mamba_run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
}

repair_cuda_if_needed() {
  if torch_import_ok; then
    info "CUDA still available after conda-forge stack"
    return 0
  fi
  warn "torch import/CUDA broken after conda installs; re-pinning CUDA PyTorch + MKL 2023.*"
  install_cuda_pytorch
  assert_cuda "after repair" || die "torch/CUDA still broken after repair. Aborting."
}

install_conda_forge_stack() {
  info "Installing conda-forge scientific stack (keep mkl=2023.*)"
  micromamba install -n "$ENV_NAME" -y -c conda-forge \
    rdkit openbabel tensorboard pyyaml easydict python-lmdb \
    openmm pdbfixer flask \
    numpy swig boost-cpp sphinx sphinx_rtd_theme \
    'mkl=2023.*' \
    'mkl-include=2023.*'
}

install_pip_and_pyg() {
  info "Installing pip packages"
  mamba_run pip install \
    meeko==0.1.dev3 wandb scipy pdb2pqr vina==1.2.2 \
    fair-esm==2.0.0 \
    omegaconf==2.3.0 e3nn==0.5.1 einops==0.7.0 biopython==1.79 biotite \
    gdown \
    torch-geometric

  info "Installing AutoDockTools_py3 from GitHub"
  mamba_run python -m pip install \
    "git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"

  info "Installing PyG CUDA wheels (cu117, binary only)"
  mamba_run pip install --only-binary=:all: \
    torch-scatter==2.1.1+pt113cu117 \
    torch-sparse==0.6.17+pt113cu117 \
    torch-cluster==1.6.1+pt113cu117 \
    torch-spline-conv==1.2.2+pt113cu117 \
    -f "$PYG_WHL_INDEX"
}

smoke_test() {
  info "Import smoke test"
  mamba_run python - <<'PY'
import torch
import torch_geometric
import torch_scatter
from torch_scatter import scatter_sum
import esm
import omegaconf
import e3nn
import Bio
import openbabel
import openmm

assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(4, 8, device="cuda")
i = torch.tensor([0, 0, 1, 1], device="cuda")
y = scatter_sum(x, i, dim=0)
print("smoke_ok", torch.__version__, torch_scatter.__version__, tuple(y.shape))
PY
}

prefetch_esm() {
  if [[ "$SKIP_ESM" -eq 1 ]]; then
    info "Skipping ESM prefetch (--skip-esm)"
    return 0
  fi
  local dir="$HOME/.cache/torch/hub/checkpoints"
  mkdir -p "$dir"
  local main="$dir/esm2_t33_650M_UR50D.pt"
  local reg="$dir/esm2_t33_650M_UR50D-contact-regression.pt"
  if [[ -f "$main" ]]; then
    info "ESM weights already present: $main"
  else
    info "Downloading ESM2 t33 weights"
    wget -c -O "$main" \
      https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
  fi
  if [[ -f "$reg" ]]; then
    info "ESM contact-regression already present: $reg"
  else
    info "Downloading ESM2 contact-regression weights"
    wget -c -O "$reg" \
      https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
  fi
}

report_data() {
  local split="data/crossdocked_v1.1_rmsd1.0_pocket10/split.pt"
  local lmdb="data/crossdocked_v1.1_rmsd1.0_pocket10_processed.lmdb"
  local tar="data/crossdocked_v1.1_rmsd1.0.tar.gz"
  local names="data/split_by_name.pt"

  info "Data status:"
  [[ -f "$tar" ]] && info "  found $tar" || warn "  missing $tar"
  [[ -f "$names" ]] && info "  found $names" || warn "  missing $names"
  [[ -f "$split" ]] && info "  found $split" || warn "  missing $split"
  [[ -e "$lmdb" ]] && info "  found $lmdb" || warn "  missing $lmdb"

  cat <<'EOF'

Manual data downloads (Google Drive; not automated by this script):
  crossdocked_v1.1_rmsd1.0.tar.gz:
    https://drive.google.com/file/d/1U0ZgITApL7EClcQiiVK_OevAV_H20L4d/view?usp=sharing
  split_by_name.pt:
    https://drive.google.com/file/d/1UVJmLvx-kcorMyDDR_LPCqR8dFPuoRtI/view?usp=sharing
  Put both under ./data then: bash prepare.sh --process-data
  Or use processed LMDB from https://zenodo.org/records/10125312
EOF
}

process_data_if_requested() {
  if [[ "$PROCESS_DATA" -ne 1 ]]; then
    report_data
    return 0
  fi

  local tar="data/crossdocked_v1.1_rmsd1.0.tar.gz"
  local names="data/split_by_name.pt"
  local raw_dir="data/crossdocked_v1.1_rmsd1.0"

  if [[ ! -f "$names" ]]; then
    report_data
    die "Missing $names; download it before --process-data"
  fi
  if [[ ! -d "$raw_dir" ]]; then
    if [[ -f "$tar" ]]; then
      info "Extracting $tar"
      mkdir -p data
      tar -xzf "$tar" -C data
    else
      report_data
      die "Missing $raw_dir and $tar; download/extract CrossDocked before --process-data"
    fi
  fi

  info "Running data_preparation/extract_pockets.py"
  mamba_run python data_preparation/extract_pockets.py
  info "Running data_preparation/split_pl_dataset.py"
  mamba_run python data_preparation/split_pl_dataset.py
  info "Dataset processing finished"
}

print_next_steps() {
  cat <<EOF

========================================================================
PocketGen env '$ENV_NAME' is ready for GPU training.

Verify:
  micromamba run -n $ENV_NAME python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Train (bash):
  CUDA_VISIBLE_DEVICES=0 micromamba run -n $ENV_NAME python train_recycle.py --config ./configs/train_model.yml

Train (fish helpers from ~/.config/fish/config.fish):
  micromamba activate $ENV_NAME
  cuda 0 python train_recycle.py --config ./configs/train_model.yml
========================================================================
EOF
}

main() {
  info "Repo root: $ROOT"
  preflight
  create_env
  install_git
  install_cuda_pytorch
  assert_cuda "after pytorch install" || die "CUDA PyTorch install failed verification"
  install_conda_forge_stack
  repair_cuda_if_needed
  install_pip_and_pyg
  smoke_test
  prefetch_esm
  process_data_if_requested
  print_next_steps
}

main "$@"
