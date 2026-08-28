#!/usr/bin/env bash

# @help-begin
# Clone micromamba env eval from pocketgen (conda packages only), reinstall
# the PocketGen pip + PyG cu117 stack, then overlay Genie evaluation deps
# (ProteinMPNN, ESMFold/openfold, TMscore) without replacing PyTorch.
#
# Usage:
#   bash reproduction/shell/set_up_genie.sh [options]
#
# Env:
#   GENIE_DIR — Genie checkout directory (overridden by --genie-dir)
#   ENV_NAME — micromamba env name (overridden by --env-name)
#   BASE_ENV — micromamba env to clone (overridden by --base-env)
#   CUDA_HOME — CUDA 11.7 toolkit for compiling openfold
#               (script also installs conda-forge GCC 11 into ENV_NAME)
#
# Prerequisite: run bash prepare.sh so BASE_ENV exists (Python 3.8,
# PyTorch 1.13.1+cu117).
#
# If no options are passed, the default behavior is equivalent to:
#   bash reproduction/shell/set_up_genie.sh \
#     --base-env pocketgen --env-name eval \
#     --genie-dir <PocketGen>/../genie
# @help-end

# @help-options-begin
#   --base-env NAME         micromamba env to clone (default: pocketgen)
#   --env-name NAME         micromamba env to create (default: eval)
#   --genie-dir DIR         Genie checkout directory (default: <PocketGen>/../genie)
#   -h, --help              show help
# @help-options-end

set -euo pipefail

POCKETGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GENIE_DIR="${GENIE_DIR:-${POCKETGEN_ROOT}/../genie}"
ENV_NAME="${ENV_NAME:-eval}"
BASE_ENV="${BASE_ENV:-pocketgen}"
TORCH_PIN="1.13.1"
LIGHTNING_PIN="1.9.5"
OPENFOLD_GIT="git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
PYG_WHL_INDEX="https://data.pyg.org/whl/torch-1.13.0+cu117.html"
CONSTRAINT_FILE=""

info() { printf '[set_up_genie] %s\n' "$*"; }
warn() { printf '[set_up_genie] WARNING: %s\n' "$*" >&2; }
die() { printf '[set_up_genie] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
	awk '/^# @help-begin$/{f=1; next} /^# @help-end$/{f=0} f' "$0"
	printf '%s\n' '#' 'Options:' '#'
	awk '/^# @help-options-begin$/{f=1; next} /^# @help-options-end$/{f=0} f' "$0"
	exit 0
}

cleanup() {
	if [[ -n "${CONSTRAINT_FILE}" && -f "${CONSTRAINT_FILE}" ]]; then
		rm -f "${CONSTRAINT_FILE}"
	fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
	case "$1" in
	-h | --help)
		usage
		;;
	--genie-dir)
		GENIE_DIR="${2:?--genie-dir requires a directory}"
		shift 2
		;;
	--env-name)
		ENV_NAME="${2:?--env-name requires a name}"
		shift 2
		;;
	--base-env)
		BASE_ENV="${2:?--base-env requires a name}"
		shift 2
		;;
	*)
		echo "error: unknown option: $1" >&2
		awk '/^# @help-begin$/{f=1; next} /^# @help-end$/{f=0} f' "$0"
		printf '%s\n' '#' 'Options:' '#'
		awk '/^# @help-options-begin$/{f=1; next} /^# @help-options-end$/{f=0} f' "$0"
		exit 1
		;;
	esac
done

if [[ "${GENIE_DIR}" != /* ]]; then
	GENIE_DIR="$(pwd)/${GENIE_DIR}"
fi

if [[ ! -d "${GENIE_DIR}" ]]; then
	die "Genie not found at ${GENIE_DIR}"
fi
GENIE_DIR="$(cd "${GENIE_DIR}" && pwd)"

command -v micromamba >/dev/null 2>&1 || die "micromamba not found on PATH"

mamba_run() {
	micromamba run -n "${ENV_NAME}" "$@"
}

env_exists() {
	micromamba env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

env_compatible() {
	local name="$1"
	micromamba run -n "${name}" python - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 8):
    raise SystemExit(f"python {sys.version_info.major}.{sys.version_info.minor}, want 3.8")
if not torch.__version__.startswith("1.13.1"):
    raise SystemExit(f"torch {torch.__version__}, want 1.13.1")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
cuda = str(torch.version.cuda or "")
if not cuda.startswith("11.7"):
    raise SystemExit(f"torch.version.cuda={cuda!r}, want 11.7")
PY
}

ensure_clone() {
	env_exists "${BASE_ENV}" || die "base env '${BASE_ENV}' not found; run bash prepare.sh first"

	if ! env_exists "${ENV_NAME}"; then
		info "Cloning micromamba env '${ENV_NAME}' from '${BASE_ENV}'"
		micromamba create -n "${ENV_NAME}" --clone "${BASE_ENV}" -y
		return 0
	fi

	if env_compatible "${ENV_NAME}"; then
		info "Env '${ENV_NAME}' is already a compatible '${BASE_ENV}' clone; skipping clone"
		return 0
	fi

	warn "Env '${ENV_NAME}' is not a compatible '${BASE_ENV}' clone; removing and recreating"
	micromamba env remove -n "${ENV_NAME}" -y
	info "Cloning micromamba env '${ENV_NAME}' from '${BASE_ENV}'"
	micromamba create -n "${ENV_NAME}" --clone "${BASE_ENV}" -y
}

nvcc_release() {
	local nvcc_bin="$1"
	"${nvcc_bin}" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -1
}

nvcc_major() {
	local release
	release="$(nvcc_release "$1")"
	printf '%s\n' "${release%%.*}"
}

assert_nvcc_11() {
	local nvcc_bin="$1"
	local major
	major="$(nvcc_major "${nvcc_bin}")"
	if [[ "${major}" == "12" ]]; then
		die "nvcc CUDA 12.x is incompatible with ${BASE_ENV} PyTorch ${TORCH_PIN}+cu117 (${nvcc_bin})"
	fi
	if [[ "${major}" != "11" ]]; then
		die "need nvcc CUDA 11.x to compile openfold, found major ${major:-unknown} (${nvcc_bin})"
	fi
}

export_cuda_home() {
	local home="$1"
	export CUDA_HOME="${home}"
	export PATH="${CUDA_HOME}/bin:${PATH}"
	if [[ -d "${CUDA_HOME}/lib64" ]]; then
		export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
	fi
	info "Using CUDA_HOME=${CUDA_HOME} ($(nvcc --version | awk '/release/{print $NF}'))"
}

install_cuda_dev_stack() {
	info "Installing CUDA 11.7 nvcc + library headers into '${ENV_NAME}' (pytorch pin kept)"
	micromamba install -n "${ENV_NAME}" -y --channel-priority flexible \
		-c nvidia -c conda-forge \
		'cuda-nvcc=11.7.*' \
		'cuda-cudart-dev=11.7.*' \
		'cuda-libraries-dev=11.7.*' \
		'pytorch=1.13.1=py3.8_cuda11.7*' \
		'pytorch-cuda=11.7' \
		'pytorch-mutex=1.0=cuda' \
		'mkl=2023.*' \
		'mkl-include=2023.*'
}

ensure_cuda_dev_headers() {
	local prefix
	prefix="$(mamba_run python -c 'import sys; print(sys.prefix)')"
	if [[ -f "${prefix}/include/cusparse.h" ]]; then
		info "CUDA library headers present (${prefix}/include/cusparse.h)"
		return 0
	fi
	install_cuda_dev_stack
	[[ -f "${prefix}/include/cusparse.h" ]] ||
		die "cuda-libraries-dev installed but ${prefix}/include/cusparse.h is missing"
	info "CUDA library headers ready (${prefix}/include/cusparse.h)"
}

configure_cuda_11() {
	local candidate=""
	local shared="${HOME}/shared_software/cuda/cuda-11.7"

	if command -v nvcc >/dev/null 2>&1 && [[ "$(nvcc_major "$(command -v nvcc)")" == "11" ]]; then
		candidate="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
		export_cuda_home "${candidate}"
		assert_nvcc_11 "$(command -v nvcc)"
		ensure_cuda_dev_headers
		return 0
	fi

	if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" && "$(nvcc_major "${CUDA_HOME}/bin/nvcc")" == "11" ]]; then
		export_cuda_home "${CUDA_HOME}"
		assert_nvcc_11 "${CUDA_HOME}/bin/nvcc"
		ensure_cuda_dev_headers
		return 0
	fi

	if [[ -x "${shared}/bin/nvcc" ]]; then
		export_cuda_home "${shared}"
		assert_nvcc_11 "${shared}/bin/nvcc"
		ensure_cuda_dev_headers
		return 0
	fi

	install_cuda_dev_stack

	local prefix
	prefix="$(mamba_run python -c 'import sys; print(sys.prefix)')"
	[[ -x "${prefix}/bin/nvcc" ]] || die "cuda-nvcc installed but ${prefix}/bin/nvcc is missing"
	export_cuda_home "${prefix}"
	assert_nvcc_11 "${prefix}/bin/nvcc"
	ensure_cuda_dev_headers
}

gcc_release() {
	local bin="$1"
	"${bin}" -dumpfullversion 2>/dev/null || "${bin}" -dumpversion
}

assert_gcc_11() {
	local cxx="$1"
	local full major minor
	full="$(gcc_release "${cxx}")"
	major="${full%%.*}"
	if [[ "${full}" == *.* ]]; then
		minor="${full#*.}"
		minor="${minor%%.*}"
	else
		minor="0"
	fi
	if [[ "${major}" != "11" ]]; then
		die "need GCC 11.x (<=11.5) to compile openfold with CUDA 11.7, found ${full} (${cxx})"
	fi
	if [[ "${minor}" -gt 5 ]]; then
		die "GCC ${full} is greater than CUDA 11.7 max 11.5.0 (${cxx})"
	fi
}

resolve_conda_gcc() {
	local prefix="$1"
	local cc cxx
	cc="${prefix}/bin/x86_64-conda-linux-gnu-gcc"
	cxx="${prefix}/bin/x86_64-conda-linux-gnu-g++"
	if [[ -x "${cc}" && -x "${cxx}" ]]; then
		printf '%s\n%s\n' "${cc}" "${cxx}"
		return 0
	fi
	local matches=()
	shopt -s nullglob
	matches=("${prefix}/bin/"*-conda-*-g++)
	shopt -u nullglob
	[[ ${#matches[@]} -gt 0 && -x "${matches[0]}" ]] ||
		die "conda-forge GCC 11 g++ not found under ${prefix}/bin"
	cxx="${matches[0]}"
	cc="${cxx%g++}gcc"
	[[ -x "${cc}" ]] || die "conda-forge GCC 11 gcc not found (${cc})"
	printf '%s\n%s\n' "${cc}" "${cxx}"
}

configure_gcc_11() {
	info "Installing conda-forge GCC 11 into '${ENV_NAME}' for openfold (pytorch pin kept)"
	micromamba install -n "${ENV_NAME}" -y --channel-priority flexible \
		-c conda-forge -c pytorch -c nvidia \
		'gxx_linux-64=11.*' \
		'gcc_linux-64=11.*' \
		'pytorch=1.13.1=py3.8_cuda11.7*' \
		'pytorch-cuda=11.7' \
		'pytorch-mutex=1.0=cuda' \
		'mkl=2023.*' \
		'mkl-include=2023.*'

	local prefix cc cxx
	prefix="$(mamba_run python -c 'import sys; print(sys.prefix)')"
	{
		read -r cc
		read -r cxx
	} < <(resolve_conda_gcc "${prefix}")
	export CC="${cc}"
	export CXX="${cxx}"
	export CUDAHOSTCXX="${cxx}"
	assert_gcc_11 "${cxx}"
	info "Using CXX=${CXX} (GCC $(gcc_release "${cxx}"))"
}

write_torch_constraint() {
	CONSTRAINT_FILE="$(mktemp "${TMPDIR:-/tmp}/eval-torch-constraint.XXXXXX")"
	printf 'torch==%s\npytorch-lightning==%s\n' "${TORCH_PIN}" "${LIGHTNING_PIN}" >"${CONSTRAINT_FILE}"
}

assert_torch_pin() {
	local label="$1"
	info "Checking torch pin (${label})"
	mamba_run python - <<PY
import torch
import torch_geometric

assert torch.__version__.startswith("${TORCH_PIN}"), torch.__version__
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
cuda = str(torch.version.cuda or "")
assert cuda.startswith("11.7"), cuda
print("torch", torch.__version__, "cuda", cuda, "pyg", torch_geometric.__version__)
PY
}

install_pip_and_pyg() {
	info "Reinstalling PocketGen pip stack (micromamba clone omits pip packages)"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" \
		meeko==0.1.dev3 wandb scipy pdb2pqr vina==1.2.2 \
		fair-esm==2.0.0 \
		omegaconf==2.3.0 e3nn==0.5.1 einops==0.7.0 biopython==1.79 biotite \
		gdown \
		torch-geometric
	info "Installing AutoDockTools_py3 from GitHub"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" \
		"git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"
	info "Installing PyG CUDA wheels (cu117, binary only)"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" --only-binary=:all: \
		torch-scatter==2.1.1+pt113cu117 \
		torch-sparse==0.6.17+pt113cu117 \
		torch-cluster==1.6.1+pt113cu117 \
		torch-spline-conv==1.2.2+pt113cu117 \
		-f "${PYG_WHL_INDEX}"
}

install_genie_package() {
	info "Installing genie editable (no-deps) so torch stays ${TORCH_PIN}"
	mamba_run python -m pip install -e "${GENIE_DIR}" --no-deps
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" \
		tqdm numpy scipy wandb pandas tensorboard \
		"pytorch_lightning==${LIGHTNING_PIN}" \
		'urllib3==1.26.14' 'charset-normalizer==2.1.1'
}

install_proteinmpnn() {
	local dest="${GENIE_DIR}/packages/ProteinMPNN"
	mkdir -p "${GENIE_DIR}/packages"
	if [[ -d "${dest}" ]]; then
		info "ProteinMPNN already present at ${dest}"
		return 0
	fi
	info "Cloning ProteinMPNN"
	git clone https://github.com/dauparas/ProteinMPNN.git "${dest}"
}

install_esmfold_openfold() {
	info "Installing fair-esm[esmfold] extras with torch==${TORCH_PIN} constraint"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" "fair-esm[esmfold]"
	info "Installing dllogger"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" \
		"dllogger @ git+https://github.com/NVIDIA/dllogger.git"
	info "Installing ESMFold-era openfold (no build isolation)"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" --no-build-isolation \
		"openfold @ ${OPENFOLD_GIT}"
	info "Installing modelcif"
	mamba_run python -m pip install -c "${CONSTRAINT_FILE}" modelcif
}

install_tmscore() {
	local dest="${GENIE_DIR}/packages/TMscore"
	mkdir -p "${dest}"
	command -v g++ >/dev/null 2>&1 || die "g++ not found; needed to build TMscore"
	command -v wget >/dev/null 2>&1 || die "wget not found; needed to download TMscore"

	if [[ ! -x "${dest}/TMscore" ]]; then
		info "Building TMscore"
		wget -q -O "${dest}/TMscore.cpp" https://zhanggroup.org/TM-score/TMscore.cpp
		g++ -O3 -ffast-math -lm -o "${dest}/TMscore" "${dest}/TMscore.cpp"
		chmod +x "${dest}/TMscore"
	else
		info "TMscore already present at ${dest}/TMscore"
	fi

	if [[ ! -x "${dest}/TMalign" ]]; then
		info "Building TMalign"
		wget -q -O "${dest}/TMalign.cpp" https://zhanggroup.org/TM-align/TMalign.cpp
		g++ -O3 -ffast-math -lm -o "${dest}/TMalign" "${dest}/TMalign.cpp"
		chmod +x "${dest}/TMalign"
	else
		info "TMalign already present at ${dest}/TMalign"
	fi
}

smoke_test() {
	info "Import smoke test (no ESMFold weight download)"
	mamba_run python - <<'PY'
import esm
import openfold
import rdkit
import torch
import torch_geometric

assert torch.__version__.startswith("1.13.1"), torch.__version__
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(
    "smoke_ok",
    torch.__version__,
    torch.version.cuda,
    rdkit.__version__,
    esm.__version__,
)
PY
}

print_next_steps() {
	cat <<EOF

========================================================================
Genie eval env '${ENV_NAME}' is ready (clone of '${BASE_ENV}' + ESMFold).

Designability:
  micromamba activate ${ENV_NAME}
  python reproduction/eval_designability.py \\
    --sample_dir reproduction/outputs/crossdocked_sample

Sampling / Vina / PoseBusters can use '${BASE_ENV}' or '${ENV_NAME}'.
========================================================================
EOF
}

main() {
	info "PocketGen root: ${POCKETGEN_ROOT}"
	info "Genie dir: ${GENIE_DIR}"
	info "Base env: ${BASE_ENV}; eval env: ${ENV_NAME}"

	export MC_ENABLE_PROXY=0
	local monorepo_prepare="${POCKETGEN_ROOT}/../../shell_script/prepare.sh"
	if [[ -f "${monorepo_prepare}" ]]; then
		# shellcheck source=/dev/null
		source "${monorepo_prepare}"
	fi

	ensure_clone
	env_compatible "${ENV_NAME}" || die "clone of '${BASE_ENV}' is not Python 3.8 / torch ${TORCH_PIN}+cu117"
	configure_cuda_11
	write_torch_constraint
	install_pip_and_pyg
	install_genie_package
	assert_torch_pin "after genie"
	install_proteinmpnn
	configure_gcc_11
	install_esmfold_openfold
	assert_torch_pin "after openfold"
	install_tmscore
	smoke_test
	print_next_steps
}

main "$@"
