# PocketGen setup (micromamba + CUDA)

Validated path from clone to training on Linux with an NVIDIA GPU.
Environment name: `pocketgen` (Python 3.8, PyTorch **1.13.1+cu117**).

**Automated install:** from the repo root run:

```bash
bash prepare.sh
# optional: bash prepare.sh --process-data
# optional: bash prepare.sh --skip-esm --env-name pocketgen
```

`prepare.sh` creates/reuses the env, installs CUDA PyTorch + deps + PyG cu117 wheels, prefetches ESM weights, and reports data status. Details below remain the human-readable reference.

> Do **not** use `pocketgen.yaml` as-is. It is a frozen machine export with pinned build hashes and is usually unsatisfiable.

Fish helpers used below (from `~/.config/fish/config.fish`):

- `cuda <gpu_ids> <cmd...>` — set `CUDA_VISIBLE_DEVICES` and run a command
- `proxy_on` / `proxy_off` — optional HTTP proxy
- `micromamba activate pocketgen`

No local CUDA toolkit install is required: PyTorch ships `pytorch-cuda=11.7` (`cu117`). A system toolkit (`use_cuda`) is only useful if you compile CUDA extensions yourself.

---

## 0. Prerequisites

- NVIDIA driver that can run CUDA 11.7 runtimes (newer drivers are fine)
- Enough disk space (env + data + ESM weights: tens of GB)

```fish
nvidia-smi
```

---

## 1. Get the repository

```fish
git clone <this-repo-url> PocketGen
cd PocketGen
```

---

## 2. Create the conda/micromamba environment

```fish
micromamba create -n pocketgen python=3.8 pip -y
micromamba activate pocketgen
```

---

## 3. Install CUDA PyTorch (critical)

Install a **CUDA** build and pin `pytorch-mutex=cuda`.  
Avoid letting later conda installs silently replace it with a CPU build.

```fish
micromamba install -y --channel-priority flexible \
  -c pytorch -c nvidia -c conda-forge \
  'pytorch=1.13.1=py3.8_cuda11.7*' \
  'pytorch-cuda=11.7' \
  'pytorch-mutex=1.0=cuda'
```

Verify:

```fish
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expect something like: `1.13.1 True` and your GPU name.

---

## 4. Install scientific / chemoinformatics conda packages

Install these **after** CUDA PyTorch. If the solver tries to replace `pytorch` with a CPU build, abort and re-pin step 3.

```fish
micromamba install -y -c conda-forge \
  rdkit openbabel tensorboard pyyaml easydict python-lmdb \
  openmm pdbfixer flask \
  numpy swig boost-cpp sphinx sphinx_rtd_theme
```

Re-check CUDA after this step:

```fish
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it becomes `False`, re-run step 3.

---

## 5. Install PyG (CUDA wheels for torch 1.13 + cu117)

Do **not** use the CPU wheel index. Prefer the exact cu117 wheels (avoids compiling against a mismatched system CUDA):

```fish
pip install torch-geometric

set -l BASE https://data.pyg.org/whl/torch-1.13.0%2Bcu117
pip install \
  $BASE/torch_scatter-2.1.1%2Bpt113cu117-cp38-cp38-linux_x86_64.whl \
  $BASE/torch_sparse-0.6.17%2Bpt113cu117-cp38-cp38-linux_x86_64.whl \
  $BASE/torch_cluster-1.6.1%2Bpt113cu117-cp38-cp38-linux_x86_64.whl \
  $BASE/torch_spline_conv-1.2.2%2Bpt113cu117-cp38-cp38-linux_x86_64.whl
```

Equivalent one-liner style:

```fish
pip install torch-scatter==2.1.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 \
  torch-cluster==1.6.1+pt113cu117 torch-spline-conv==1.2.2+pt113cu117 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --only-binary=:all:
```

Verify:

```fish
python -c "import torch, torch_geometric, torch_scatter; from torch_scatter import scatter_sum; x=torch.randn(4,8,device='cuda'); i=torch.tensor([0,0,1,1],device='cuda'); print(torch_scatter.__version__, scatter_sum(x,i,dim=0).shape)"
```

---

## 6. Pip packages required by train / generate

Upstream README pip list is incomplete. Also install:

```fish
pip install meeko==0.1.dev3 wandb scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

pip install fair-esm==2.0.0
pip install omegaconf==2.3.0 e3nn==0.5.1 einops==0.7.0 biopython==1.79 biotite
pip install gdown   # optional, for Google Drive downloads
```

---

## 7. Dataset (CrossDocked, for training)

Put files under `./data`:

1. [crossdocked_v1.1_rmsd1.0.tar.gz](https://drive.google.com/file/d/1U0ZgITApL7EClcQiiVK_OevAV_H20L4d/view?usp=sharing)
2. [split_by_name.pt](https://drive.google.com/file/d/1UVJmLvx-kcorMyDDR_LPCqR8dFPuoRtI/view?usp=sharing)

Then:

```fish
# extract the tar into ./data if needed
python data_preparation/extract_pockets.py
python data_preparation/split_pl_dataset.py
```

Or use the processed LMDB + split from [Zenodo](https://zenodo.org/records/10125312) (needs `*.lmdb` and `*_split.pt` / `split.pt` as expected by the config).

Config used by training: `configs/train_model.yml`  
(`dataset.path` / `dataset.split` point under `./data/crossdocked_v1.1_rmsd1.0_pocket10`).

---

## 8. ESM2 weights (auto-download on first run)

Training/generation load `esm2_t33_650M_UR50D`. Prefetch if desired:

```fish
mkdir -p ~/.cache/torch/hub/checkpoints
wget -c -O ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget -c -O ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt \
  https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

---

## 9. Train

`train_recycle.py` defaults to `--device cuda:0` and uses Weights & Biases.

```fish
micromamba activate pocketgen
# optional: wandb login

cuda 0 python train_recycle.py --config ./configs/train_model.yml
```

Binding MOAD:

```fish
cuda 0 python train_recycle.py --config ./configs/train_model_moad.yml
```

---

## 10. Generate (optional; needs checkpoint)

Download [checkpoint.pt](https://drive.google.com/file/d/1cuvdiu3bXyni71A2hoeZSWT1NOsNfeD_/view?usp=sharing) into `./checkpoints/checkpoint.pt`:

```fish
mkdir -p checkpoints tmp
# if Google Drive is slow/blocked, enable proxy first: proxy_on
gdown --fuzzy 'https://drive.google.com/file/d/1cuvdiu3bXyni71A2hoeZSWT1NOsNfeD_/view?usp=sharing' \
  -O checkpoints/checkpoint.pt

cuda 0 python generate_new.py
```

---

## Pitfalls learned on this machine

1. **`pocketgen.yaml` fails to solve** — pinned Anaconda builds / missing channels / strict priority.
2. **README `pytorch-cuda=11.6` + `pyg` can resolve to CPU torch** (`pytorch-mutex=cpu`). Always verify `torch.cuda.is_available()`.
3. **Later `micromamba install openmm ...` can uninstall `pytorch-cuda` / `pyg`**. Re-pin CUDA torch (step 3) and reinstall PyG CUDA wheels (step 5) if that happens.
4. **Building PyG extensions from source against system CUDA 12/13 fails** with torch cu117. Use the prebuilt `+pt113cu117` wheels.
5. **Disk full** causes `libmamba Write failed` during extract (e.g. `libboost-headers`). Free space / `micromamba clean -a` first.
6. Generation needs `./checkpoints/checkpoint.pt` and a `tmp/` directory under the run folder.

---

## Quick sanity checklist

```fish
micromamba activate pocketgen
python -c "import torch; assert torch.cuda.is_available()"
python -c "import esm, torch_geometric, torch_scatter, omegaconf, e3nn, Bio, openbabel, openmm"
ls data/crossdocked_v1.1_rmsd1.0_pocket10/split.pt
cuda 0 python train_recycle.py --config ./configs/train_model.yml
```
