# Molecular Generative Model

This repository provides implementations of **molecular generative models** using **VAE** and **Diffusion Models**, with built-in pipelines for **training, sampling, and optimization**.  
It also includes scripts for **external dataset application** and **end-to-end pipelines**.

---

## 📂 Paper Data

The datasets used in this project can be found at the following links:

- [**ChEMBL**](https://chembl.gitbook.io/chembl-interface-documentation/downloads)  
- [**QM9**](https://drive.google.com/file/d/1JZ_Z5bjS0RsX_BRWtrplMN9vZpL78-T7/view?usp=drive_link)  
- [**GEOM-Drug**](https://dataverse.harvard.edu/file.xhtml?fileId=4360331&version=2.0)  
- [**ZINC**](https://drive.google.com/file/d/1N44fpvCKEqI3xorXH7Q9sOq2f4ylCUwz/view)  

---

## ⚡ One-File Quickstart Pipeline

We provide a **single executable script** (`run_pipeline.sh`) to run the full process (**training → sampling → optimization**).  

**Save as `run_pipeline.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail

# ========= Default Config =========
DATA="./data/ChEMBL.smi"       # Path to training dataset (.smi)
TARGET="2RMA"                  # Target ID for affinity optimization
REF_MOL="./data/reference.smi" # Reference molecule for similarity optimization
NUM_SAMPLES=1000               # Number of molecules to sample
RUN_NAME="$(date +%Y%m%d_%H%M%S)"
OUTDIR="./results/${RUN_NAME}"
LOGDIR="${OUTDIR}/logs"
# =================================

usage() {
  cat <<EOF
Usage: $0 [--data PATH] [--target ID] [--ref_mol PATH] [--num_samples N] [--run_name NAME]

Examples:
  $0 --data ./data/ChEMBL.smi --num_samples 1000
  $0 --data ./data/my_dataset.smi --target 3AF2 --ref_mol ./data/reference.smi --num_samples 500 --run_name myrun
EOF
  exit 0
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --data) DATA="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --ref_mol) REF_MOL="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; OUTDIR="./results/${RUN_NAME}"; LOGDIR="${OUTDIR}/logs"; shift 2 ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

mkdir -p "$OUTDIR" "$LOGDIR"

echo "=== CONFIG ==="
echo "DATA        = $DATA"
echo "TARGET      = $TARGET"
echo "REF_MOL     = $REF_MOL"
echo "NUM_SAMPLES = $NUM_SAMPLES"
echo "OUTDIR      = $OUTDIR"
echo "=============="

# 1) Train VAE
echo "[1/5] Training VAE..."
python train_vae.py --data "$DATA" 2>&1 | tee "${LOGDIR}/01_train_vae.log"

# 2) Train Diffusion
echo "[2/5] Training diffusion model..."
python train_diffusion.py --data "$DATA" 2>&1 | tee "${LOGDIR}/02_train_diffusion.log"

# 3) Sampling
echo "[3/5] Sampling ${NUM_SAMPLES} molecules..."
python sample.py --model diffusion --num_samples "$NUM_SAMPLES" --out "${OUTDIR}/generated.smi" 2>&1 | tee "${LOGDIR}/03_sample.log"

# 4) Optimize for binding affinity
echo "[4/5] Affinity optimization (target=${TARGET})..."
# If optimize_affinity.py supports --out, keep it; otherwise it will save to its default location.
if python optimize_affinity.py --help 2>/dev/null | grep -q -- "--out"; then
  python optimize_affinity.py --model diffusion --target "$TARGET" --out "${OUTDIR}/optimized_affinity.smi" 2>&1 | tee "${LOGDIR}/04_opt_affinity.log"
else
  python optimize_affinity.py --model diffusion --target "$TARGET" 2>&1 | tee "${LOGDIR}/04_opt_affinity.log"
fi

# 5) Optimize for similarity
echo "[5/5] Similarity optimization (ref_mol=${REF_MOL})..."
if python optimize_similarity.py --help 2>/dev/null | grep -q -- "--out"; then
  python optimize_similarity.py --model diffusion --ref_mol "$REF_MOL" --out "${OUTDIR}/optimized_similarity.smi" 2>&1 | tee "${LOGDIR}/05_opt_similarity.log"
else
  python optimize_similarity.py --model diffusion --ref_mol "$REF_MOL" 2>&1 | tee "${LOGDIR}/05_opt_similarity.log"
fi

echo "✅ Done. Outputs & logs saved to: ${OUTDIR}"
```

**Run example:**

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh \
  --data ./data/ChEMBL.smi \
  --target 2RMA \
  --ref_mol ./data/reference.smi \
  --num_samples 1000 \
  --run_name demo_run
```

---

## 🏋️ Training

### Train the VAE
```bash
python train_vae.py --data ./data/ChEMBL.smi
```

### Train the Diffusion model
```bash
python train_diffusion.py --data ./data/ChEMBL.smi
```

---

## 💡 Sampling

Generate molecules using trained models:

```bash
python sample.py --model diffusion --num_samples 1000 --out ./results/generated.smi
```

---

## ⚙️ Optimization

### Optimize for binding affinity
```bash
python optimize_affinity.py --model diffusion --target 2RMA
```

### Optimize for molecular similarity
```bash
python optimize_similarity.py --model diffusion --ref_mol ./data/reference.smi
```

---

## 🔧 External Dataset Application

To apply the framework on a new dataset, prepare a `.smi` file and specify it via the `--data` argument.

### Train on a new dataset
```bash
python train_diffusion.py --data ./data/my_dataset.smi
```

### Sample molecules from the new model
```bash
python sample.py --model diffusion --num_samples 500 --out ./results/my_generated.smi
```

---

## 📌 Example Pipelines (Common Use Cases)

### End-to-end training and sampling
```bash
python train_vae.py --data ./data/ChEMBL.smi
python train_diffusion.py --data ./data/ChEMBL.smi
python sample.py --model diffusion --num_samples 1000 --out ./results/generated.smi
```

### Affinity optimization on an external dataset
```bash
python train_diffusion.py --data ./data/my_dataset.smi
python optimize_affinity.py --model diffusion --target 3AF2
```

### Similarity-driven molecular generation
```bash
python sample.py --model diffusion --num_samples 200 --out ./results/candidates.smi
python optimize_similarity.py --model diffusion --ref_mol ./data/reference.smi
```

---

## 📖 Citation

If you find this code useful, please cite our paper.
```
[XXX]
```
