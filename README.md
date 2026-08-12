# KAT-Net

<p align="center">
  <img src="katnet_overview.png" alt="KAT-Net model overview" width="900">
</p>

<p align="center">
  <b>KAT-Net: a hyperbolic routing framework for joint lysine acylation site prediction with shared sequence–structure representations</b>
</p>

KAT-Net is a multi-task framework for the joint prediction of lysine crotonylation (Kcr), succinylation (Ksucc), and acetylation (Kac) sites. The framework integrates lysine-centred sequence representations, AlphaFold2-derived structural information, PAE-aware graph modelling, and task-driven hyperbolic routing to exploit shared information among related lysine acylations while preserving task-specific evidence.

This repository provides the final training and testing code, structural feature extraction workflow, cross-task leakage audit code, original and cleaned datasets, external Kcr evaluation datasets, data used for the independent Kac/Ksucc functional analysis, validation-derived decision thresholds, and pretrained five-fold checkpoints distributed through GitHub Releases.

---

## 1. Main features

- **Joint Kcr/Ksucc/Kac prediction** using a unified multi-task framework.
- **ESM-2 sequence representation** combined with lysine-centred physicochemical and positional priors.
- **Center-Anchored Bi-Mamba** for directional lysine-centred sequence modelling.
- **PAE-aware EGNN** for confidence-regulated structural message passing.
- **Task-driven hyperbolic routing** for modification-specific structural retrieval.
- **Masked asymmetric focal loss** for unavailable task labels and task-dependent class imbalance.
- **Five-fold ensemble inference** using thresholds determined exclusively from development-fold validation predictions.
- **Cross-task leakage control** using centre-normalized lysine windows before final model training.

---

## 2. Repository structure

```text
KAT-Net/
├── README.md
├── katnet_overview.png
│
├── train.py
├── test.py
├── extract_structure_features.py
│
├── checkpoints/
│   └── README.md
│
├── audit/
│   ├── cross_task_leakage_audit.py
│   └── validation_thresholds.json
│
└── datasets/
    ├── Kcr/
    │   └── original Kcr benchmark files
    │
    ├── Ksucc/
    │   └── original Ksucc benchmark files
    │
    ├── Kac/
    │   └── original Kac benchmark files
    │
    ├── data_cleaned/
    │   └── final cleaned multi-task dataset
    │
    ├── go/
    │   └── four files used for the independent Kac/Ksucc functional analysis
    │
    └── cross-species/
        ├── animal/
        ├── human/
        ├── microorganism/
        └── plant/
```

The five pretrained fold-specific model checkpoints are distributed through the GitHub **v1.0.0 Release** rather than stored directly in the normal Git history because of their file sizes.

---

## 3. Datasets

### 3.1 Original benchmark datasets

The original source benchmark files are retained under:

```text
datasets/Kcr/
datasets/Ksucc/
datasets/Kac/
```

KAT-Net was developed from three publicly available lysine-acylation benchmarks.

| Task | Source benchmark | Development positive | Development negative | Independent-test positive | Independent-test negative |
|---|---|---:|---:|---:|---:|
| Kcr | DeepCap-Kcr | 6,803 | 5,381 | 2,989 | 2,989 |
| Ksucc | LMSuccSite | 4,576 | 4,374 | 253 | 2,973 |
| Kac | TransPTM NHAC | 761 | 4,550 | 150 | 942 |

The original task-specific training and independent-test assignments were retained as the initial partitions before cross-task decontamination.

---

### 3.2 Cleaned multi-task dataset

The final cleaned dataset used in the revised experiments is stored under:

```text
datasets/data_cleaned/
```

The multi-task labels follow:

```text
 1 = positive annotation for the corresponding PTM
 0 = source-defined negative annotation for the corresponding PTM
-1 = unavailable annotation for the corresponding PTM
```

A label of `-1` is excluded from the corresponding task loss and is not interpreted as a negative label.

Identical central 31-residue lysine-centred windows were assigned to the same cross-validation fold.

---

### 3.3 Cross-task redundancy control

Because the three original benchmarks were constructed independently, a dedicated cross-task redundancy audit was performed before final multi-task training.

The audit covers the six off-diagonal comparisons between the independent test set of one PTM and the training data of the other two PTMs.

The final window-level criterion is:

```text
centre-normalized window : 31 residues
sequence identity        : >= 0.90
coverage                 : >= 0.80
```

Training samples showing exact normalized-window conflicts, known same-protein or same-site relationships where identifiers were recoverable, or near-window conflicts were removed.

All independent test sets were retained unchanged.

A total of **781 unique training samples** were removed:

| Task | Positive removed | Negative removed | Total removed |
|---|---:|---:|---:|
| Kcr | 172 | 144 | 316 |
| Ksucc | 164 | 143 | 307 |
| Kac | 33 | 125 | 158 |
| **Total** | **369** | **412** | **781** |

A post-filter audit found no remaining strict or near-window conflicts under the predefined criterion.

Because traceable UniProt/site identifiers were unavailable for part of the original processed Kcr data, complete protein-level homology independence could not be established uniformly for Kcr-related comparisons. This limitation is explicitly retained in the manuscript and audit output.

---

### 3.4 External Kcr datasets

External Kcr datasets are stored under:

```text
datasets/cross-species/
├── animal/
├── human/
├── microorganism/
└── plant/
```

The external evaluation includes:

```text
Rice
Papaya
Zebrafish
Toxoplasma gondii
Lung cancer-related proteins
Independent human benchmark
```

Rice, papaya, and the independent human benchmark contain both positive and negative samples and therefore support full binary-classification evaluation.

Zebrafish, *Toxoplasma gondii*, and lung cancer-related datasets contain experimentally reported positive sites only and are therefore interpreted only as known-site recognition datasets.

None of these external datasets was used for model fitting, hyperparameter selection, checkpoint selection, or decision-threshold determination.

---

### 3.5 Independent Kac/Ksucc functional-analysis data

The following directory contains the four processed files used for the independent functional analysis based on the PXD049146 mouse lung proteome, acetylome, and succinylome study:

```text
datasets/go/
```

The analysis used experimentally identified Kac and Ksucc sites as independent KAT-Net prediction inputs.

Proteins quantified in the same PXD049146 proteomic study were used as the GO enrichment background.

Kcr was not evaluated because crotonylation was not profiled in PXD049146.

---

## 4. Model architecture

### 4.1 Centre-aware sequence branch

KAT-Net uses the pretrained protein language model:

```text
facebook/esm2_t33_650M_UR50D
```

The first six Transformer encoder layers are retained.

The token embedding layer is frozen, whereas the retained six encoder layers remain trainable during final optimisation.

Three trainable prompt representations are appended to each side of the ESM-2 residue representation.

Centre-Specific Representation Enhancement (CSRE) incorporates residue-level physicochemical properties and a lysine-centred Gaussian positional prior.

The combined sequence representation is projected to a hidden dimension of 64 and processed by a two-layer Center-Anchored Bi-Mamba encoder.

---

### 4.2 Structural branch

Protein structures are generated using ColabFold based on the AlphaFold2 framework.

The structural-processing pipeline provides:

```text
coords : residue C-alpha coordinates
plddt  : per-residue predicted confidence
pae    : predicted aligned error
sasa   : legacy NPZ key storing DSSP-derived relative solvent accessibility (RSA)
ss     : three-state secondary structure
disto  : 64-bin distogram probabilities
```

The NPZ key name `sasa` is retained only for compatibility with the existing KAT-Net training and testing code.

The stored quantity is **DSSP-derived relative solvent accessibility (RSA)** rather than FreeSASA-derived absolute SASA.

An eight-dimensional Laplacian positional encoding is calculated from a symmetrized seven-nearest-neighbour graph constructed from residue coordinates.

The node features are processed by a two-layer PAE-aware EGNN with hidden dimension 64.

PAE is used as an edge-level confidence regulator so that uncertain residue relationships contribute less strongly to structural propagation.

---

### 4.3 Task-driven hyperbolic routing

For each task, the central lysine sequence representation is combined with a task-specific learnable anchor.

Structural representations are projected into the Poincaré ball.

Centre-to-residue hyperbolic distance is then used as a topology bias for task-conditioned structural retrieval.

The task-specific structural context is finally fused with the sequence query before prediction.

---

### 4.4 Training objective

KAT-Net uses masked asymmetric focal loss with unavailable-label masking and learnable task uncertainty weighting.

The final reference settings are:

```text
gamma_pos = 0
gamma_neg = 2
margin    = 0.05

hyperbolic curvature = 2.0
topology bias strength = 1.0

log-var clip = [-1.5, 2.0]
```

Unavailable task labels are excluded from the corresponding task loss.

---

## 5. Pretrained checkpoints

The five pretrained fold-specific checkpoints used in the manuscript are distributed through the GitHub **v1.0.0 Release**.

Download:

```text
katnet_3ptm_fold_1.pth
katnet_3ptm_fold_2.pth
katnet_3ptm_fold_3.pth
katnet_3ptm_fold_4.pth
katnet_3ptm_fold_5.pth
```

After downloading, place them under:

```text
checkpoints/
├── README.md
├── katnet_3ptm_fold_1.pth
├── katnet_3ptm_fold_2.pth
├── katnet_3ptm_fold_3.pth
├── katnet_3ptm_fold_4.pth
└── katnet_3ptm_fold_5.pth
```

See:

```text
checkpoints/README.md
```

for additional instructions.

---

## 6. Validation-derived decision thresholds

The final validation-derived thresholds are stored in:

```text
audit/validation_thresholds.json
```

They were determined exclusively from pooled out-of-fold validation predictions and were frozen before evaluation of any independent or external test dataset.

| Task | Threshold |
|---|---:|
| Kcr | 0.6071 |
| Ksucc | 0.5561 |
| Kac | 0.4135 |

Independent and external test labels were not used for model fitting, hyperparameter selection, checkpoint selection, or decision-threshold determination.

---

## 7. Environment

Model training/testing and structural feature generation were performed in separate environments.

### 7.1 Model training and testing environment

The final model environment used:

| Package | Version |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.43.3 |
| torch-geometric | 2.8.0.post1 |
| NumPy | 1.26.4 |
| Pandas | 2.1.4 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| Biopython | 1.81 |
| einops | 0.8.1 |
| tqdm | 4.65.0 |

A basic environment can be created using:

```bash
conda create -n katnet python=3.10
conda activate katnet
```

Install the CUDA 12.1 PyTorch build:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the main Python dependencies:

```bash
pip install \
  transformers==4.43.3 \
  torch-geometric==2.8.0.post1 \
  numpy==1.26.4 \
  pandas==2.1.4 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  biopython==1.81 \
  einops==0.8.1 \
  tqdm==4.65.0
```

If CUDA 12.1 is not available, install the PyTorch build compatible with the local CUDA environment.

---

### 7.2 Structural feature extraction environment

Structural features were generated in a separate environment using:

| Package | Version |
|---|---|
| Python | 3.9.23 |
| ColabFold | 1.5.5 |
| alphafold-colabfold | 2.3.6 |
| Biopython | 1.82 |
| DSSP | 3.0.0 |
| JAX | 0.4.30 |
| jaxlib | 0.4.30 |
| NumPy | 1.26.4 |
| Pandas | 1.5.3 |
| SciPy | 1.13.1 |

Create the environment:

```bash
conda create -n katnet-structure python=3.9
conda activate katnet-structure
```

Install DSSP:

```bash
conda install -c salilab dssp=3.0.0
```

Install the main structural-processing dependencies:

```bash
pip install \
  colabfold==1.5.5 \
  alphafold-colabfold==2.3.6 \
  biopython==1.82 \
  numpy==1.26.4 \
  pandas==1.5.3 \
  scipy==1.13.1
```

JAX and the CUDA-specific ColabFold dependencies should be installed according to the CUDA configuration of the target machine.

---

## 8. Structural feature generation

The final structural workflow uses ColabFold/AlphaFold2 predictions and DSSP-derived residue features.

The structure-prediction configuration used in the experiments was:

```text
model type       : alphafold2_ptm
MSA mode         : single_sequence
number of models : 1
recycles         : 3
random seed      : 42
relaxation       : disabled
save-all outputs : enabled
```

Saving the full ColabFold outputs is important because PAE and the 64-bin distogram cannot be reconstructed from PDB coordinates alone.

Run the structural feature extraction script according to its command-line interface:

```bash
python extract_structure_features.py --help
```

Example:

```bash
python extract_structure_features.py \
  --input <input_file> \
  --output-dir <output_directory>
```

Precomputed structural NPZ files are not distributed in the normal Git repository because they can be regenerated using the released structural feature extraction workflow.

---

## 9. Training

Final training uses predefined five-fold partitions.

Reference settings:

```text
optimizer     : AdamW
batch size    : 64
learning rate : 1e-4
max epochs    : 70
early stopping patience : 15
random seed   : 42
number of folds : 5

checkpoint selection:
validation Macro-MCC
```

The final cleaned multi-task data are stored under:

```text
datasets/data_cleaned/
```

Run:

```bash
python train.py
```

Before training, configure the local paths required by the script for:

```text
cleaned dataset
structural feature directory
ESM-2 model/cache
checkpoint output directory
```

---

## 10. Testing

Download the five pretrained model checkpoints from the GitHub **v1.0.0 Release** and place them in:

```text
checkpoints/
```

The frozen validation-derived thresholds are provided in:

```text
audit/validation_thresholds.json
```

Run:

```bash
python test.py
```

Final ensemble probabilities are obtained by averaging predictions from the five fold-specific checkpoints.

The frozen task-specific thresholds are then applied to the averaged probabilities.

For standard binary datasets, the evaluation includes metrics such as:

```text
ACC
MCC
AUROC
AUPRC
F1-score
Precision
Recall / Sensitivity
Specificity
```

For positive-only external datasets, only known-site recognition or positive-site recall should be interpreted.

AUROC, MCC, specificity, and other two-class metrics are not applicable to positive-only datasets.

---

## 11. Cross-task leakage audit

The final cross-task leakage audit script is stored under:

```text
audit/cross_task_leakage_audit.py
```

The released audit code implements the protocol reported in the revised manuscript and Supplementary Materials:

```text
normalized lysine window : 31 residues
near-window identity     : >= 0.90
near-window coverage     : >= 0.80

unique training removals : 781
independent test sets modified : no
remaining window conflicts after filtering : 0
```

All input paths are supplied at runtime and no server-specific absolute paths are required.

Example:

```bash
python audit/cross_task_leakage_audit.py \
  --kcr-train-pos <Kcr_train_positive> \
  --kcr-train-neg <Kcr_train_negative> \
  --kcr-test-pos <Kcr_test_positive> \
  --kcr-test-neg <Kcr_test_negative> \
  --ksucc-train-pos <Ksucc_train_positive> \
  --ksucc-train-neg <Ksucc_train_negative> \
  --ksucc-test-pos <Ksucc_test_positive> \
  --ksucc-test-neg <Ksucc_test_negative> \
  --kac-csv <Kac_csv> \
  --output-dir <audit_output>
```

---

## 12. Reproducibility notes

- The independent-test assignments were not modified by the cross-task filtering procedure.
- External datasets were not used for model fitting or threshold selection.
- Decision thresholds were derived only from pooled development-fold validation predictions.
- Final predictions use the mean probability from five independently trained fold-specific checkpoints.
- The NPZ field `sasa` is retained only for backward compatibility and stores DSSP-derived RSA.
- Precomputed AlphaFold2 structural features are required for model inference unless they are regenerated using the released structural workflow.
- Reported inference time using precomputed structural features does not include AlphaFold2/ColabFold structure-generation time.
- Predictions involving low-pLDDT residues or high-PAE relationships should be interpreted cautiously because confidence-aware processing cannot correct inaccurate predicted structures.
- Positive-only external datasets should not be interpreted as complete binary-classification benchmarks.

---

## 13. Citation

If you use KAT-Net, its released datasets, or pretrained checkpoints, please cite:

```text
Wu H, Lin Y, Zhu L, Yang S.
KAT-Net: a hyperbolic routing framework for joint lysine acylation site prediction
with shared sequence–structure representations.
```

The final journal citation and DOI will be added after publication.

---

## 14. Data availability

The source code, released datasets, audit resources, and reproducibility files are provided in this repository.

The five pretrained model checkpoints are distributed through the GitHub **v1.0.0 Release**.

Original public datasets remain attributable to their corresponding source studies.

---

## 15. Contact

For questions regarding KAT-Net, the datasets, or reproducibility, please open an issue in this repository.

Corresponding authors:

```text
Lun Zhu: zl@cczu.edu.cn
Sen Yang: ys@cczu.edu.cn
```
