# KAT-Net

<p align="center">
  <img src="katnet_overview.png" alt="KAT-Net framework" width="900">
</p>

<p align="center">
  <b>KAT-Net: a hyperbolic routing framework for joint lysine acylation site prediction with shared sequence–structure representations</b>
</p>

KAT-Net is a multi-task deep learning framework for the joint prediction of three lysine acylation modifications:

* lysine crotonylation (**Kcr**)
* lysine succinylation (**Ksucc**)
* lysine acetylation (**Kac**)

The model integrates protein language-model representations, lysine-centred sequence information, AlphaFold2-derived structural information, PAE-aware graph modelling, and task-driven hyperbolic routing.

The repository provides the final training and testing code, structural feature-generation workflow, cross-task redundancy audit, cleaned datasets, external evaluation datasets, validation-derived thresholds, and pretrained five-fold model checkpoints.

---

## 1. Overview

KAT-Net contains four major components:

1. **Centre-aware sequence representation**

   * ESM-2 protein language model
   * representation-level soft prompts
   * Centre-Specific Representation Enhancement (CSRE)
   * Center-Anchored Bi-Mamba

2. **PAE-aware structural representation**

   * AlphaFold2/ColabFold structural predictions
   * residue coordinates
   * pLDDT
   * PAE
   * DSSP-derived relative solvent accessibility (**RSA**)
   * three-state secondary structure
   * 64-bin distogram
   * Laplacian positional encoding
   * PAE-aware EGNN

3. **Task-driven hyperbolic routing**

   * task-specific learnable anchors
   * Poincaré-ball projection
   * hyperbolic centre-to-residue distance
   * task-conditioned structural retrieval

4. **Masked multi-task learning**

   * masked asymmetric focal loss
   * unavailable task labels excluded from the corresponding loss
   * task uncertainty weighting

---

## 2. Repository structure

```text
KAT-Net/
├── README.md
├── katnet_overview.png
│
├── train.py
├── test.py
├── prepare_structure_features.py
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
    │   └── processed files for the independent Kac/Ksucc functional analysis
    │
    └── cross-species/
        ├── animal/
        ├── human/
        ├── microorganism/
        └── plant/
```

Large pretrained model files are distributed through **GitHub Releases** rather than stored directly in the normal Git history.

---

## 3. Development datasets

KAT-Net was developed using three publicly available lysine-acylation benchmarks.

| Task  | Source benchmark | Development positive | Development negative | Independent-test positive | Independent-test negative |
| ----- | ---------------- | -------------------: | -------------------: | ------------------------: | ------------------------: |
| Kcr   | DeepCap-Kcr      |                6,803 |                5,381 |                     2,989 |                     2,989 |
| Ksucc | LMSuccSite       |                4,576 |                4,374 |                       253 |                     2,973 |
| Kac   | TransPTM NHAC    |                  761 |                4,550 |                       150 |                       942 |

The original task-specific independent-test sets were retained during the final data-cleaning procedure.

The original source datasets are provided under:

```text
datasets/Kcr/
datasets/Ksucc/
datasets/Kac/
```

The final cleaned multi-task dataset is provided under:

```text
datasets/data_cleaned/
```

---

## 4. Multi-task labels

The multi-task label convention is:

```text
 1 = positive annotation for the corresponding PTM
 0 = source-defined negative annotation for the corresponding PTM
-1 = annotation unavailable for the corresponding PTM
```

An unavailable label (`-1`) is **masked from the corresponding task loss** and is not treated as a negative annotation.

Identical central 31-residue lysine-centred windows are assigned to the same cross-validation fold.

---

## 5. Cross-task redundancy control

Because the three source benchmarks were originally constructed independently, an additional cross-task redundancy audit was performed before final multi-task training.

The audit covers the six off-diagonal comparisons:

```text
Kcr test   vs Ksucc training
Kcr test   vs Kac training

Ksucc test vs Kcr training
Ksucc test vs Kac training

Kac test   vs Kcr training
Kac test   vs Ksucc training
```

All sequences are aligned relative to the candidate lysine and represented by a centre-normalized 31-residue window.

The final near-window criterion is:

```text
window length      : 31 residues
sequence identity  : >= 0.90
coverage           : >= 0.80
```

Training samples are flagged when they show:

```text
exact centre-normalized-window overlap
known same-parent-protein relationship where identifiers are available
known same-protein/same-lysine-site relationship where identifiers are available
near-window similarity satisfying the predefined identity/coverage criterion
```

Only conflicting training/development samples are removed.

**Independent test sets remain unchanged.**

After deduplication across the six cross-task routes, **781 unique training samples** were removed:

| Task      | Positive removed | Negative removed | Total removed |
| --------- | ---------------: | ---------------: | ------------: |
| Kcr       |              172 |              144 |           316 |
| Ksucc     |              164 |              143 |           307 |
| Kac       |               33 |              125 |           158 |
| **Total** |          **369** |          **412** |       **781** |

A complete post-filter audit found no remaining strict or near-window conflicts under the predefined criterion.

Because complete traceable protein/site identifiers were unavailable for part of the original processed Kcr data, uniform protein-level homology independence cannot be established for all Kcr-related comparisons. The released audit therefore explicitly reports this limitation.

The reproducibility script is provided at:

```text
audit/cross_task_leakage_audit.py
```

---

## 6. Sequence branch

### 6.1 ESM-2

KAT-Net uses:

```text
facebook/esm2_t33_650M_UR50D
```

The first six Transformer encoder layers are retained.

In the final configuration:

```text
ESM-2 token embedding layer : frozen
retained Transformer layers : trainable
number of retained layers   : 6
```

---

### 6.2 Soft Prompt Adapter

Three trainable prompt representations are appended to each side of the residue-level ESM-2 representation:

```text
[P_N ; H_ESM ; P_C]
```

The prompts are introduced at the representation level rather than being re-encoded as additional amino-acid tokens.

---

### 6.3 Centre-Specific Representation Enhancement

CSRE combines:

```text
4-dimensional physicochemical residue properties
lysine-centred Gaussian positional weighting
learnable relative-position embedding
```

The Gaussian positional prior uses:

```text
sigma = 3.0
```

to emphasize residues close to the candidate lysine while preserving broader sequence context.

---

### 6.4 Center-Anchored Bi-Mamba

The combined sequence representation is projected to:

```text
hidden dimension = 64
```

and processed by:

```text
2 bidirectional Mamba layers
```

Forward and reverse directional representations are combined to produce the final centre-aware sequence representation.

---

## 7. Structural feature generation

Protein structures are predicted using **ColabFold based on AlphaFold2**.

The structural preprocessing workflow is implemented in:

```text
prepare_structure_features.py
```

The final structural representation contains:

```text
coords : residue C-alpha coordinates
plddt  : per-residue AlphaFold2 confidence
pae    : predicted aligned error
rsa    : DSSP-derived relative solvent accessibility
ss     : three-state secondary structure
disto  : 64-bin distogram probabilities
```

The solvent-accessibility feature used throughout the released implementation is **relative solvent accessibility (RSA)** obtained from DSSP.

The same terminology and feature name are used consistently in:

```text
paper
model figure
README
structural feature-generation code
NPZ structural files
training code
testing code
```

---

## 8. ColabFold configuration

The structural prediction workflow used the following configuration:

```text
model type       : alphafold2_ptm
MSA mode         : single_sequence
number of models : 1
recycles         : 3
random seed      : 42
relaxation       : disabled
save-all outputs : enabled
```

Saving the complete ColabFold output is required because PAE and distogram information cannot be reconstructed from PDB coordinates alone.

The structural workflow subsequently uses DSSP to derive:

```text
relative solvent accessibility (RSA)
three-state secondary structure
```

---

## 9. Generate structural features

The structural script requires a CSV containing at least:

```text
Core_Key
Dynamic_Sequence
Center_K_Index
```

Basic usage:

```bash
python prepare_structure_features.py \
  --csv-path <input.csv> \
  --work-dir <structure_work_directory> \
  --output-dir <structure_feature_directory>
```

The complete pipeline can be executed with:

```bash
python prepare_structure_features.py \
  --stage all \
  --csv-path <input.csv> \
  --work-dir <structure_work_directory> \
  --output-dir <structure_feature_directory>
```

Individual stages are also supported:

```text
prepare
run
convert
check
all
```

Each generated NPZ file contains:

```text
coords
plddt
rsa
ss
pae
disto
```

---

## 10. PAE-aware structural encoder

The structural node representation combines:

```text
pLDDT
RSA
three-state secondary structure
8-dimensional Laplacian positional encoding
```

A symmetrized seven-nearest-neighbour coordinate graph is used to construct the Laplacian positional encoding.

The structural encoder contains:

```text
2 PAE-aware EGNN layers
hidden dimension = 64
```

The 64-bin distogram is projected into an edge representation.

PAE is used as an edge-confidence regulator:

```text
higher PAE
    ↓
lower structural message weight
```

This allows uncertain AlphaFold2 residue relationships to contribute less strongly to structural message propagation.

---

## 11. Task-driven hyperbolic routing

The structural embeddings are projected into a Poincaré ball.

For each task, the central lysine sequence representation is combined with a task-specific learnable anchor.

Hyperbolic centre-to-residue distance is incorporated as a topology bias during structural retrieval.

The final configuration uses:

```text
initial hyperbolic curvature : 2.0
topology bias initialization : 1.0
hidden dimension             : 64
```

The curvature parameter remains learnable during training.

---

## 12. Training objective

KAT-Net uses masked asymmetric focal loss.

The final loss configuration is:

```text
gamma_pos = 0.0
gamma_neg = 2.0
margin    = 0.05
```

Task uncertainty weighting is applied through learnable log-variance parameters with:

```text
log-var clipping = [-1.5, 2.0]
```

Unavailable task labels (`-1`) are excluded from the corresponding task loss.

---

## 13. Final training configuration

The final five-fold training protocol uses:

```text
optimizer              : AdamW
batch size             : 64
learning rate          : 1e-4
maximum epochs         : 70
early-stopping patience: 15
random seed            : 42
number of folds        : 5
hidden dimension       : 64
number of prompts      : 3
```

Checkpoint selection is based only on validation performance.

The development data use predefined folds:

```text
Fold = 1, 2, 3, 4, 5
```

For each fold:

```text
4 folds → training
1 fold  → validation
```

Run:

```bash
python train.py \
  --csv-path <cleaned_training_csv> \
  --npz-dir <structural_feature_directory> \
  --output-dir <training_output_directory> \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --epochs 70 \
  --patience 15 \
  --seed 42
```

By default, ESM-2 is loaded from:

```text
facebook/esm2_t33_650M_UR50D
```

A local Hugging Face model directory can instead be supplied with:

```bash
--esm-model <local_ESM2_directory>
```

---

## 14. Pretrained checkpoints

The final model consists of five fold-specific checkpoints:

```text
katnet_3ptm_fold_1.pth
katnet_3ptm_fold_2.pth
katnet_3ptm_fold_3.pth
katnet_3ptm_fold_4.pth
katnet_3ptm_fold_5.pth
```

Because of their file sizes, the `.pth` files are distributed through the repository's **GitHub Release** rather than stored directly in the normal Git history.

After downloading, place all five files under:

```text
checkpoints/
├── README.md
├── katnet_3ptm_fold_1.pth
├── katnet_3ptm_fold_2.pth
├── katnet_3ptm_fold_3.pth
├── katnet_3ptm_fold_4.pth
└── katnet_3ptm_fold_5.pth
```

Do not rename the checkpoint files unless the checkpoint pattern passed to `test.py` is changed accordingly.

---

## 15. Validation-derived decision thresholds

Final task-specific thresholds are stored in:

```text
audit/validation_thresholds.json
```

The thresholds were determined exclusively from pooled out-of-fold validation predictions:

| Task  | Threshold |
| ----- | --------: |
| Kcr   |    0.6071 |
| Ksucc |    0.5561 |
| Kac   |    0.4135 |

The independent and external test labels were **not used** for:

```text
model selection
checkpoint selection
hyperparameter selection
threshold optimisation
```

The thresholds were frozen before independent testing.

---

## 16. Five-fold ensemble testing

After downloading the five checkpoints and generating the required structural features, run:

```bash
python test.py \
  --test-csv <independent_test.csv> \
  --npz-dir <structural_feature_directory> \
  --checkpoint-dir checkpoints \
  --threshold-file audit/validation_thresholds.json \
  --output-dir results/katnet \
  --batch-size 64
```

The five fold-specific models independently generate prediction probabilities.

Final probabilities are calculated as:

```text
P_final =
mean(
    P_fold1,
    P_fold2,
    P_fold3,
    P_fold4,
    P_fold5
)
```

The frozen task-specific validation thresholds are then applied to the ensemble probabilities.

---

## 17. Testing outputs

The testing script produces:

```text
independent_test_predictions.csv
fold_metrics.csv
ensemble_metrics.csv
ensemble_roc_curve.csv
ensemble_pr_curve.csv
```

For standard binary datasets, reported metrics include:

```text
ACC
MCC
F1-score
Precision
Sensitivity
Specificity
AUROC
AUPRC
```

For positive-only external datasets, two-class metrics such as AUROC, MCC and specificity are not applicable. Such datasets are interpreted only as known-site recognition evaluations.

---

## 18. External Kcr evaluation

The external Kcr datasets are stored under:

```text
datasets/cross-species/
```

and organized as:

```text
animal/
human/
microorganism/
plant/
```

The external evaluation includes datasets from:

```text
rice
papaya
zebrafish
Toxoplasma gondii
lung cancer-related proteins
independent human data
```

Datasets containing both positive and negative samples support complete binary evaluation.

Positive-only datasets are evaluated only in terms of recognition of experimentally reported modification sites.

None of these external datasets is used for model fitting, checkpoint selection, hyperparameter selection or decision-threshold determination.

---

## 19. Independent functional analysis

Processed files used for the independent Kac/Ksucc functional analysis are provided under:

```text
datasets/go/
```

The analysis is based on the independent PXD049146 mouse lung proteome, acetylome and succinylome dataset.

Experimentally reported Kac and Ksucc sites are processed through the same KAT-Net sequence–structure prediction workflow.

Proteins quantified in the same proteomic study are used as the GO enrichment background.

The analysis is intended to evaluate functional plausibility and biological concordance of the predictions rather than establish functional causality for individual lysine sites.

---

## 20. Software environment

Model training/testing and structural feature generation were performed in separate environments.

### Model training and testing

Reference environment:

| Package         | Version     |
| --------------- | ----------- |
| Python          | 3.10.19     |
| PyTorch         | 2.5.1+cu121 |
| Transformers    | 4.43.3      |
| torch-geometric | 2.8.0.post1 |
| NumPy           | 1.26.4      |
| Pandas          | 2.1.4       |
| SciPy           | 1.15.3      |
| scikit-learn    | 1.7.2       |
| Biopython       | 1.81        |
| einops          | 0.8.1       |
| tqdm            | 4.65.0      |

Example:

```bash
conda create -n katnet python=3.10
conda activate katnet
```

For CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the principal Python dependencies:

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

If CUDA 12.1 is unavailable, install the PyTorch build appropriate for the local CUDA environment.

---

### Structural feature generation

Reference structural environment:

| Package             | Version |
| ------------------- | ------- |
| Python              | 3.9.23  |
| ColabFold           | 1.5.5   |
| alphafold-colabfold | 2.3.6   |
| Biopython           | 1.82    |
| DSSP                | 3.0.0   |
| JAX                 | 0.4.30  |
| jaxlib              | 0.4.30  |
| NumPy               | 1.26.4  |
| Pandas              | 1.5.3   |
| SciPy               | 1.13.1  |

Example:

```bash
conda create -n katnet-structure python=3.9
conda activate katnet-structure
```

Install DSSP:

```bash
conda install -c salilab dssp=3.0.0
```

Install the main Python dependencies:

```bash
pip install \
  colabfold==1.5.5 \
  alphafold-colabfold==2.3.6 \
  biopython==1.82 \
  numpy==1.26.4 \
  pandas==1.5.3 \
  scipy==1.13.1
```

JAX and CUDA-specific ColabFold dependencies should be installed according to the local CUDA configuration.

---

## 21. Reproducibility notes

The following rules are used throughout the released workflow:

```text
Independent-test assignments are preserved during cross-task filtering.

Cross-task redundancy filtering is applied only to development/training data.

Unavailable multi-task labels are masked rather than converted to negatives.

Identical central 31-residue windows are assigned to the same fold.

Structural solvent accessibility is represented as DSSP-derived RSA.

Five fold-specific models are used for final ensemble prediction.

Decision thresholds are determined only from pooled OOF validation data.

Independent and external test labels are not used for threshold optimisation.

AlphaFold2/ColabFold structure-generation time is separate from model inference time.

Positive-only external datasets are not interpreted as binary-classification benchmarks.
```

---

## 22. Structural prediction uncertainty

KAT-Net uses predicted structural information and therefore remains dependent on AlphaFold2 structural quality.

pLDDT and PAE are incorporated to represent structural confidence, and PAE-aware propagation reduces the influence of uncertain residue relationships.

However, confidence-aware processing cannot correct an inaccurate predicted structure.

Predictions involving:

```text
low-pLDDT residues
high-PAE residue relationships
poorly constrained structural regions
```

should therefore be interpreted together with sequence evidence and treated as candidates for subsequent experimental investigation.

---

## 23. Citation

If you use KAT-Net, the released datasets, or the pretrained checkpoints, please cite:

```text
Wu H, Lin Y, Zhu L, Yang S.
KAT-Net: a hyperbolic routing framework for joint lysine acylation site prediction
with shared sequence–structure representations.
```

The final journal citation and DOI will be added after publication.

---

## 24. Data and model availability

The repository provides:

```text
source code
training code
testing code
structural feature-generation code
cross-task redundancy audit
original benchmark datasets
final cleaned multi-task dataset
external Kcr evaluation datasets
independent functional-analysis data
validation-derived decision thresholds
reproducibility documentation
```

The five pretrained model checkpoints are distributed separately through the GitHub Release.

---

## 25. Contact

For questions regarding KAT-Net, its datasets or reproducibility, please open an issue in this repository.

Corresponding authors:

```text
Lun Zhu
zl@cczu.edu.cn

Sen Yang
ys@cczu.edu.cn
```
