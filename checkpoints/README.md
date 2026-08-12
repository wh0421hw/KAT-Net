# KAT-Net Pretrained Checkpoints

This directory is reserved for the five pretrained fold-specific KAT-Net checkpoints used in the manuscript.

The `.pth` checkpoint files are **not stored directly in the normal Git repository** because of their large file sizes.

They are distributed through the GitHub **v1.0.0 Release**.

---

## Checkpoint files

Download the following five files from the GitHub Release:

```text
katnet_3ptm_fold_1.pth
katnet_3ptm_fold_2.pth
katnet_3ptm_fold_3.pth
katnet_3ptm_fold_4.pth
katnet_3ptm_fold_5.pth
```

After downloading, place all five files in this directory:

```text
checkpoints/
├── README.md
├── katnet_3ptm_fold_1.pth
├── katnet_3ptm_fold_2.pth
├── katnet_3ptm_fold_3.pth
├── katnet_3ptm_fold_4.pth
└── katnet_3ptm_fold_5.pth
```

Keep the original checkpoint filenames unchanged unless the checkpoint-loading logic in `test.py` is also updated.

---

## Five-fold ensemble inference

The five checkpoints correspond to the predefined five-fold training procedure used in the manuscript.

For each candidate lysine, the five fold-specific models generate prediction probabilities:

```text
P_fold1
P_fold2
P_fold3
P_fold4
P_fold5
```

The final ensemble probability is calculated as:

```text
P_final = mean(
    P_fold1,
    P_fold2,
    P_fold3,
    P_fold4,
    P_fold5
)
```

The averaged probability is then converted to a binary prediction using the frozen task-specific validation threshold.

---

## Validation-derived thresholds

The final thresholds are stored in:

```text
../audit/validation_thresholds.json
```

The values used in the manuscript are:

| Task | Threshold |
|---|---:|
| Kcr | 0.6071 |
| Ksucc | 0.5561 |
| Kac | 0.4135 |

These thresholds were derived exclusively from pooled out-of-fold validation predictions.

Independent-test and external-test labels were not used for threshold selection.

---

## Using the pretrained checkpoints

After downloading all five `.pth` files and placing them under:

```text
checkpoints/
```

return to the repository root directory.

Run:

```bash
python test.py
```

The testing workflow uses:

```text
checkpoints/
    five pretrained fold-specific models

audit/validation_thresholds.json
    frozen task-specific thresholds

datasets/
    independent or external evaluation data

structural features
    AlphaFold2/ColabFold-derived KAT-Net structural inputs
```

The testing script averages the five model probabilities before applying the corresponding task threshold.

---

## Required files

Before running pretrained inference, the expected directory structure is:

```text
KAT-Net/
├── test.py
│
├── checkpoints/
│   ├── README.md
│   ├── katnet_3ptm_fold_1.pth
│   ├── katnet_3ptm_fold_2.pth
│   ├── katnet_3ptm_fold_3.pth
│   ├── katnet_3ptm_fold_4.pth
│   └── katnet_3ptm_fold_5.pth
│
├── audit/
│   └── validation_thresholds.json
│
└── datasets/
```

---

## Structural inputs

KAT-Net requires AlphaFold2/ColabFold-derived structural features.

The released structural feature extraction workflow provides:

```text
coords
plddt
pae
sasa
ss
disto
```

The NPZ key:

```text
sasa
```

is a legacy compatibility name.

It stores:

```text
DSSP-derived relative solvent accessibility (RSA)
```

rather than FreeSASA-derived absolute SASA.

If structural NPZ files are not already available, regenerate them using the structural feature extraction script provided in the main repository.

---

## Notes

- All five checkpoints are required to reproduce the reported five-fold ensemble predictions.
- Checkpoint filenames should remain unchanged for direct compatibility with the released testing code.
- Decision thresholds must be loaded from `audit/validation_thresholds.json`.
- Do not optimise thresholds using independent or external test labels.
- The checkpoints correspond to the final KAT-Net configuration reported in the manuscript.
- External positive-only datasets should be interpreted only as known-site recognition evaluations.
- Structural feature generation is performed separately from checkpoint inference.

---

## Citation

If you use these pretrained checkpoints, please cite:

```text
Wu H, Lin Y, Zhu L, Yang S.
KAT-Net: a hyperbolic routing framework for joint lysine acylation site prediction
with shared sequence–structure representations.
```

The final journal citation and DOI will be added after publication.
