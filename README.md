# KAT-Net: Joint prediction of lysine acylation sites by integrating sequence, structure and hyperbolic routing

KAT-Net is a computational framework for the joint prediction of lysine crotonylation (Kcr), succinylation (Ksucc) and acetylation (Kac) sites. The model integrates lysine-centred sequence information, structural context and task-driven hyperbolic routing to capture shared and task-specific patterns among different lysine acylation types.

This repository provides the datasets and the main training and testing scripts used for KAT-Net.

<p align="center">
  <img src="katnet_overview.png" alt="Overview of the KAT-Net framework" width="850">
</p>

## Features

* **Joint lysine acylation prediction:** KAT-Net predicts Kcr, Ksucc and Kac sites within a unified framework.
* **Lysine-centred modelling:** Each sample is organized around a candidate lysine residue.
* **Protein language model representation:** ESM-2 is used to extract residue-level sequence representations.
* **Structure-informed prediction:** AlphaFold2-derived structural features are used when available.
* **Task-driven fusion:** Hyperbolic routing is used to integrate sequence and structural information for different acylation tasks.
* **Five-fold ensemble evaluation:** The testing script reports the final ensemble performance across trained folds.

## Table of Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Datasets](#datasets)
4. [External Model Files](#external-model-files)
5. [Training](#training)
6. [Testing](#testing)
7. [Output Metrics](#output-metrics)
8. [Citation](#citation)
9. [Contact](#contact)

## Installation

Clone this repository:

```bash
git clone https://github.com/wh0421hw/KAT-Net.git
cd KAT-Net
```

Create a Python environment:

```bash
conda create -n katnet python=3.9
conda activate katnet
```

Install the required packages:

```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

Please install a PyTorch version that is compatible with your CUDA environment.

## Repository Structure

```text
KAT-Net/
├── README.md
├── train.py
├── test.py
└── datasets/
    └── ...
```

* `train.py`: training script for KAT-Net.
* `test.py`: testing script for five-fold ensemble evaluation.
* `datasets/`: datasets used for model training and evaluation.

## Datasets

The datasets used in this study are provided in the `datasets/` directory. Each sample is organized around a candidate lysine residue.

The label convention is:

```text
1  = experimentally confirmed positive site
0  = negative sample
-1 = unavailable label for the corresponding task
```

Unavailable labels are masked during model training and do not contribute to the corresponding task-specific loss.

Before running the scripts, please check the dataset format and update the file paths in `train.py` and `test.py` according to your local environment.

## External Model Files

KAT-Net uses `facebook/esm2_t33_650M_UR50D` as the pretrained protein language model. The ESM-2 model files are not included in this repository. Users can download the model automatically through the Hugging Face Transformers library or provide a local path to the downloaded ESM-2 directory.

AlphaFold2-derived structural features are used as structural inputs when available. The full AlphaFold2 parameter files are not included in this repository because of their large file size. Users who need to regenerate structural features should obtain AlphaFold2-related resources from the official release and update the corresponding feature paths in the scripts.

## Training

Before training, set the required paths in `train.py`:

```python
CSV_PATH = ""
NPZ_DIR = ""
ESM_PATH = ""
```

Then run:

```bash
python train.py
```

The script trains KAT-Net for joint Kcr, Ksucc and Kac prediction. Fold-specific model checkpoints are saved during training.

## Testing

Before testing, set the required paths in `test.py`, including the ESM-2 path, test datasets, feature directories and trained model checkpoints.

Then run:

```bash
python test.py
```

The testing script performs five-fold ensemble inference and reports the final evaluation metrics.

## Output Metrics

The testing script reports the following metrics:

* Accuracy (ACC)
* Area under the ROC curve (AUC)
* Matthews correlation coefficient (MCC)
* F1-score
* Sensitivity (Sn)
* Specificity (Sp)

For datasets containing only one class, AUC is not calculated.

## Citation

If you use KAT-Net or the resources in this repository, please cite the associated manuscript:

```text
Wu H, Yang S, Zhu L. KAT-Net: Joint prediction of lysine acylation sites by integrating sequence, structure and hyperbolic routing.
```

## Contact

For questions about the code or datasets, please open an issue in this repository.
