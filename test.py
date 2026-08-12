# -*- coding: utf-8 -*-
"""KAT-Net five-fold ensemble testing."""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel


TASKS = ("Kcr", "Ksucc", "Kac")
LABELS = ("Kcr_Label", "Ksucc_Label", "Kac_Label")
N_TASKS = 3
HIDDEN = 64
LPE_DIM = 8

AA_PHYS = {
    "A": [1.8, 0, 0, 0], "R": [-4.5, 1, 1, 0], "N": [-3.5, 0, 0.2, 0],
    "D": [-3.5, -1, 0.2, 0], "C": [2.5, 0, 0.1, 0], "Q": [-3.5, 0, 0.5, 0],
    "E": [-3.5, -1, 0.5, 0], "G": [-0.4, 0, -1, 0], "H": [-3.2, 0.5, 0.5, 1],
    "I": [4.5, 0, 0.6, 0], "L": [3.8, 0, 0.6, 0], "K": [-3.9, 1, 0.8, 0],
    "M": [1.9, 0, 0.6, 0], "F": [2.8, 0, 0.8, 1], "P": [-1.6, 0, 0.1, 0],
    "S": [-0.8, 0, -0.5, 0], "T": [-0.7, 0, 0.1, 0], "W": [-0.9, 0, 1.5, 1],
    "Y": [-1.3, 0, 1, 1], "V": [4.2, 0, 0.4, 0], "X": [0, 0, 0, 0],
}


# Reproducibility
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phys_features(seq):
    return torch.tensor(
        [AA_PHYS.get(x.upper(), AA_PHYS["X"]) for x in seq],
        dtype=torch.float32,
    )


# Dataset
def load_test_frame(path):
    df = pd.read_csv(path)
    required = {"Dynamic_Sequence", "Core_Key", "Center_K_Index"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["Dynamic_Sequence"] = (
        df["Dynamic_Sequence"].astype(str).str.upper().str.replace(" ", "", regex=False)
    )
    df["Core_Key"] = df["Core_Key"].astype(str)
    df["Center_K_Index"] = pd.to_numeric(df["Center_K_Index"], errors="raise").astype(int)

    for col in LABELS:
        if col not in df.columns:
            df[col] = -1
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

        if not set(df[col].unique()).issubset({-1, 0, 1}):
            raise ValueError(f"Invalid labels in {col}")

    for row in df.itertuples(index=False):
        seq = row.Dynamic_Sequence
        center = int(row.Center_K_Index)
        if center < 0 or center >= len(seq) or seq[center] != "K":
            raise ValueError(f"Invalid center K: {row.Core_Key}")

    return df.reset_index(drop=True)


class KATDataset(Dataset):
    def __init__(self, frame, npz_dir, tokenizer):
        self.df = frame
        self.npz_dir = Path(npz_dir)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["Dynamic_Sequence"])
        key = str(row["Core_Key"])
        center = int(row["Center_K_Index"])
        path = self.npz_dir / f"{key}.npz"

        if not path.is_file():
            raise FileNotFoundError(f"Missing NPZ: {path}")

        enc = self.tokenizer(seq, return_tensors="pt", truncation=False, padding=False)

        with np.load(path) as d:
            struct = {
                "coords": torch.tensor(d["coords"], dtype=torch.float32),
                "plddt": torch.tensor(d["plddt"], dtype=torch.float32),
                "sasa": torch.tensor(d["sasa"], dtype=torch.float32),
                "ss": torch.tensor(d["ss"], dtype=torch.float32),
                "pae": torch.tensor(d["pae"], dtype=torch.float32),
                "disto": torch.tensor(d["disto"], dtype=torch.float32),
            }

        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            phys_features(seq),
            struct,
            center,
            torch.tensor([row[x] for x in LABELS], dtype=torch.float32),
        )


def collate_fn(batch):
    ids, masks, phys, structs, centers, labels = zip(*batch)
    B, T, L = len(batch), max(map(len, ids)), max(map(len, phys))

    p_ids = torch.zeros(B, T, dtype=torch.long)
    p_masks = torch.zeros(B, T, dtype=torch.long)
    p_phys = torch.zeros(B, L, 4)
    coords = torch.zeros(B, L, 3)
    plddt = torch.zeros(B, L)
    sasa = torch.zeros(B, L)
    ss = torch.zeros(B, L, 3)
    pae = torch.full((B, L, L), 30.0)
    disto = torch.zeros(B, L, L, 64)

    for i in range(B):
        t, n = len(ids[i]), len(phys[i])

        p_ids[i, :t] = ids[i]
        p_masks[i, :t] = masks[i]
        p_phys[i, :n] = phys[i]
        coords[i, :n] = structs[i]["coords"][:n]
        plddt[i, :n] = structs[i]["plddt"][:n]
        sasa[i, :n] = structs[i]["sasa"][:n]
        ss[i, :n] = structs[i]["ss"][:n]
        pae[i, :n, :n] = structs[i]["pae"][:n, :n]
        disto[i, :n, :n] = structs[i]["disto"][:n, :n]

    struct = (coords, plddt, sasa, ss, pae, disto)

    return (
        p_ids,
        p_masks,
        p_phys,
        struct,
        torch.tensor(centers, dtype=torch.long),
        torch.stack(labels),
    )


def to_device(batch, device):
    ids, mask, phys, struct, center, labels = batch
    return (
        ids.to(device),
        mask.to(device),
        phys.to(device),
        tuple(x.to(device) for x in struct),
        center.to(device),
        labels.to(device),
    )


# Soft prompt
class SoftPromptAdapter(nn.Module):
    def __init__(self, dim=1280, n=3):
        super().__init__()
        self.n = n
        self.prompt = nn.Parameter(torch.randn(1, n * 2, dim))
        nn.init.xavier_uniform_(self.prompt)

    def forward(self, x):
        p = self.prompt.expand(x.size(0), -1, -1)
        return torch.cat((p[:, :self.n], x, p[:, self.n:]), dim=1), self.n


# CSRE
class ChemoSpatialEmbedding(nn.Module):
    def __init__(self, dim=64, sigma=3.0):
        super().__init__()
        self.phys = nn.Sequential(
            nn.Linear(4, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim)
        )
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rel_pos = nn.Embedding(401, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, center):
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).float().unsqueeze(0).expand(B, -1)
        rel = pos - center.unsqueeze(1).float()
        weight = torch.exp(-(rel ** 2) / (2 * self.sigma ** 2)).unsqueeze(-1)
        idx = torch.clamp((rel + 200).long(), 0, 400)
        return self.norm(self.phys(x) * weight + self.rel_pos(idx))


# Mamba
class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dropout=0.2):
        super().__init__()
        self.inner = expand * d_model
        self.rank = math.ceil(self.inner / 16)
        self.state = d_state

        self.in_proj = nn.Linear(d_model, self.inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.inner, self.inner, d_conv,
            groups=self.inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.inner, self.rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .repeat(self.inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)

        u = F.silu(
            self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        )

        dt, b, c = torch.split(
            self.x_proj(u),
            [self.rank, self.state, self.state],
            dim=-1,
        )

        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.inner, self.state, device=x.device)
        outputs = []

        for i in range(L):
            dti = dt[:, i].unsqueeze(-1)
            h = (
                torch.exp(dti * A) * h
                + (dti * b[:, i].unsqueeze(1)) * u[:, i].unsqueeze(-1)
            )
            outputs.append((h * c[:, i].unsqueeze(1)).sum(-1))

        y = (torch.stack(outputs, dim=1) + u * self.D) * F.silu(z)
        return x + self.dropout(self.out_proj(y))


class CenterAnchoredMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1280 + HIDDEN, HIDDEN)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.layers = nn.ModuleList([MambaBlock() for _ in range(2)])
        self.norm2 = nn.LayerNorm(HIDDEN)

    def forward(self, esm, csre, prompt_len):
        L = csre.size(1)
        csre = F.pad(csre, (0, 0, prompt_len, prompt_len))
        x = self.norm1(self.proj(torch.cat((esm, csre), dim=-1)))

        for layer in self.layers:
            x = (layer(x) + layer(x.flip(1)).flip(1)) / 2

        return self.norm2(x[:, prompt_len:prompt_len + L])


# PAE-aware EGNN
class EGNNLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.edge = nn.Sequential(
            nn.Linear(HIDDEN * 2 + 16 + 1, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )

        self.att = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())

        self.node = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

        self.coord = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, 1, bias=False),
        )

    def forward(self, h, x, edge_attr, pae_weight):
        B, N, _ = h.shape
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        radial = (diff ** 2).sum(-1, keepdim=True)

        hi = h.unsqueeze(2).expand(B, N, N, -1)
        hj = h.unsqueeze(1).expand(B, N, N, -1)

        m = self.edge(
            torch.cat((hi, hj, radial, edge_attr), dim=-1)
        ) * pae_weight

        agg = (m * self.att(m)).sum(2)

        h = h + self.node(torch.cat((h, agg), dim=-1))
        x = x + (diff * self.coord(m)).sum(2)

        return h, x


class StructureEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.node_embed = nn.Linear(5 + LPE_DIM, HIDDEN)
        self.disto_embed = nn.Linear(64, 16)
        self.pae_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = nn.ModuleList([EGNNLayer() for _ in range(2)])
        self.norm = nn.LayerNorm(HIDDEN)

    @staticmethod
    def laplacian_pe(coords):
        B, N, _ = coords.shape
        dist = torch.cdist(coords, coords)
        k = min(7, N)
        idx = torch.topk(dist, k=k, largest=False).indices

        adj = torch.zeros(B, N, N, device=coords.device)
        bi = torch.arange(B, device=coords.device).view(-1, 1, 1)
        ni = torch.arange(N, device=coords.device).view(1, -1, 1)

        adj[bi, ni, idx] = 1
        adj = ((adj + adj.transpose(1, 2)) > 0).float()

        degree = torch.diag_embed(
            (adj.sum(-1) + 1e-6).pow(-0.5)
        )

        eye = torch.eye(N, device=coords.device).unsqueeze(0)
        lap = eye - degree @ adj @ degree + 1e-6 * eye

        try:
            vec = torch.linalg.eigh(lap).eigenvectors[:, :, 1:LPE_DIM + 1]
        except RuntimeError:
            vec = torch.zeros(B, N, 0, device=coords.device)

        return F.pad(vec, (0, max(0, LPE_DIM - vec.size(-1))))

    def forward(self, struct):
        coords, plddt, sasa, ss, pae, disto = struct

        node = torch.cat(
            (
                plddt.unsqueeze(-1),
                sasa.unsqueeze(-1),
                ss,
                self.laplacian_pe(coords),
            ),
            dim=-1,
        )

        h = F.gelu(self.node_embed(node))
        edge = F.gelu(self.disto_embed(disto))
        weight = torch.exp(-pae * self.pae_scale.abs()).unsqueeze(-1)

        x = coords
        for layer in self.layers:
            h, x = layer(h, x, edge, weight)

        return self.norm(h)


# Hyperbolic routing
class HyperbolicRouting(nn.Module):
    def __init__(self, c_init=1.0):
        super().__init__()

        self.task_anchors = nn.Parameter(torch.randn(N_TASKS, HIDDEN))
        nn.init.xavier_uniform_(self.task_anchors)

        self.c_raw = nn.Parameter(
            torch.tensor(
                [np.log(np.exp(c_init) - 1)],
                dtype=torch.float32,
            )
        )

        self.geo_scale = nn.Parameter(torch.tensor(1.0))
        self.Wq = nn.Linear(HIDDEN, HIDDEN)
        self.Wk = nn.Linear(HIDDEN, HIDDEN)
        self.Wv = nn.Linear(HIDDEN, HIDDEN)
        self.proj_struct = nn.Linear(HIDDEN, HIDDEN)

        self.gate = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    @property
    def c(self):
        return F.softplus(self.c_raw) + 1e-5

    def hyp_dist(self, x, y):
        x2 = x.norm(dim=-1, keepdim=True).pow(2)
        y2 = y.norm(dim=-1, keepdim=True).pow(2)
        d2 = (x - y).norm(dim=-1, keepdim=True).pow(2)

        denominator = torch.clamp(
            (1 - self.c * x2) * (1 - self.c * y2),
            min=1e-5,
        )

        z = torch.clamp(
            1 + self.c * (2 * d2 / denominator),
            min=1 + 1e-6,
        )

        return torch.acosh(z) / torch.sqrt(self.c)

    def forward(self, seq, struct, center):
        B = seq.size(0)
        bi = torch.arange(B, device=seq.device)

        p = self.proj_struct(struct)
        norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-5)

        z = (
            torch.tanh(torch.sqrt(self.c) * norm)
            / (torch.sqrt(self.c) * norm)
            * p
        )

        d_hyp = self.hyp_dist(
            z,
            z[bi, center].unsqueeze(1),
        )

        query = (
            seq[bi, center].unsqueeze(1)
            + self.task_anchors.unsqueeze(0)
        )

        q = self.Wq(query)
        k = self.Wk(struct)
        v = self.Wv(struct)

        att = (
            torch.matmul(q, k.transpose(-2, -1))
            / math.sqrt(HIDDEN)
        )

        att = F.softmax(
            att - d_hyp.transpose(1, 2) * self.geo_scale,
            dim=-1,
        )

        context = torch.matmul(att, v)
        combined = torch.cat((query, context), dim=-1)
        gate = self.gate(combined)
        fused = gate * query + (1 - gate) * context

        return self.out(torch.cat((query, fused), dim=-1)) + query


# KAT-Net
class KATNet(nn.Module):
    def __init__(self, esm, prompts=3, dropout=0.3, sigma=3.0):
        super().__init__()

        self.esm2 = esm
        self.prompt_adapter = SoftPromptAdapter(n=prompts)
        self.csre_embedding = ChemoSpatialEmbedding(sigma=sigma)
        self.seq_encoder = CenterAnchoredMamba()
        self.struct_encoder = StructureEncoder()
        self.fusion_module = HyperbolicRouting()

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            for _ in range(N_TASKS)
        ])

    def forward(self, ids, mask, phys, struct, center):
        esm = self.esm2(
            input_ids=ids,
            attention_mask=mask,
        ).last_hidden_state

        L = phys.size(1)
        esm = esm[:, 1:L + 1]

        esm, prompt_len = self.prompt_adapter(esm)
        csre = self.csre_embedding(phys, center)

        seq = self.seq_encoder(esm, csre, prompt_len)
        struct = self.struct_encoder(struct)
        features = self.fusion_module(seq, struct, center)

        return torch.cat(
            [head(features[:, i]) for i, head in enumerate(self.task_heads)],
            dim=-1,
        )


# Checkpoints
def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_state(checkpoint):
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def build_model(checkpoint, esm_source, device):
    esm = EsmModel.from_pretrained(esm_source)
    esm.encoder.layer = esm.encoder.layer[:6]

    prompts = int(checkpoint.get("num_prompts", 3))
    dropout = float(checkpoint.get("dropout", 0.3))
    sigma = float(checkpoint.get("sigma", checkpoint.get("sigma_init", 3.0)))

    model = KATNet(
        esm,
        prompts=prompts,
        dropout=dropout,
        sigma=sigma,
    ).to(device)

    model.load_state_dict(
        get_state(checkpoint),
        strict=True,
    )

    return model.eval()


def checkpoint_paths(directory, pattern):
    paths = [
        Path(directory) / pattern.format(fold=i)
        for i in range(1, 6)
    ]

    missing = [p for p in paths if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing checkpoints:\n"
            + "\n".join(map(str, missing))
        )

    return paths


# Frozen thresholds
def load_thresholds(path):
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if data.get("test_labels_used", False):
        raise ValueError("Invalid threshold file: test labels were used.")

    values = {}

    for task in TASKS:
        value = data["tasks"][task]
        values[task] = float(
            value["threshold"] if isinstance(value, dict) else value
        )

    return values


# Prediction
@torch.no_grad()
def predict(model, loader, device):
    labels, probabilities = [], []

    for batch in loader:
        ids, mask, phys, struct, center, y = to_device(batch, device)

        logits = model(
            ids,
            mask,
            phys,
            struct,
            center,
        )

        labels.append(y.cpu().numpy())
        probabilities.append(
            torch.sigmoid(logits).cpu().numpy()
        )

    return np.concatenate(labels), np.concatenate(probabilities)


# Metrics
def calculate_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "Threshold": float(threshold),
        "N": len(y),
        "Positive": int((y == 1).sum()),
        "Negative": int((y == 0).sum()),
        "ACC": accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
    }


def evaluate(labels, probabilities, thresholds, scope, fold=None):
    rows = []

    for i, task in enumerate(TASKS):
        valid = labels[:, i] != -1

        if not valid.any():
            continue

        y = labels[valid, i].astype(int)
        p = probabilities[valid, i]

        if np.unique(y).size < 2:
            continue

        rows.append({
            "Scope": scope,
            "Fold": fold,
            "Task": task,
            **calculate_metrics(y, p, thresholds[task]),
        })

    return rows


# ROC and PR
def save_curves(labels, probabilities, output_dir):
    roc_rows, pr_rows = [], []

    for i, task in enumerate(TASKS):
        valid = labels[:, i] != -1

        if not valid.any():
            continue

        y = labels[valid, i].astype(int)
        p = probabilities[valid, i]

        if np.unique(y).size < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        for a, b, c in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "Task": task,
                "FPR": a,
                "TPR": b,
                "Threshold": c,
                "AUROC": auc,
            })

        precision, recall, thresholds = precision_recall_curve(y, p)
        thresholds = np.append(thresholds, np.nan)
        auprc = average_precision_score(y, p)

        for a, b, c in zip(precision, recall, thresholds):
            pr_rows.append({
                "Task": task,
                "Precision": a,
                "Recall": b,
                "Threshold": c,
                "AUPRC": auprc,
            })

    if roc_rows:
        pd.DataFrame(roc_rows).to_csv(
            output_dir / "ensemble_roc_curve.csv",
            index=False,
        )

    if pr_rows:
        pd.DataFrame(pr_rows).to_csv(
            output_dir / "ensemble_pr_curve.csv",
            index=False,
        )


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="KAT-Net five-fold ensemble testing")

    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--threshold-file", type=Path, required=True)

    p.add_argument(
        "--esm-model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model ID or local model directory.",
    )

    p.add_argument("--output-dir", type=Path, default=Path("results/katnet"))
    p.add_argument(
        "--checkpoint-pattern",
        default="katnet_3ptm_fold_{fold}.pth",
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    if not args.test_csv.is_file():
        raise FileNotFoundError(args.test_csv)

    if not args.npz_dir.is_dir():
        raise FileNotFoundError(args.npz_dir)

    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(args.checkpoint_dir)

    if not args.threshold_file.is_file():
        raise FileNotFoundError(args.threshold_file)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; use --overwrite."
            )
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    frame = load_test_frame(args.test_csv)
    thresholds = load_thresholds(args.threshold_file)

    paths = checkpoint_paths(
        args.checkpoint_dir,
        args.checkpoint_pattern,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)

    dataset = KATDataset(
        frame,
        args.npz_dir,
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print("\nKAT-Net five-fold ensemble test")
    print(f"Device     : {device}")
    print(f"Samples    : {len(frame)}")
    print(f"Thresholds : {thresholds}")
    print("Thresholds : validation-derived only")
    print("Test search: disabled\n")

    fold_probs = []
    fold_metrics = []
    common_labels = None

    for fold, path in enumerate(paths, start=1):
        print(f"Testing fold {fold}/5")

        checkpoint = load_checkpoint(path, device)
        model = build_model(
            checkpoint,
            args.esm_model,
            device,
        )

        labels, probabilities = predict(
            model,
            loader,
            device,
        )

        if common_labels is None:
            common_labels = labels
        elif not np.array_equal(common_labels, labels):
            raise RuntimeError("Test row order changed between folds.")

        fold_probs.append(probabilities)

        fold_metrics.extend(
            evaluate(
                labels,
                probabilities,
                thresholds,
                scope="Single_Fold",
                fold=fold,
            )
        )

        del model, checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_probs = np.stack(fold_probs, axis=0)
    ensemble = fold_probs.mean(axis=0)
    std = fold_probs.std(axis=0)

    ensemble_metrics = evaluate(
        common_labels,
        ensemble,
        thresholds,
        scope="Five_Fold_Ensemble",
    )

    output = frame.copy()

    for fold in range(5):
        for i, task in enumerate(TASKS):
            output[f"{task}_Fold{fold + 1}_Probability"] = (
                fold_probs[fold, :, i]
            )

    for i, task in enumerate(TASKS):
        output[f"{task}_Ensemble_Probability"] = ensemble[:, i]
        output[f"{task}_Probability_SD"] = std[:, i]
        output[f"{task}_Threshold"] = thresholds[task]
        output[f"{task}_Prediction"] = (
            ensemble[:, i] >= thresholds[task]
        ).astype(int)

    output.to_csv(
        args.output_dir / "independent_test_predictions.csv",
        index=False,
    )

    pd.DataFrame(fold_metrics).to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )

    ensemble_df = pd.DataFrame(ensemble_metrics)

    ensemble_df.to_csv(
        args.output_dir / "ensemble_metrics.csv",
        index=False,
    )

    save_curves(
        common_labels,
        ensemble,
        args.output_dir,
    )

    print("\nFive-fold ensemble metrics:")

    if ensemble_df.empty:
        print("No labelled two-class task found; predictions only.")
    else:
        print(ensemble_df.to_string(index=False))

    print(f"\nOutput: {args.output_dir}")
    print("[Status] PASS")


if __name__ == "__main__":
    main()