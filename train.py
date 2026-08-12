# -*- coding: utf-8 -*-
"""KAT-Net fixed-fold training."""

import argparse
import copy
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
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
class KATDataset(Dataset):
    def __init__(self, csv_path, npz_dir, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.npz_dir = Path(npz_dir)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["Dynamic_Sequence"])
        key = str(row["Core_Key"])
        center = int(row["Center_K_Index"])

        enc = self.tokenizer(seq, return_tensors="pt", truncation=False, padding=False)

        with np.load(self.npz_dir / f"{key}.npz") as d:
            # "rsa" stores DSSP-derived relative solvent accessibility.
            structure = {
                "coords": torch.tensor(d["coords"], dtype=torch.float32),
                "plddt": torch.tensor(d["plddt"], dtype=torch.float32),
                "rsa": torch.tensor(d["rsa"], dtype=torch.float32),
                "ss": torch.tensor(d["ss"], dtype=torch.float32),
                "pae": torch.tensor(d["pae"], dtype=torch.float32),
                "disto": torch.tensor(d["disto"], dtype=torch.float32),
            }

        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            phys_features(seq),
            structure,
            center,
            torch.tensor([row[x] for x in LABELS], dtype=torch.float32),
        )


def collate_fn(batch):
    ids, masks, phys, structs, centers, labels = zip(*batch)
    B = len(batch)
    T = max(len(x) for x in ids)
    L = max(len(x) for x in phys)

    p_ids = torch.zeros(B, T, dtype=torch.long)
    p_masks = torch.zeros(B, T, dtype=torch.long)
    p_phys = torch.zeros(B, L, 4)

    coords = torch.zeros(B, L, 3)
    plddt = torch.zeros(B, L)
    rsa = torch.zeros(B, L)
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
        rsa[i, :n] = structs[i]["rsa"][:n]
        ss[i, :n] = structs[i]["ss"][:n]
        pae[i, :n, :n] = structs[i]["pae"][:n, :n]
        disto[i, :n, :n] = structs[i]["disto"][:n, :n]

    structure = (coords, plddt, rsa, ss, pae, disto)
    return p_ids, p_masks, p_phys, structure, torch.tensor(centers), torch.stack(labels)


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
        self.phys = nn.Sequential(nn.Linear(4, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rel_pos = nn.Embedding(401, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, center):
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).float().unsqueeze(0).expand(B, -1)
        rel = pos - center.unsqueeze(1).float()
        w = torch.exp(-(rel ** 2) / (2 * self.sigma ** 2)).unsqueeze(-1)
        idx = torch.clamp((rel + 200).long(), 0, 400)
        return self.norm(self.phys(x) * w + self.rel_pos(idx))


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
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))

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


# PAE-aware EGNN (node features include pLDDT, DSSP-derived RSA and SS)
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

        m = self.edge(torch.cat((hi, hj, radial, edge_attr), dim=-1)) * pae_weight
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

        d = torch.diag_embed((adj.sum(-1) + 1e-6).pow(-0.5))
        eye = torch.eye(N, device=coords.device).unsqueeze(0)
        lap = eye - d @ adj @ d + 1e-6 * eye

        try:
            vec = torch.linalg.eigh(lap).eigenvectors[:, :, 1:LPE_DIM + 1]
        except RuntimeError:
            vec = torch.zeros(B, N, 0, device=coords.device)

        return F.pad(vec, (0, max(0, LPE_DIM - vec.size(-1))))

    def forward(self, struct):
        coords, plddt, rsa, ss, pae, disto = struct

        node = torch.cat(
            (plddt.unsqueeze(-1), rsa.unsqueeze(-1), ss, self.laplacian_pe(coords)),
            dim=-1,
        )
        h = F.gelu(self.node_embed(node))
        edge = F.gelu(self.disto_embed(disto))
        pae_weight = torch.exp(-pae * self.pae_scale.abs()).unsqueeze(-1)

        x = coords
        for layer in self.layers:
            h, x = layer(h, x, edge, pae_weight)

        return self.norm(h)


# Hyperbolic routing
class HyperbolicRouting(nn.Module):
    def __init__(self, c_init=2.0):
        super().__init__()
        self.task_anchors = nn.Parameter(torch.randn(N_TASKS, HIDDEN))
        nn.init.xavier_uniform_(self.task_anchors)

        self.c_raw = nn.Parameter(
            torch.tensor([np.log(np.exp(c_init) - 1)], dtype=torch.float32)
        )
        self.geo_scale = nn.Parameter(torch.tensor(1.0))

        self.Wq = nn.Linear(HIDDEN, HIDDEN)
        self.Wk = nn.Linear(HIDDEN, HIDDEN)
        self.Wv = nn.Linear(HIDDEN, HIDDEN)
        self.proj_struct = nn.Linear(HIDDEN, HIDDEN)

        self.gate = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.Sigmoid())
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

        denom = torch.clamp((1 - self.c * x2) * (1 - self.c * y2), min=1e-5)
        z = torch.clamp(1 + self.c * (2 * d2 / denom), min=1 + 1e-6)
        return torch.acosh(z) / torch.sqrt(self.c)

    def forward(self, seq, struct, center):
        B = seq.size(0)
        bi = torch.arange(B, device=seq.device)

        p = self.proj_struct(struct)
        n = p.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        z = torch.tanh(torch.sqrt(self.c) * n) / (torch.sqrt(self.c) * n) * p

        center_z = z[bi, center].unsqueeze(1)
        d_hyp = self.hyp_dist(z, center_z)

        center_seq = seq[bi, center].unsqueeze(1)
        query = center_seq + self.task_anchors.unsqueeze(0)

        q = self.Wq(query)
        k = self.Wk(struct)
        v = self.Wv(struct)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HIDDEN)
        att = F.softmax(att - d_hyp.transpose(1, 2) * self.geo_scale, dim=-1)

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
        esm = self.esm2(input_ids=ids, attention_mask=mask).last_hidden_state

        if self.training:
            esm = esm + torch.randn_like(esm) * 0.002

        L = phys.size(1)
        esm = esm[:, 1:L + 1]

        esm, prompt_len = self.prompt_adapter(esm)
        csre = self.csre_embedding(phys, center)
        seq = self.seq_encoder(esm, csre, prompt_len)
        struct = self.struct_encoder(struct)
        features = self.fusion_module(seq, struct, center)

        logits = torch.cat(
            [head(features[:, i]) for i, head in enumerate(self.task_heads)],
            dim=-1,
        )
        return logits


# Masked asymmetric loss
class MaskedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, clip=0.05):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.log_vars = nn.Parameter(torch.zeros(N_TASKS))

    def forward(self, logits, targets):
        total = logits.new_tensor(0.0)
        log_vars = self.log_vars.clamp(-1.5, 2.0)

        for t in range(N_TASKS):
            valid = targets[:, t] != -1
            if not valid.any():
                continue

            y = targets[valid, t]
            p = torch.sigmoid(logits[valid, t])

            pos = -y * (1 - p).pow(self.gp) * torch.log(p + 1e-8)
            pn = torch.clamp(p - self.clip, min=0.0)
            neg = -(1 - y) * pn.pow(self.gn) * torch.log(1 - p + 1e-8)

            loss = (pos + neg).mean()
            precision = torch.exp(-log_vars[t])
            total += 0.5 * precision * loss + log_vars[t]

        return total


# ESM-2 setup
def build_esm(source, policy):
    esm = EsmModel.from_pretrained(source)
    esm.encoder.layer = esm.encoder.layer[:6]

    if policy == "embedding_only":
        for p in esm.embeddings.parameters():
            p.requires_grad = False
        for layer in esm.encoder.layer:
            for p in layer.parameters():
                p.requires_grad = True
    elif policy == "all":
        for p in esm.parameters():
            p.requires_grad = False
    else:
        for p in esm.parameters():
            p.requires_grad = True

    return esm


def build_model(args, device):
    model = KATNet(
        build_esm(args.esm_model, args.esm_freeze_policy),
        prompts=args.num_prompts,
        dropout=args.dropout,
        sigma=args.sigma,
    ).to(device)

    loss = MaskedAsymmetricLoss(
        args.gamma_pos,
        args.gamma_neg,
        args.loss_clip,
    ).to(device)

    return model, loss


# Metrics
def metric_at_threshold(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "Threshold": float(threshold),
        "ACC": accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
    }


def best_threshold(y, p, thresholds):
    best = None

    for th in thresholds:
        m = metric_at_threshold(y, p, th)

        if best is None or (
            m["MCC"], m["Sensitivity"], m["Specificity"]
        ) > (
            best["MCC"], best["Sensitivity"], best["Specificity"]
        ):
            best = m

    return best


def evaluate_validation(labels, probs, step):
    result = {}
    thresholds = np.arange(0.30, 0.70 + 1e-12, step)

    for i, task in enumerate(TASKS):
        valid = labels[:, i] != -1
        y = labels[valid, i].astype(int)
        p = probs[valid, i]

        if np.unique(y).size < 2:
            raise ValueError(f"{task}: validation contains only one class.")

        result[task] = best_threshold(y, p, thresholds)

    result["Average_MCC"] = float(np.mean([result[t]["MCC"] for t in TASKS]))
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probs = [], []

    for batch in loader:
        ids, mask, phys, struct, center, y = to_device(batch, device)
        logits = model(ids, mask, phys, struct, center)
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(labels), np.concatenate(probs)


# Dataset check
def validate_data(csv_path, npz_dir):
    frame = pd.read_csv(csv_path)

    required = {"Dynamic_Sequence", "Core_Key", "Center_K_Index", "Fold", *LABELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame["Fold"] = pd.to_numeric(frame["Fold"], errors="raise").astype(int)

    if sorted(frame["Fold"].unique()) != [1, 2, 3, 4, 5]:
        raise ValueError("Fold must contain 1-5.")

    for col in LABELS:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
        if not set(frame[col].unique()).issubset({-1, 0, 1}):
            raise ValueError(f"Invalid labels in {col}.")

    if (frame[list(LABELS)] == -1).all(axis=1).any():
        raise ValueError("Rows with all labels masked were found.")

    if "Window_Group" in frame.columns:
        if (frame.groupby("Window_Group")["Fold"].nunique() > 1).any():
            raise ValueError("Window_Group crosses folds.")

    for row in frame.itertuples(index=False):
        seq = str(row.Dynamic_Sequence)
        center = int(row.Center_K_Index)
        key = str(row.Core_Key)

        if center < 0 or center >= len(seq) or seq[center].upper() != "K":
            raise ValueError(f"Invalid center K: {key}")

        if not (Path(npz_dir) / f"{key}.npz").is_file():
            raise FileNotFoundError(f"Missing NPZ: {key}.npz")

    print("\nLabel counts")
    for task, col in zip(TASKS, LABELS):
        print(
            f"{task}: +{(frame[col] == 1).sum()} "
            f"-{(frame[col] == 0).sum()} "
            f"masked={(frame[col] == -1).sum()}"
        )

    print("\nFold sizes")
    print(frame["Fold"].value_counts().sort_index().to_string())

    return frame


# Checkpoint
def save_checkpoint(path, model, loss_fn, fold, epoch, metrics, args):
    torch.save(
        {
            "model": model.state_dict(),
            "loss": loss_fn.state_dict(),
            "fold": fold,
            "best_epoch": epoch,
            "validation_metrics": metrics,
            "tasks": list(TASKS),
            "esm_model": args.esm_model,
            "esm_layers": 6,
            "esm_freeze_policy": args.esm_freeze_policy,
            "num_prompts": args.num_prompts,
            "dropout": args.dropout,
            "sigma": args.sigma,
            "gamma_pos": args.gamma_pos,
            "gamma_neg": args.gamma_neg,
            "loss_clip": args.loss_clip,
            "threshold_source": "validation only",
            "test_labels_used": False,
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# Fixed-fold training
def train_fold(frame, dataset, args, fold, device, output_dir, oof_probs):
    seed_all(args.seed + fold)

    train_idx = frame.index[frame["Fold"] != fold].to_numpy()
    val_idx = frame.index[frame["Fold"] == fold].to_numpy()

    g = torch.Generator().manual_seed(args.seed + fold)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model, loss_fn = build_model(args, device)

    optimizer = AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.learning_rate,
    )

    checkpoint = output_dir / "checkpoints" / f"katnet_3ptm_fold_{fold}.pth"

    best_mcc = -np.inf
    best_epoch = 0
    best_metrics = None
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_fn.train()
        loss_sum = 0.0

        for batch in train_loader:
            ids, mask, phys, struct, center, labels = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask, phys, struct, center)
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()

        val_labels, val_probs = predict(model, val_loader, device)
        metrics = evaluate_validation(val_labels, val_probs, args.fold_threshold_step)

        improved = metrics["Average_MCC"] > best_mcc

        if improved:
            best_mcc = metrics["Average_MCC"]
            best_epoch = epoch
            best_metrics = copy.deepcopy(metrics)
            wait = 0
            save_checkpoint(checkpoint, model, loss_fn, fold, epoch, metrics, args)
        else:
            wait += 1

        row = {
            "Fold": fold,
            "Epoch": epoch,
            "Loss": loss_sum / max(len(train_loader), 1),
            "Average_MCC": metrics["Average_MCC"],
        }

        for task in TASKS:
            for metric in ("MCC", "AUROC", "AUPRC", "Threshold"):
                row[f"{task}_{metric}"] = metrics[task][metric]

        history.append(row)

        print(
            f"Fold {fold} | Epoch {epoch:02d} | "
            f"Loss={row['Loss']:.4f} | MCC={metrics['Average_MCC']:.4f}"
            f"{' | saved' if improved else ''}"
        )

        if wait >= args.patience:
            print(f"Fold {fold} early stopped; best epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(
        output_dir / "logs" / f"fold_{fold}.csv",
        index=False,
    )

    ckpt = load_checkpoint(checkpoint, device)
    model.load_state_dict(ckpt["model"])
    loss_fn.load_state_dict(ckpt["loss"])

    labels, probs = predict(model, val_loader, device)

    expected = frame.loc[val_idx, list(LABELS)].to_numpy(dtype=np.float32)
    if not np.array_equal(labels, expected):
        raise RuntimeError("OOF row-order mismatch.")

    oof_probs[val_idx] = probs

    result = {
        "Fold": fold,
        "Train_N": len(train_idx),
        "Validation_N": len(val_idx),
        "Best_Epoch": best_epoch,
        "Best_Average_MCC": best_mcc,
    }

    for task in TASKS:
        result[f"{task}_MCC"] = best_metrics[task]["MCC"]
        result[f"{task}_AUROC"] = best_metrics[task]["AUROC"]
        result[f"{task}_AUPRC"] = best_metrics[task]["AUPRC"]

    del model, loss_fn, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# OOF thresholds and curves
def save_curves(labels, probs, output_dir):
    roc_rows, pr_rows = [], []

    for i, task in enumerate(TASKS):
        valid = labels[:, i] != -1
        y = labels[valid, i].astype(int)
        p = probs[valid, i]

        fpr, tpr, th = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        for a, b, c in zip(fpr, tpr, th):
            roc_rows.append(
                {"Task": task, "FPR": a, "TPR": b, "Threshold": c, "AUROC": auc}
            )

        precision, recall, th = precision_recall_curve(y, p)
        th = np.append(th, np.nan)
        auprc = average_precision_score(y, p)

        for a, b, c in zip(precision, recall, th):
            pr_rows.append(
                {
                    "Task": task,
                    "Precision": a,
                    "Recall": b,
                    "Threshold": c,
                    "AUPRC": auprc,
                }
            )

    pd.DataFrame(roc_rows).to_csv(output_dir / "oof_roc_curve.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "oof_pr_curve.csv", index=False)


def finalize_oof(frame, labels, probs, args, output_dir):
    if np.isnan(probs[labels != -1]).any():
        raise RuntimeError("Incomplete OOF predictions.")

    grid = np.arange(args.oof_threshold_step, 1.0, args.oof_threshold_step)
    metrics_rows = []

    payload = {
        "source": "pooled out-of-fold validation only",
        "criterion": "maximum MCC",
        "test_labels_used": False,
        "tasks": {},
    }

    for i, task in enumerate(TASKS):
        valid = labels[:, i] != -1
        y = labels[valid, i].astype(int)
        p = probs[valid, i]

        metrics = best_threshold(y, p, grid)
        payload["tasks"][task] = {"threshold": metrics["Threshold"]}
        metrics_rows.append({"Task": task, **metrics})

    out = frame[["Core_Key", "Fold", *LABELS]].copy()

    for i, task in enumerate(TASKS):
        out[f"{task}_OOF_Probability"] = probs[:, i]

    out.to_csv(output_dir / "oof_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "oof_metrics.csv", index=False)

    (output_dir / "validation_thresholds.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    save_curves(labels, probs, output_dir)

    return metrics_df, payload


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="KAT-Net fixed-fold training")

    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/katnet"))

    p.add_argument(
        "--esm-model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model ID or local model directory.",
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-prompts", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=3.0)

    p.add_argument("--gamma-pos", type=float, default=0.0)
    p.add_argument("--gamma-neg", type=float, default=2.0)
    p.add_argument("--loss-clip", type=float, default=0.05)

    p.add_argument("--fold-threshold-step", type=float, default=0.05)
    p.add_argument("--oof-threshold-step", type=float, default=0.001)

    p.add_argument(
        "--esm-freeze-policy",
        choices=("embedding_only", "all", "none"),
        default="embedding_only",
    )

    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    if not args.csv_path.is_file():
        raise FileNotFoundError(args.csv_path)

    if not args.npz_dir.is_dir():
        raise FileNotFoundError(args.npz_dir)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; use --overwrite."
            )
        shutil.rmtree(args.output_dir)

    (args.output_dir / "checkpoints").mkdir(parents=True)
    (args.output_dir / "logs").mkdir()

    frame = validate_data(args.csv_path, args.npz_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    dataset = KATDataset(args.csv_path, args.npz_dir, tokenizer)

    oof_labels = frame[list(LABELS)].to_numpy(dtype=np.float32)
    oof_probs = np.full((len(frame), N_TASKS), np.nan, dtype=np.float64)

    print("\nKAT-Net fixed-fold training")
    print(f"Device     : {device}")
    print(f"Samples    : {len(frame)}")
    print("Folds      : fixed 1-5")
    print("Tasks      : Kcr, Ksucc, Kac")
    print("Thresholds : pooled OOF validation only\n")

    fold_results = []

    for fold in range(1, 6):
        print(f"========== Fold {fold}/5 ==========")
        fold_results.append(
            train_fold(
                frame,
                dataset,
                args,
                fold,
                device,
                args.output_dir,
                oof_probs,
            )
        )

    pd.DataFrame(fold_results).to_csv(
        args.output_dir / "fold_summary.csv",
        index=False,
    )

    metrics, thresholds = finalize_oof(
        frame,
        oof_labels,
        oof_probs,
        args,
        args.output_dir,
    )

    manifest = {
        "model": "KAT-Net",
        "tasks": list(TASKS),
        "folds": [1, 2, 3, 4, 5],
        "esm_model": args.esm_model,
        "esm_layers": 6,
        "esm_freeze_policy": args.esm_freeze_policy,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "threshold_source": "pooled OOF validation only",
        "test_labels_used": False,
        "thresholds": {
            task: thresholds["tasks"][task]["threshold"]
            for task in TASKS
        },
    }

    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nPooled OOF metrics:")
    print(metrics.to_string(index=False))
    print(f"\nOutput: {args.output_dir}")
    print("[Status] PASS")


if __name__ == "__main__":
    main()
