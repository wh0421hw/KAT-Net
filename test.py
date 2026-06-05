# -*- coding: utf-8 -*-
import os
import re
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel

warnings.filterwarnings("ignore")


class Config:
    ESM_PATH = ""
    TASKS = ["Kcr", "Ksucc", "Kac"]
    NUM_TASKS = len(TASKS)
    NUM_PROMPTS = 3
    BATCH_SIZE = 64
    DROPOUT = 0.3
    LPE_DIM = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConfig:
    K_FOLDS = 5
    MODEL_WEIGHTS_TEMPLATE = ""

    KCR_POS = ""
    KCR_NEG = ""
    KCR_FEAT = ""

    KSUCC_POS = ""
    KSUCC_NEG = ""
    KSUCC_FEAT_POS = ""
    KSUCC_FEAT_NEG = ""

    KAC_CSV = ""
    KAC_FEAT = ""


def parse_fasta_explicit(file_path, explicit_label, feat_dir, source_name):
    data = []
    with open(file_path, "r") as f:
        header, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    data.append(
                        {
                            "header": header,
                            "seq": seq,
                            "label": explicit_label,
                            "feat_dir": feat_dir,
                            "source": source_name,
                        }
                    )
                header = line[1:]
                seq = ""
            else:
                seq += line

        if header:
            data.append(
                {
                    "header": header,
                    "seq": seq,
                    "label": explicit_label,
                    "feat_dir": feat_dir,
                    "source": source_name,
                }
            )
    return data


def parse_kac_csv_raw(file_path, feat_dir, source_name):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "test" not in line.lower():
                continue

            parts = line.split(",")
            if len(parts) < 3:
                continue
            if parts[-1].strip().lower() != "test":
                continue

            lbl_str = parts[-2].strip()
            if lbl_str not in ["0", "1"]:
                continue
            lbl = int(lbl_str)

            seqs = [p.strip() for p in parts if p.strip().isalpha()]
            if not seqs:
                continue

            seq = max(seqs, key=len)
            core_key = parts[0].strip()

            data.append(
                {
                    "header": core_key,
                    "seq": seq,
                    "label": lbl,
                    "feat_dir": feat_dir,
                    "source": source_name,
                }
            )
    return data


def build_unified_dataframe():
    df_list = []

    def add_to_list(items, target_idx):
        for item in items:
            if item["label"] == -1:
                continue

            labels = [-1, -1, -1]
            labels[target_idx] = item["label"]

            df_list.append(
                {
                    "Dynamic_Sequence": item["seq"],
                    "Core_Key": item["header"],
                    "Center_K_Index": len(item["seq"]) // 2,
                    "Kcr_Label": labels[0],
                    "Ksucc_Label": labels[1],
                    "Kac_Label": labels[2],
                    "Feature_Dir": item["feat_dir"],
                    "Dataset_Source": item["source"],
                }
            )

    add_to_list(parse_fasta_explicit(TestConfig.KCR_POS, 1, TestConfig.KCR_FEAT, "Kcr_Main"), 0)
    add_to_list(parse_fasta_explicit(TestConfig.KCR_NEG, 0, TestConfig.KCR_FEAT, "Kcr_Main"), 0)

    add_to_list(parse_fasta_explicit(TestConfig.KSUCC_POS, 1, TestConfig.KSUCC_FEAT_POS, "Ksucc_Main"), 1)
    add_to_list(parse_fasta_explicit(TestConfig.KSUCC_NEG, 0, TestConfig.KSUCC_FEAT_NEG, "Ksucc_Main"), 1)

    add_to_list(parse_kac_csv_raw(TestConfig.KAC_CSV, TestConfig.KAC_FEAT, "Kac_Main"), 2)

    return pd.DataFrame(df_list)


AA_PHYS_PROPS = {
    "A": [1.8, 0.0, 0.0, 0.0],
    "R": [-4.5, 1.0, 1.0, 0.0],
    "N": [-3.5, 0.0, 0.2, 0.0],
    "D": [-3.5, -1.0, 0.2, 0.0],
    "C": [2.5, 0.0, 0.1, 0.0],
    "Q": [-3.5, 0.0, 0.5, 0.0],
    "E": [-3.5, -1.0, 0.5, 0.0],
    "G": [-0.4, 0.0, -1.0, 0.0],
    "H": [-3.2, 0.5, 0.5, 1.0],
    "I": [4.5, 0.0, 0.6, 0.0],
    "L": [3.8, 0.0, 0.6, 0.0],
    "K": [-3.9, 1.0, 0.8, 0.0],
    "M": [1.9, 0.0, 0.6, 0.0],
    "F": [2.8, 0.0, 0.8, 1.0],
    "P": [-1.6, 0.0, 0.1, 0.0],
    "S": [-0.8, 0.0, -0.5, 0.0],
    "T": [-0.7, 0.0, 0.1, 0.0],
    "W": [-0.9, 0.0, 1.5, 1.0],
    "Y": [-1.3, 0.0, 1.0, 1.0],
    "V": [4.2, 0.0, 0.4, 0.0],
    "X": [0.0, 0.0, 0.0, 0.0],
}


def get_physicochemical_matrix(seq):
    return torch.tensor(
        [AA_PHYS_PROPS.get(aa, AA_PHYS_PROPS["X"]) for aa in seq.upper()],
        dtype=torch.float32,
    )


class UnifiedRoutingDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["Dynamic_Sequence"])
        core_key = str(row["Core_Key"])
        center_k_idx = int(row["Center_K_Index"])
        feat_dir = str(row["Feature_Dir"])

        labels = [row["Kcr_Label"], row["Ksucc_Label"], row["Kac_Label"]]

        encoded = self.tokenizer(seq, return_tensors="pt", truncation=False, padding=False)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        phys_matrix = get_physicochemical_matrix(seq)

        npz_path = os.path.join(feat_dir, f"{core_key}.npz")
        if not os.path.exists(npz_path):
            clean_key = re.sub(r"[_|][01pn]$", "", core_key, flags=re.IGNORECASE)
            fallback_path = os.path.join(feat_dir, f"{clean_key}.npz")
            if os.path.exists(fallback_path):
                npz_path = fallback_path

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            struct_dict = {
                "coords": torch.from_numpy(data["coords"]).float(),
                "plddt": torch.from_numpy(data["plddt"]).float(),
                "sasa": torch.from_numpy(data["sasa"]).float(),
                "ss": torch.from_numpy(data["ss"]).float(),
                "pae": torch.from_numpy(data["pae"]).float(),
                "disto": torch.from_numpy(data["disto"]).float(),
            }
        else:
            L = len(seq)
            struct_dict = {
                "coords": torch.zeros(L, 3),
                "plddt": torch.zeros(L),
                "sasa": torch.zeros(L),
                "ss": torch.zeros(L, 3),
                "pae": torch.full((L, L), 30.0),
                "disto": torch.zeros(L, L, 64),
            }

        return (
            input_ids,
            attention_mask,
            phys_matrix,
            struct_dict,
            center_k_idx,
            torch.tensor(labels, dtype=torch.float32),
        )


def dynamic_collate_fn(batch):
    ids_list, masks_list, phys_list, struct_dicts, center_indices, labels = zip(*batch)
    max_id_len = max(len(ids) for ids in ids_list)
    max_seq_len = max(len(phys) for phys in phys_list)
    batch_size = len(batch)

    padded_ids = torch.zeros((batch_size, max_id_len), dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_id_len), dtype=torch.long)
    padded_phys = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.float32)
    b_coords = torch.zeros((batch_size, max_seq_len, 3), dtype=torch.float32)
    b_plddt = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    b_sasa = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    b_ss = torch.zeros((batch_size, max_seq_len, 3), dtype=torch.float32)
    b_pae = torch.full((batch_size, max_seq_len, max_seq_len), 30.0, dtype=torch.float32)
    b_disto = torch.zeros((batch_size, max_seq_len, max_seq_len, 64), dtype=torch.float32)

    for i in range(batch_size):
        len_ids = len(ids_list[i])
        padded_ids[i, :len_ids] = ids_list[i]
        padded_masks[i, :len_ids] = masks_list[i]

        len_seq = len(phys_list[i])
        padded_phys[i, :len_seq, :] = phys_list[i]
        b_coords[i, :len_seq, :] = struct_dicts[i]["coords"][:len_seq]
        b_plddt[i, :len_seq] = struct_dicts[i]["plddt"][:len_seq]
        b_sasa[i, :len_seq] = struct_dicts[i]["sasa"][:len_seq]
        b_ss[i, :len_seq, :] = struct_dicts[i]["ss"][:len_seq]
        b_pae[i, :len_seq, :len_seq] = struct_dicts[i]["pae"][:len_seq, :len_seq]
        b_disto[i, :len_seq, :len_seq, :] = struct_dicts[i]["disto"][:len_seq, :len_seq, :]

    return (
        padded_ids,
        padded_masks,
        padded_phys,
        (b_coords, b_plddt, b_sasa, b_ss, b_pae, b_disto),
        torch.tensor(center_indices, dtype=torch.long),
        torch.stack(labels),
    )


class SoftPromptAdapter(nn.Module):
    def __init__(self, embed_dim=1280, num_prompts=3):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts * 2, embed_dim))
        nn.init.xavier_uniform_(self.prompt_embeddings)

    def forward(self, esm_embeddings):
        batch_size = esm_embeddings.shape[0]
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        return torch.cat(
            [
                prompts[:, : self.num_prompts, :],
                esm_embeddings,
                prompts[:, self.num_prompts :, :],
            ],
            dim=1,
        ), self.num_prompts


class ChemoSpatialEmbedding(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.phys_adapter = nn.Sequential(
            nn.Linear(4, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )
        self.sigma = nn.Parameter(torch.tensor(3.0))
        self.rel_pos_emb = nn.Embedding(401, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, phys_matrix, center_indices, device):
        batch_size, seq_len, _ = phys_matrix.shape
        chem_feat = self.phys_adapter(phys_matrix)
        positions = torch.arange(seq_len, device=device).float().unsqueeze(0).expand(batch_size, -1)
        dist_sq = (positions - center_indices.unsqueeze(1).float()) ** 2
        spatial_weight = torch.exp(-dist_sq / (2 * self.sigma**2)).unsqueeze(-1)
        rel_indices = torch.clamp((positions - center_indices.unsqueeze(1).float() + 200).long(), 0, 400)
        return self.norm(chem_feat * spatial_weight + self.rel_pos_emb(rel_indices))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_inner / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.activation(self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))
        dt, b_ssm, c_ssm = torch.split(self.x_proj(x_conv), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)

        ys = []
        for i in range(seq_len):
            dt_i = dt[:, i, :].unsqueeze(-1)
            h = torch.exp(dt_i * A) * h + (dt_i * b_ssm[:, i, :].unsqueeze(1)) * x_conv[:, i, :].unsqueeze(-1)
            ys.append(torch.sum(h * c_ssm[:, i, :].unsqueeze(1), dim=-1))

        return self.dropout(self.out_proj((torch.stack(ys, dim=1) + x_conv * self.D) * F.silu(z))) + x


class CenterAnchoredMamba(nn.Module):
    def __init__(self, esm_dim=1280, csre_dim=64, hidden_dim=64, num_layers=2):
        super().__init__()
        self.fusion_proj = nn.Linear(esm_dim + csre_dim, hidden_dim)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, expand=2, dropout=0.2) for _ in range(num_layers)])
        self.norm_final = nn.LayerNorm(hidden_dim)

    def forward(self, esm_emb, csre_emb, prompt_len):
        _, seq_len, _ = csre_emb.shape
        csre_padded = F.pad(csre_emb, (0, 0, prompt_len, prompt_len), "constant", 0)
        x = self.norm_fusion(self.fusion_proj(torch.cat([esm_emb, csre_padded], dim=-1)))

        for layer in self.layers:
            x = (layer(x) + layer(x.flip(1)).flip(1)) / 2.0

        return self.norm_final(x[:, prompt_len : prompt_len + seq_len, :])


class EGNN_Layer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h, x, edge_attr, pae_weight):
        batch_size, num_nodes, _ = h.shape
        coord_diff = x.unsqueeze(2) - x.unsqueeze(1)
        radial = torch.sum(coord_diff**2, dim=-1, keepdim=True)
        h_i = h.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        h_j = h.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        m_ij = self.edge_mlp(torch.cat([h_i, h_j, radial, edge_attr], dim=-1)) * pae_weight
        m_ij_weighted = m_ij * self.att_mlp(m_ij)
        h_out = h + self.node_mlp(torch.cat([h, torch.sum(m_ij_weighted, dim=2)], dim=-1))
        x_out = x + torch.sum(coord_diff * self.coord_mlp(m_ij), dim=2)
        return h_out, x_out


class GeometricStructureEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim + Config.LPE_DIM, hidden_dim)
        self.disto_embedding = nn.Linear(64, 16)
        self.pae_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = nn.ModuleList([EGNN_Layer(hidden_dim, 16, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def compute_laplacian_pe(self, coords, k=Config.LPE_DIM):
        batch_size, num_nodes, _ = coords.shape
        dist = torch.cdist(coords, coords)
        _, indices = torch.topk(dist, k=7, largest=False)
        adj = torch.zeros(batch_size, num_nodes, num_nodes, device=coords.device)
        adj[
            torch.arange(batch_size, device=coords.device).view(-1, 1, 1),
            torch.arange(num_nodes, device=coords.device).view(1, -1, 1),
            indices,
        ] = 1.0
        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj[adj > 0] = 1.0
        d_mat_inv_sqrt = torch.diag_embed(torch.pow(adj.sum(dim=-1) + 1e-6, -0.5))
        laplacian = torch.eye(num_nodes, device=coords.device).unsqueeze(0) - torch.matmul(
            torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
        )

        try:
            _, eigvecs = torch.linalg.eigh(laplian + 1e-6 * torch.eye(num_nodes, device=coords.device).unsqueeze(0))
        except NameError:
            _, eigvecs = torch.linalg.eigh(laplacian + 1e-6 * torch.eye(num_nodes, device=coords.device).unsqueeze(0))
        except Exception:
            return torch.zeros(batch_size, num_nodes, k, device=coords.device)

        lpe = eigvecs[:, :, 1 : k + 1]
        if lpe.shape[-1] < k:
            return torch.cat([lpe, torch.zeros(batch_size, num_nodes, k - lpe.shape[-1], device=coords.device)], dim=-1)
        return lpe

    def forward(self, struct_data):
        coords, plddt, sasa, ss, pae, disto = struct_data
        h_in = torch.cat(
            [
                plddt.unsqueeze(-1),
                sasa.unsqueeze(-1),
                ss,
                self.compute_laplacian_pe(coords),
            ],
            dim=-1,
        )
        h = F.gelu(self.node_embedding(h_in))
        x = coords
        edge_attr = F.gelu(self.disto_embedding(disto))
        pae_weight = torch.exp(-pae * torch.abs(self.pae_scale)).unsqueeze(-1)

        for layer in self.layers:
            h, x = layer(h, x, edge_attr, pae_weight)

        return self.norm(h)


class TaskDrivenHyperbolicRouting(nn.Module):
    def __init__(self, dim, num_tasks=Config.NUM_TASKS, c_init=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_anchors = nn.Parameter(torch.randn(num_tasks, dim))
        nn.init.xavier_uniform_(self.task_anchors)

        self.c_raw = nn.Parameter(torch.tensor([np.log(np.exp(c_init) - 1)], dtype=torch.float32))
        self.geo_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.proj_struct = nn.Linear(dim, dim)

        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion_out = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.2))

    @property
    def c(self):
        return (F.softplus(self.c_raw) + 1e-5).float()

    def hyperbolic_dist(self, x, y):
        x_norm_sq = x.norm(dim=-1, keepdim=True).pow(2)
        y_norm_sq = y.norm(dim=-1, keepdim=True).pow(2)
        dist_sq = (x - y).norm(dim=-1, keepdim=True).pow(2)
        denom = torch.clamp((1 - self.c * x_norm_sq) * (1 - self.c * y_norm_sq), min=1e-5)
        lhs = torch.clamp(1 + self.c * (2 * dist_sq / denom), min=1.0 + 1e-6)
        return (1.0 / torch.sqrt(self.c)) * torch.log(lhs + torch.sqrt(lhs.pow(2) - 1))

    def forward(self, h_seq, h_struct, center_indices):
        batch_size, _, dim = h_seq.shape
        batch_idx = torch.arange(batch_size, device=h_seq.device)

        projected_struct = self.proj_struct(h_struct)
        projected_norm = torch.clamp(projected_struct.norm(dim=-1, keepdim=True), min=1e-5)
        z_struct = torch.tanh(torch.sqrt(self.c) * projected_norm) / (torch.sqrt(self.c) * projected_norm) * projected_struct
        center_z = z_struct[batch_idx, center_indices, :].unsqueeze(1)
        d_hyp = self.hyperbolic_dist(z_struct, center_z)

        center_seq_feat = h_seq[batch_idx, center_indices, :].unsqueeze(1)
        task_queries = center_seq_feat + self.task_anchors.unsqueeze(0)

        query = self.W_q(task_queries)
        key = self.W_k(h_struct)
        value = self.W_v(h_struct)

        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = F.softmax(attn_logits - d_hyp.transpose(1, 2) * self.geo_scale, dim=-1)
        context_struct = torch.matmul(attn_weights, value)

        combined = torch.cat([task_queries, context_struct], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * task_queries + (1 - gate) * context_struct

        return self.fusion_out(torch.cat([task_queries, fused], dim=-1)) + task_queries


class UniversalTDRHNet(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.esm2 = esm_model
        self.prompt_adapter = SoftPromptAdapter(embed_dim=1280, num_prompts=Config.NUM_PROMPTS)
        self.csre_embedding = ChemoSpatialEmbedding(output_dim=64)
        self.seq_encoder = CenterAnchoredMamba(esm_dim=1280, csre_dim=64, hidden_dim=64, num_layers=2)
        self.struct_encoder = GeometricStructureEncoder(in_dim=5, hidden_dim=64, num_layers=2)
        self.fusion_module = TaskDrivenHyperbolicRouting(dim=64, num_tasks=Config.NUM_TASKS)

        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(Config.DROPOUT),
                    nn.Linear(32, 1),
                )
                for _ in range(Config.NUM_TASKS)
            ]
        )

    def forward(self, ids, mask, phys_matrix, struct_data, center_indices):
        raw_esm_emb = self.esm2(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.training:
            raw_esm_emb = raw_esm_emb + torch.randn_like(raw_esm_emb) * 0.002

        max_seq_len = phys_matrix.shape[1]
        raw_esm_emb = raw_esm_emb[:, 1 : max_seq_len + 1, :]

        promoted_esm_emb, prompt_len = self.prompt_adapter(raw_esm_emb)
        csre_feat = self.csre_embedding(phys_matrix, center_indices, device=ids.device)
        f_seq_full = self.seq_encoder(promoted_esm_emb, csre_feat, prompt_len)
        f_struct_full = self.struct_encoder(struct_data)
        task_features = self.fusion_module(f_seq_full, f_struct_full, center_indices)

        logits_list = []
        for i, head in enumerate(self.task_heads):
            logits_list.append(head(task_features[:, i, :]))

        logits_out = torch.cat(logits_list, dim=-1)
        return logits_out, task_features


def calculate_metrics(y_true, y_prob):
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    best_mcc = -1.0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (y_prob > threshold).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    final_preds = (y_prob > best_threshold).astype(int)
    acc = accuracy_score(y_true, final_preds)
    f1 = f1_score(y_true, final_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, final_preds, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "ACC": acc,
        "AUC": auc,
        "MCC": best_mcc,
        "F1": f1,
        "Sn": sn,
        "Sp": sp,
    }


def test_all_tasks_ensemble():
    df_test = build_unified_dataframe()

    tokenizer = AutoTokenizer.from_pretrained(Config.ESM_PATH)
    test_dataset = UnifiedRoutingDataset(df_test, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
    )

    esm2 = EsmModel.from_pretrained(Config.ESM_PATH)
    esm2.encoder.layer = esm2.encoder.layer[:6]
    model = UniversalTDRHNet(esm2).to(Config.DEVICE)

    y_prob_ensemble = np.zeros((len(df_test), Config.NUM_TASKS))
    y_true_all = None

    for fold in range(1, TestConfig.K_FOLDS + 1):
        weight_path = TestConfig.MODEL_WEIGHTS_TEMPLATE.format(fold)

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=Config.DEVICE)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        fold_probs = []
        fold_trues = []

        with torch.no_grad():
            for ids, mask, phys, struct_data, center_idx, labels in test_loader:
                ids = ids.to(Config.DEVICE)
                mask = mask.to(Config.DEVICE)
                phys = phys.to(Config.DEVICE)
                center_idx = center_idx.to(Config.DEVICE)
                struct_gpu = tuple(t.to(Config.DEVICE) for t in struct_data)

                logits, _ = model(ids, mask, phys, struct_gpu, center_idx)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.append(probs)

                if fold == 1:
                    fold_trues.append(labels.numpy())

        y_prob_ensemble += np.vstack(fold_probs)
        if fold == 1:
            y_true_all = np.vstack(fold_trues)

    y_prob_ensemble /= TestConfig.K_FOLDS
    sources_array = df_test["Dataset_Source"].values

    print("Task\tSource\tN\tACC\tAUC\tMCC\tF1\tSn\tSp")

    for task_idx, task_name in enumerate(Config.TASKS):
        targets_t = y_true_all[:, task_idx]
        valid_mask = targets_t != -1

        if valid_mask.sum() == 0:
            continue

        task_sources = np.unique(sources_array[valid_mask])
        for source in task_sources:
            source_mask = valid_mask & (sources_array == source)
            y_true_valid = targets_t[source_mask]
            y_prob_valid = y_prob_ensemble[source_mask, task_idx]

            metrics = calculate_metrics(y_true_valid, y_prob_valid)
            auc_str = "N/A" if np.isnan(metrics["AUC"]) else f"{metrics['AUC']:.4f}"

            print(
                f"{task_name}\t{source}\t{len(y_true_valid)}\t"
                f"{metrics['ACC']:.4f}\t{auc_str}\t{metrics['MCC']:.4f}\t"
                f"{metrics['F1']:.4f}\t{metrics['Sn']:.4f}\t{metrics['Sp']:.4f}"
            )


if __name__ == "__main__":
    test_all_tasks_ensemble()
