# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, EsmModel
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
import math

warnings.filterwarnings("ignore")


# ============ 1. Global configuration ============
class Config:
    CSV_PATH = ""
    NPZ_DIR = ""
    ESM_PATH = ""

    TASKS = ['Kcr', 'Ksucc', 'Kac']
    NUM_TASKS = len(TASKS)
    NUM_PROMPTS = 3
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 70
    PATIENCE = 15
    DROPOUT = 0.3
    K_FOLDS = 5
    LPE_DIM = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ 2. Dataset and collation ============
AA_PHYS_PROPS = {
    'A': [1.8, 0.0, 0.0, 0.0], 'R': [-4.5, 1.0, 1.0, 0.0], 'N': [-3.5, 0.0, 0.2, 0.0],
    'D': [-3.5, -1.0, 0.2, 0.0], 'C': [2.5, 0.0, 0.1, 0.0], 'Q': [-3.5, 0.0, 0.5, 0.0],
    'E': [-3.5, -1.0, 0.5, 0.0], 'G': [-0.4, 0.0, -1.0, 0.0], 'H': [-3.2, 0.5, 0.5, 1.0],
    'I': [4.5, 0.0, 0.6, 0.0], 'L': [3.8, 0.0, 0.6, 0.0], 'K': [-3.9, 1.0, 0.8, 0.0],
    'M': [1.9, 0.0, 0.6, 0.0], 'F': [2.8, 0.0, 0.8, 1.0], 'P': [-1.6, 0.0, 0.1, 0.0],
    'S': [-0.8, 0.0, -0.5, 0.0], 'T': [-0.7, 0.0, 0.1, 0.0], 'W': [-0.9, 0.0, 1.5, 1.0],
    'Y': [-1.3, 0.0, 1.0, 1.0], 'V': [4.2, 0.0, 0.4, 0.0], 'X': [0.0, 0.0, 0.0, 0.0]
}


def get_physicochemical_matrix(seq):
    return torch.tensor([AA_PHYS_PROPS.get(aa, AA_PHYS_PROPS['X']) for aa in seq.upper()], dtype=torch.float32)


class DynamicMTLDataset(Dataset):
    def __init__(self, csv_path, npz_dir, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.npz_dir = npz_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['Dynamic_Sequence'])
        core_key = str(row['Core_Key'])
        center_k_idx = int(row['Center_K_Index'])

        # Task labels; -1 indicates unavailable labels
        labels = [row['Kcr_Label'], row['Ksucc_Label'], row['Kac_Label']]

        encoded = self.tokenizer(seq, return_tensors='pt', truncation=False, padding=False)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        phys_matrix = get_physicochemical_matrix(seq)

        npz_path = os.path.join(self.npz_dir, f"{core_key}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            struct_dict = {
                'coords': torch.from_numpy(data['coords']).float(),
                'plddt': torch.from_numpy(data['plddt']).float(),
                'sasa': torch.from_numpy(data['sasa']).float(),
                'ss': torch.from_numpy(data['ss']).float(),
                'pae': torch.from_numpy(data['pae']).float(),
                'disto': torch.from_numpy(data['disto']).float()
            }
        else:
            L = len(seq)
            struct_dict = {
                'coords': torch.zeros(L, 3), 'plddt': torch.zeros(L),
                'sasa': torch.zeros(L), 'ss': torch.zeros(L, 3),
                'pae': torch.full((L, L), 30.0), 'disto': torch.zeros(L, L, 64)
            }

        return input_ids, attention_mask, phys_matrix, struct_dict, center_k_idx, torch.tensor(labels,
                                                                                               dtype=torch.float32)


def dynamic_collate_fn(batch):
    ids_list, masks_list, phys_list, struct_dicts, center_indices, labels = zip(*batch)
    max_id_len = max([len(ids) for ids in ids_list])
    max_seq_len = max([len(phys) for phys in phys_list])
    B = len(batch)

    padded_ids = torch.zeros((B, max_id_len), dtype=torch.long)
    padded_masks = torch.zeros((B, max_id_len), dtype=torch.long)
    padded_phys = torch.zeros((B, max_seq_len, 4), dtype=torch.float32)
    b_coords = torch.zeros((B, max_seq_len, 3), dtype=torch.float32)
    b_plddt = torch.zeros((B, max_seq_len), dtype=torch.float32)
    b_sasa = torch.zeros((B, max_seq_len), dtype=torch.float32)
    b_ss = torch.zeros((B, max_seq_len, 3), dtype=torch.float32)
    b_pae = torch.full((B, max_seq_len, max_seq_len), 30.0, dtype=torch.float32)
    b_disto = torch.zeros((B, max_seq_len, max_seq_len, 64), dtype=torch.float32)

    for i in range(B):
        L_id = len(ids_list[i]);
        padded_ids[i, :L_id] = ids_list[i];
        padded_masks[i, :L_id] = masks_list[i]
        L_seq = len(phys_list[i]);
        padded_phys[i, :L_seq, :] = phys_list[i]
        b_coords[i, :L_seq, :] = struct_dicts[i]['coords'][:L_seq]
        b_plddt[i, :L_seq] = struct_dicts[i]['plddt'][:L_seq]
        b_sasa[i, :L_seq] = struct_dicts[i]['sasa'][:L_seq]
        b_ss[i, :L_seq, :] = struct_dicts[i]['ss'][:L_seq]
        b_pae[i, :L_seq, :L_seq] = struct_dicts[i]['pae'][:L_seq, :L_seq]
        b_disto[i, :L_seq, :L_seq, :] = struct_dicts[i]['disto'][:L_seq, :L_seq, :]

    return (padded_ids, padded_masks, padded_phys, (b_coords, b_plddt, b_sasa, b_ss, b_pae, b_disto),
            torch.tensor(center_indices, dtype=torch.long), torch.stack(labels))


# ============ 3. Feature extraction modules ============
class SoftPromptAdapter(nn.Module):
    def __init__(self, embed_dim=1280, num_prompts=3):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts * 2, embed_dim))
        nn.init.xavier_uniform_(self.prompt_embeddings)

    def forward(self, esm_embeddings):
        B = esm_embeddings.shape[0]
        prompts = self.prompt_embeddings.expand(B, -1, -1)
        return torch.cat([prompts[:, :self.num_prompts, :], esm_embeddings, prompts[:, self.num_prompts:, :]],
                         dim=1), self.num_prompts


class ChemoSpatialEmbedding(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.phys_adapter = nn.Sequential(nn.Linear(4, output_dim // 2), nn.GELU(),
                                          nn.Linear(output_dim // 2, output_dim))
        self.sigma = nn.Parameter(torch.tensor(3.0))
        self.rel_pos_emb = nn.Embedding(401, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, phys_matrix, center_indices, device):
        B, L, _ = phys_matrix.shape
        chem_feat = self.phys_adapter(phys_matrix)
        positions = torch.arange(L, device=device).float().unsqueeze(0).expand(B, -1)
        dist_sq = (positions - center_indices.unsqueeze(1).float()) ** 2
        spatial_weight = torch.exp(-dist_sq / (2 * self.sigma ** 2)).unsqueeze(-1)
        rel_indices = torch.clamp((positions - center_indices.unsqueeze(1).float() + 200).long(), 0, 400)
        return self.norm(chem_feat * spatial_weight + self.rel_pos_emb(rel_indices))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_inner / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_proj, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.activation(self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2))
        dt, B_ssm, C_ssm = torch.split(self.x_proj(x_conv), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(L):
            dt_i = dt[:, i, :].unsqueeze(-1)
            h = torch.exp(dt_i * A) * h + (dt_i * B_ssm[:, i, :].unsqueeze(1)) * x_conv[:, i, :].unsqueeze(-1)
            ys.append(torch.sum(h * C_ssm[:, i, :].unsqueeze(1), dim=-1))
        return self.dropout(self.out_proj((torch.stack(ys, dim=1) + x_conv * self.D) * F.silu(z))) + x


class CenterAnchoredMamba(nn.Module):
    def __init__(self, esm_dim=1280, csre_dim=64, hidden_dim=64, num_layers=2):
        super().__init__()
        self.fusion_proj = nn.Linear(esm_dim + csre_dim, hidden_dim)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, expand=2, dropout=0.2) for _ in range(num_layers)])
        self.norm_final = nn.LayerNorm(hidden_dim)

    def forward(self, esm_emb, csre_emb, prompt_len):
        B, L_seq, _ = csre_emb.shape
        csre_padded = F.pad(csre_emb, (0, 0, prompt_len, prompt_len), "constant", 0)
        x = self.norm_fusion(self.fusion_proj(torch.cat([esm_emb, csre_padded], dim=-1)))
        for layer in self.layers: x = (layer(x) + layer(x.flip(1)).flip(1)) / 2.0
        return self.norm_final(x[:, prompt_len:prompt_len + L_seq, :])


class EGNN_Layer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_mlp = nn.Sequential(nn.Linear(node_dim + hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, node_dim))
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, h, x, edge_attr, pae_weight):
        B, N, _ = h.shape
        coord_diff = x.unsqueeze(2) - x.unsqueeze(1)
        radial = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)
        h_i, h_j = h.unsqueeze(2).expand(B, N, N, -1), h.unsqueeze(1).expand(B, N, N, -1)
        m_ij = self.edge_mlp(torch.cat([h_i, h_j, radial, edge_attr], dim=-1)) * pae_weight
        m_ij_weighted = m_ij * self.att_mlp(m_ij)
        return h + self.node_mlp(torch.cat([h, torch.sum(m_ij_weighted, dim=2)], dim=-1)), x + torch.sum(
            coord_diff * self.coord_mlp(m_ij), dim=2)


class GeometricStructureEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.node_embedding = nn.Linear(in_dim + Config.LPE_DIM, hidden_dim)
        self.disto_embedding = nn.Linear(64, 16)
        self.pae_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = nn.ModuleList([EGNN_Layer(hidden_dim, 16, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def compute_laplacian_pe(self, coords, k=Config.LPE_DIM):
        B, N, _ = coords.shape
        dist = torch.cdist(coords, coords)
        _, indices = torch.topk(dist, k=7, largest=False)
        adj = torch.zeros(B, N, N, device=coords.device)
        adj[torch.arange(B, device=coords.device).view(-1, 1, 1), torch.arange(N, device=coords.device).view(1, -1,
                                                                                                             1), indices] = 1.0
        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj[adj > 0] = 1.0
        d_mat_inv_sqrt = torch.diag_embed(torch.pow(adj.sum(dim=-1) + 1e-6, -0.5))
        L = torch.eye(N, device=coords.device).unsqueeze(0) - torch.matmul(torch.matmul(d_mat_inv_sqrt, adj),
                                                                           d_mat_inv_sqrt)
        try:
            _, eigvecs = torch.linalg.eigh(L + 1e-6 * torch.eye(N, device=coords.device).unsqueeze(0))
        except:
            return torch.zeros(B, N, k, device=coords.device)
        lpe = eigvecs[:, :, 1:k + 1]
        return torch.cat([lpe, torch.zeros(B, N, k - lpe.shape[-1], device=coords.device)], dim=-1) if lpe.shape[
                                                                                                           -1] < k else lpe

    def forward(self, struct_data):
        coords, plddt, sasa, ss, pae, disto = struct_data
        h_in = torch.cat([plddt.unsqueeze(-1), sasa.unsqueeze(-1), ss, self.compute_laplacian_pe(coords)], dim=-1)
        h, x, edge_attr = F.gelu(self.node_embedding(h_in)), coords, F.gelu(self.disto_embedding(disto))
        pae_weight = torch.exp(-pae * torch.abs(self.pae_scale)).unsqueeze(-1)
        for layer in self.layers: h, x = layer(h, x, edge_attr, pae_weight)
        return self.norm(h)


# ============ 4. Task-driven hyperbolic routing ============
class TaskDrivenHyperbolicRouting(nn.Module):
    def __init__(self, dim, num_tasks=Config.NUM_TASKS, c_init=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_anchors = nn.Parameter(torch.randn(num_tasks, dim))
        nn.init.xavier_uniform_(self.task_anchors)

        self.c_raw = nn.Parameter(torch.tensor([np.log(np.exp(c_init) - 1)], dtype=torch.float32))
        self.geo_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.W_q = nn.Linear(dim, dim);
        self.W_k = nn.Linear(dim, dim);
        self.W_v = nn.Linear(dim, dim)
        self.proj_struct = nn.Linear(dim, dim)

        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion_out = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.2))

    @property
    def c(self): return (F.softplus(self.c_raw) + 1e-5).float()

    def hyperbolic_dist(self, x, y):
        x_norm_sq, y_norm_sq = x.norm(dim=-1, keepdim=True).pow(2), y.norm(dim=-1, keepdim=True).pow(2)
        dist_sq = (x - y).norm(dim=-1, keepdim=True).pow(2)
        denom = torch.clamp((1 - self.c * x_norm_sq) * (1 - self.c * y_norm_sq), min=1e-5)
        lhs = torch.clamp(1 + self.c * (2 * dist_sq / denom), min=1.0 + 1e-6)
        return (1.0 / torch.sqrt(self.c)) * torch.log(lhs + torch.sqrt(lhs.pow(2) - 1))

    def forward(self, h_seq, h_struct, center_indices):
        B, L, D = h_seq.shape
        batch_idx = torch.arange(B, device=h_seq.device)

        z_struct = torch.tanh(
            torch.sqrt(self.c) * torch.clamp(self.proj_struct(h_struct).norm(dim=-1, keepdim=True), min=1e-5)) / (
                               torch.sqrt(self.c) * torch.clamp(self.proj_struct(h_struct).norm(dim=-1, keepdim=True),
                                                                min=1e-5)) * self.proj_struct(h_struct)
        center_z = z_struct[batch_idx, center_indices, :].unsqueeze(1)
        d_hyp = self.hyperbolic_dist(z_struct, center_z)

        center_seq_feat = h_seq[batch_idx, center_indices, :].unsqueeze(1)
        task_queries = center_seq_feat + self.task_anchors.unsqueeze(0)

        Q = self.W_q(task_queries)
        K = self.W_k(h_struct)
        V = self.W_v(h_struct)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = F.softmax(attn_logits - d_hyp.transpose(1, 2) * self.geo_scale, dim=-1)
        context_struct = torch.matmul(attn_weights, V)

        combined = torch.cat([task_queries, context_struct], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * task_queries + (1 - gate) * context_struct

        return self.fusion_out(torch.cat([task_queries, fused], dim=-1)) + task_queries


# ============ 5. Multi-task model architecture ============
class UniversalTDRHNet(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.esm2 = esm_model
        self.prompt_adapter = SoftPromptAdapter(embed_dim=1280, num_prompts=Config.NUM_PROMPTS)
        self.csre_embedding = ChemoSpatialEmbedding(output_dim=64)
        self.seq_encoder = CenterAnchoredMamba(esm_dim=1280, csre_dim=64, hidden_dim=64, num_layers=2)
        self.struct_encoder = GeometricStructureEncoder(in_dim=5, hidden_dim=64, num_layers=2)
        self.fusion_module = TaskDrivenHyperbolicRouting(dim=64, num_tasks=Config.NUM_TASKS)

        # Task-specific classification heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(Config.DROPOUT), nn.Linear(32, 1))
            for _ in range(Config.NUM_TASKS)
        ])

    def forward(self, ids, mask, phys_matrix, struct_data, center_indices):
        raw_esm_emb = self.esm2(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.training: raw_esm_emb = raw_esm_emb + torch.randn_like(raw_esm_emb) * 0.002

        max_seq_len = phys_matrix.shape[1]
        raw_esm_emb = raw_esm_emb[:, 1:max_seq_len + 1, :]

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


# ============ 6. Masked asymmetric focal loss ============
class MaskedAsymmetricLoss(nn.Module):
    """Masked asymmetric focal loss for multi-task PTM prediction."""

    def __init__(self, num_tasks=Config.NUM_TASKS, gamma_pos=0.0, gamma_neg=2.0, clip=0.05):
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip  # Margin for negative samples

        # Learnable task uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, logits, targets):
        total_loss = 0.0

        # Clamp task uncertainty parameters
        clamped_log_vars = torch.clamp(self.log_vars, min=-1.5, max=2.0)

        for t in range(self.num_tasks):
            target_t = targets[:, t]
            valid_mask = (target_t != -1)

            if valid_mask.sum() == 0:
                continue

            valid_logits = logits[valid_mask, t]
            valid_targets = target_t[valid_mask]

            # ASL probabilities
            probs = torch.sigmoid(valid_logits)

            # Positive loss
            pos_loss = -valid_targets * (1 - probs).pow(self.gamma_pos) * torch.log(probs + 1e-8)

            # Negative loss with probability margin
            probs_neg = torch.clamp(probs - self.clip, min=0.0)
            neg_loss = -(1 - valid_targets) * probs_neg.pow(self.gamma_neg) * torch.log(1 - probs + 1e-8)

            task_loss = (pos_loss + neg_loss).mean()

            # Task uncertainty weighting
            precision = torch.exp(-clamped_log_vars[t])
            loss_t_weighted = 0.5 * precision * task_loss + clamped_log_vars[t]
            total_loss += loss_t_weighted

        return total_loss


# ============ 7. Training pipeline (5-fold CV) ============
def main():
    print(f"Device: {Config.DEVICE}")
    print("Model: KAT-Net (masked asymmetric loss)")

    tokenizer = AutoTokenizer.from_pretrained(Config.ESM_PATH)
    full_dataset = DynamicMTLDataset(Config.CSV_PATH, Config.NPZ_DIR, tokenizer)
    kfold = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)

    cv_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n>>> FOLD {fold + 1}/{Config.K_FOLDS}")

        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        train_loader = DataLoader(train_sub, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate_fn)
        val_loader = DataLoader(val_sub, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)

        esm2 = EsmModel.from_pretrained(Config.ESM_PATH)
        for param in esm2.embeddings.parameters(): param.requires_grad = False
        esm2.encoder.layer = esm2.encoder.layer[:6]

        model = UniversalTDRHNet(esm2).to(Config.DEVICE)
        criterion = MaskedAsymmetricLoss().to(Config.DEVICE)

        optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=Config.LR)

        best_fold_avg_mcc = -1
        best_fold_metrics = {}
        patience_counter = 0

        for ep in range(Config.EPOCHS):
            model.train()
            total_loss = 0
            for ids, mask, phys, struct_data, c_idx, lbl in tqdm(train_loader, leave=False,
                                                                 desc=f"Fold {fold + 1} Ep {ep + 1} Train"):
                ids, mask, phys, c_idx = ids.to(Config.DEVICE), mask.to(Config.DEVICE), phys.to(
                    Config.DEVICE), c_idx.to(Config.DEVICE)
                lbl = lbl.to(Config.DEVICE)
                s_gpu = tuple(t.to(Config.DEVICE) for t in struct_data)

                optimizer.zero_grad()
                logits, _ = model(ids, mask, phys, s_gpu, c_idx)

                # Masked ASL loss
                loss = criterion(logits, lbl)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_y_true = {i: [] for i in range(Config.NUM_TASKS)}
            val_y_prob = {i: [] for i in range(Config.NUM_TASKS)}

            with torch.no_grad():
                for ids, mask, phys, struct_data, c_idx, lbl in tqdm(val_loader, leave=False, desc="Eval"):
                    ids, mask, phys, c_idx = ids.to(Config.DEVICE), mask.to(Config.DEVICE), phys.to(
                        Config.DEVICE), c_idx.to(Config.DEVICE)
                    s_gpu = tuple(t.to(Config.DEVICE) for t in struct_data)

                    logits, _ = model(ids, mask, phys, s_gpu, c_idx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    targets = lbl.numpy()

                    for i in range(Config.NUM_TASKS):
                        for b in range(targets.shape[0]):
                            if targets[b, i] != -1:
                                val_y_true[i].append(targets[b, i])
                                val_y_prob[i].append(probs[b, i])

            metrics_str = ""
            current_ep_mcc_sum = 0
            valid_tasks = 0
            current_metrics = {}

            for i, ptm_name in enumerate(Config.TASKS):
                if len(val_y_true[i]) > 0:
                    try:
                        auc = roc_auc_score(val_y_true[i], val_y_prob[i])
                    except:
                        auc = 0.5

                    best_th_mcc = -1
                    for th in np.arange(0.3, 0.7, 0.05):
                        preds = [1 if p > th else 0 for p in val_y_prob[i]]
                        mcc = matthews_corrcoef(val_y_true[i], preds)
                        if mcc > best_th_mcc: best_th_mcc = mcc

                    current_metrics[ptm_name] = {'MCC': best_th_mcc, 'AUC': auc}
                    metrics_str += f"| {ptm_name} (MCC:{best_th_mcc:.3f}) "
                    current_ep_mcc_sum += best_th_mcc
                    valid_tasks += 1

            avg_mcc = current_ep_mcc_sum / valid_tasks if valid_tasks > 0 else 0

            # Task weights
            task_weights_str = " | ".join(
                [f"W_{name}:{torch.exp(-torch.clamp(criterion.log_vars[i], -1.5, 2.0)).item():.2f}" for i, name in
                 enumerate(Config.TASKS)])

            if avg_mcc > best_fold_avg_mcc:
                best_fold_avg_mcc = avg_mcc
                best_fold_metrics = current_metrics
                patience_counter = 0
                torch.save({'model': model.state_dict(), 'loss': criterion.state_dict()},
                           f'ptmbest_model_fold_{fold + 1}.pth')
                stat_str = "Saved *"
            else:
                patience_counter += 1
                stat_str = ""

            print(
                f"Ep {ep + 1:2d} | Loss: {total_loss / len(train_loader):.4f} | Avg MCC: {avg_mcc:.4f} {metrics_str} | {task_weights_str} {stat_str}")

            if patience_counter >= Config.PATIENCE:
                break

        cv_results.append(best_fold_metrics)
        print(f"Fold {fold + 1} best Avg MCC: {best_fold_avg_mcc:.4f}")

    print("\n" + "=" * 60)
    print("5-Fold Cross Validation Final Results")
    print("=" * 60)

    final_avg = {ptm: {'MCC': 0, 'AUC': 0} for ptm in Config.TASKS}
    for res in cv_results:
        for ptm in Config.TASKS:
            if ptm in res:
                final_avg[ptm]['MCC'] += res[ptm]['MCC']
                final_avg[ptm]['AUC'] += res[ptm]['AUC']

    overall_mcc = 0
    for ptm in Config.TASKS:
        m_mcc = final_avg[ptm]['MCC'] / Config.K_FOLDS
        m_auc = final_avg[ptm]['AUC'] / Config.K_FOLDS
        overall_mcc += m_mcc
        print(f"{ptm:>6} | Avg MCC: {m_mcc:.4f} | Avg AUC: {m_auc:.4f}")

    print("-" * 60)
    print(f"OVERALL MODEL AVG MCC: {overall_mcc / Config.NUM_TASKS:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()