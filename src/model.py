import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


DEFAULT_MOLDEBERTA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MolDeBERTa-small",
)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim, vocab_size=5, embed_dim=128, num_filters=64, filter_sizes=(2, 3, 4)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes])
        self.attn_pool = nn.ModuleList([nn.Conv1d(num_filters, 1, kernel_size=1) for _ in filter_sizes])
        self.res_proj = nn.Linear(embed_dim, output_dim)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(output_dim)

    @staticmethod
    def _same_pad(x, kernel_size):
        left = (kernel_size - 1) // 2
        right = kernel_size // 2
        return F.pad(x, (left, right))

    def forward(self, x):
        emb = self.embedding(x)
        x = emb.transpose(1, 2)
        pooled = []
        for conv, pool in zip(self.convs, self.attn_pool):
            h = F.gelu(conv(self._same_pad(x, conv.kernel_size[0])))
            score = pool(h).squeeze(1)
            alpha = torch.softmax(score, dim=-1).unsqueeze(1)
            attn_summary = (h * alpha).sum(dim=-1)
            max_summary = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            pooled.append(attn_summary + max_summary)
        h = torch.cat(pooled, dim=1)
        out = self.fc(self.dropout(h))
        res = self.res_proj(emb.mean(dim=1))
        return self.norm(out + res)


class RNAFMFeatureExtractor(nn.Module):
    def __init__(self, output_dim, model_layer=12):
        super().__init__()
        self.enabled = True
        self.init_error = None
        self.backend_name = "RNA-FM"
        try:
            import fm

            self.model, self.alphabet = fm.pretrained.rna_fm_t12()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.model_layer = model_layer
            self.proj = nn.Linear(640, output_dim)
            self.dropout = nn.Dropout(0.2)
        except Exception as exc:
            self.enabled = False
            self.init_error = exc
            self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, seqs, fallback_tensor):
        if not self.enabled:
            return self.proj(fallback_tensor)

        items = [(f"seq_{i}", s) for i, s in enumerate(seqs)]
        device = self.proj.weight.device
        _, _, tokens = self.batch_converter(items)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.model_layer])
            cls = out["representations"][self.model_layer][:, 0, :]

        return self.dropout(self.proj(cls))


class MolecularLMFeatureExtractor(nn.Module):
    def __init__(self, output_dim, model_dir=None):
        super().__init__()
        self.enabled = True
        self.init_error = None
        self.backend_name = "MolDeBERTa"
        if model_dir:
            self.source = model_dir
        else:
            self.source = DEFAULT_MOLDEBERTA_DIR
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.source)
            self.model = AutoModel.from_pretrained(self.source)
            self.max_length = int(getattr(self.model.config, "max_position_embeddings", 256))
            for p in self.model.parameters():
                p.requires_grad = False
            self.proj = nn.Linear(self.model.config.hidden_size, output_dim)
            self.dropout = nn.Dropout(0.2)
        except Exception as exc:
            self.enabled = False
            self.init_error = exc
            self.max_length = 256
            self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, smiles_list, fallback_tensor):
        if not self.enabled:
            return self.proj(fallback_tensor)

        device = self.proj.weight.device
        enc = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.dropout(self.proj(pooled))


class DrugGraphTransformerMol(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=128, heads=4):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        out_per_head = max(1, hidden_dim // heads)
        self.conv1 = TransformerConv(hidden_dim, out_per_head, heads=heads, edge_dim=edge_in_dim, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim, out_per_head, heads=heads, edge_dim=edge_in_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data_batch):
        x = self.node_proj(data_batch.x)
        h1 = self.conv1(x, data_batch.edge_index, data_batch.edge_attr)
        x = self.norm1(F.relu(h1) + x)
        h2 = self.conv2(x, data_batch.edge_index, data_batch.edge_attr)
        x = self.norm2(F.relu(h2) + x)
        g_mean = global_mean_pool(x, data_batch.batch)
        g_max = global_max_pool(x, data_batch.batch)
        return self.readout(torch.cat([g_mean, g_max], dim=-1))


class RelationAwareSpatialAttention(MessagePassing):
    def __init__(self, hidden_dim, num_relations, max_sp=3, dropout=0.1):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.max_sp = max_sp

        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.sp_emb = nn.Embedding(max_sp + 1, hidden_dim)

        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, edge_index, edge_type, edge_sp):
        edge_sp = edge_sp.clamp(max=self.max_sp)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        out = self.propagate(edge_index=edge_index, q=q, k=k, v=v, edge_type=edge_type, edge_sp=edge_sp)
        out = self.out_proj(out)
        x = self.norm(x + self.dropout(out))
        x = self.norm(x + self.ffn(x))
        return x

    def message(self, q_i, k_j, v_j, edge_type, edge_sp, index, ptr, size_i):
        r = self.rel_emb(edge_type)
        s = self.sp_emb(edge_sp)

        key = k_j + r
        value = v_j + r

        score = (q_i * key).sum(dim=-1) / math.sqrt(self.hidden_dim)
        score = score + (q_i * s).sum(dim=-1) / math.sqrt(self.hidden_dim)

        alpha = softmax(score, index, ptr, size_i)
        alpha = self.dropout(alpha)

        return value * alpha.unsqueeze(-1)


class HeteroRelationEncoder(nn.Module):
    def __init__(self, hidden_dim, num_relations, max_sp=3, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [RelationAwareSpatialAttention(hidden_dim, num_relations, max_sp=max_sp, dropout=0.1) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_type, edge_sp):
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_sp)
        return x


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, a, b):
        g = self.gate(torch.cat([a, b], dim=-1))
        out = g * a + (1.0 - g) * b
        return self.norm(out)


class PairInteractionHead(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim=256, dropout=0.2):
        super().__init__()
        pair_dim = hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, m_vec, d_vec):
        pair_feat = torch.cat([m_vec, d_vec, m_vec * d_vec, torch.abs(m_vec - d_vec)], dim=-1)
        mlp_logit = self.mlp(pair_feat)
        bilinear_logit = self.bilinear(m_vec, d_vec)
        gate = self.gate(pair_feat)
        logits = gate * mlp_logit + (1.0 - gate) * bilinear_logit
        return logits.squeeze(-1)


class PairConditionedNeighborhoodReadout(nn.Module):
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _ensure_neighbor_cache(self, bundle):
        if hasattr(bundle, "_pair_neighbor_cache"):
            return bundle._pair_neighbor_cache

        num_nodes = bundle.num_mirna + bundle.num_drug
        rel_neighbors = {
            rel: [[] for _ in range(num_nodes)]
            for rel in range(bundle.num_relations)
        }

        edge_index = bundle.edge_index.detach().cpu()
        edge_type = bundle.edge_type.detach().cpu()
        for idx in range(edge_index.shape[1]):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            rel = int(edge_type[idx])
            rel_neighbors[rel][src].append(dst)

        bundle._pair_neighbor_cache = rel_neighbors
        return rel_neighbors

    def _aggregate_for_node(self, query, node_id, rel_ids, x_global, neighbor_cache, device):
        contexts = []
        scale = math.sqrt(self.hidden_dim)
        for rel in rel_ids:
            neighbors = neighbor_cache[rel][node_id]
            if not neighbors:
                continue
            neigh_idx = torch.tensor(neighbors, device=device, dtype=torch.long)
            neigh_repr = x_global[neigh_idx]
            rel_repr = self.rel_emb.weight[rel].unsqueeze(0)
            key = self.key_proj(neigh_repr + rel_repr)
            value = self.value_proj(neigh_repr + rel_repr)
            scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1) / scale
            alpha = torch.softmax(scores, dim=0)
            contexts.append(torch.sum(value * alpha.unsqueeze(-1), dim=0))

        if not contexts:
            return x_global.new_zeros(self.hidden_dim)

        stacked = torch.stack(contexts, dim=0).mean(dim=0)
        return self.norm(self.out_proj(stacked))

    def _aggregate_for_targets(self, query, target_ids, rel_id, x_global, device):
        if not target_ids:
            return x_global.new_zeros(self.hidden_dim)

        unique_ids = sorted(set(int(t) for t in target_ids))
        neigh_idx = torch.tensor(unique_ids, device=device, dtype=torch.long)
        neigh_repr = x_global[neigh_idx]
        rel_repr = self.rel_emb.weight[rel_id].unsqueeze(0)
        key = self.key_proj(neigh_repr + rel_repr)
        value = self.value_proj(neigh_repr + rel_repr)
        scores = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_dim)
        alpha = torch.softmax(scores, dim=0)
        return self.norm(self.out_proj(torch.sum(value * alpha.unsqueeze(-1), dim=0)))

    def _collect_two_hop_targets(self, node_id, first_rel, second_rel, neighbor_cache):
        targets = []
        for mid in neighbor_cache[first_rel][node_id]:
            targets.extend(neighbor_cache[second_rel][mid])
        return targets

    def forward(self, m_vec, d_vec, pair_idx, x_global, bundle):
        device = x_global.device
        pair_feat = torch.cat([m_vec, d_vec, m_vec * d_vec, torch.abs(m_vec - d_vec)], dim=-1)
        queries = self.query_proj(pair_feat)
        neighbor_cache = self._ensure_neighbor_cache(bundle)

        m_ctx_list = []
        d_ctx_list = []
        m_path_ctx_list = []
        d_path_ctx_list = []
        for row in range(pair_idx.size(0)):
            m_idx = int(pair_idx[row, 0].item())
            d_idx = int(pair_idx[row, 1].item()) + bundle.num_mirna
            query = queries[row]
            m_ctx = self._aggregate_for_node(
                query=query,
                node_id=m_idx,
                rel_ids=(0, 2),
                x_global=x_global,
                neighbor_cache=neighbor_cache,
                device=device,
            )
            d_ctx = self._aggregate_for_node(
                query=query,
                node_id=d_idx,
                rel_ids=(1, 3),
                x_global=x_global,
                neighbor_cache=neighbor_cache,
                device=device,
            )
            m_two_hop_targets = self._collect_two_hop_targets(
                node_id=m_idx,
                first_rel=0,
                second_rel=3,
                neighbor_cache=neighbor_cache,
            )
            d_two_hop_targets = self._collect_two_hop_targets(
                node_id=d_idx,
                first_rel=1,
                second_rel=2,
                neighbor_cache=neighbor_cache,
            )
            m_path_ctx = self._aggregate_for_targets(
                query=query,
                target_ids=m_two_hop_targets,
                rel_id=3,
                x_global=x_global,
                device=device,
            )
            d_path_ctx = self._aggregate_for_targets(
                query=query,
                target_ids=d_two_hop_targets,
                rel_id=2,
                x_global=x_global,
                device=device,
            )
            m_ctx_list.append(m_ctx)
            d_ctx_list.append(d_ctx)
            m_path_ctx_list.append(m_path_ctx)
            d_path_ctx_list.append(d_path_ctx)

        m_ctx = torch.stack(m_ctx_list, dim=0)
        d_ctx = torch.stack(d_ctx_list, dim=0)
        m_path_ctx = torch.stack(m_path_ctx_list, dim=0)
        d_path_ctx = torch.stack(d_path_ctx_list, dim=0)
        return m_ctx, d_ctx, m_path_ctx, d_path_ctx


class RASPMDAModel(nn.Module):
    def __init__(
        self,
        drug_node_dim,
        drug_edge_dim,
        hidden_dim=128,
        num_relations=5,
        kg_max_sp=3,
        moldeberta_dir=None,
    ):
        super().__init__()

        self.mirna_cnn = CNNFeatureExtractor(output_dim=hidden_dim)
        self.mirna_rnafm = RNAFMFeatureExtractor(output_dim=hidden_dim)

        self.drug_graph_encoder = DrugGraphTransformerMol(
            node_in_dim=drug_node_dim,
            edge_in_dim=drug_edge_dim,
            hidden_dim=hidden_dim,
            heads=4,
        )
        self.drug_molecular_lm = MolecularLMFeatureExtractor(output_dim=hidden_dim, model_dir=moldeberta_dir)
        if not self.mirna_rnafm.enabled:
            raise RuntimeError(
                f"Failed to enable RNA-FM backend: {self.mirna_rnafm.init_error}"
            ) from self.mirna_rnafm.init_error
        if not self.drug_molecular_lm.enabled:
            raise RuntimeError(
                f"Failed to enable MolDeBERTa backend from '{self.drug_molecular_lm.source}': "
                f"{self.drug_molecular_lm.init_error}"
            ) from self.drug_molecular_lm.init_error

        self.seq_fusion = GatedFusion(hidden_dim)
        self.drug_struct_fusion = GatedFusion(hidden_dim)
        self.drug_fusion = GatedFusion(hidden_dim)
        self.mirna_fusion = GatedFusion(hidden_dim)

        self.kg_encoder = HeteroRelationEncoder(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            max_sp=kg_max_sp,
            num_layers=2,
        )
        self.pair_readout = PairConditionedNeighborhoodReadout(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
        )
        self.m_context_fusion = GatedFusion(hidden_dim)
        self.d_context_fusion = GatedFusion(hidden_dim)
        self.m_path_fusion = GatedFusion(hidden_dim)
        self.d_path_fusion = GatedFusion(hidden_dim)

        self.classifier = PairInteractionHead(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=256,
            dropout=0.2,
        )

    def backend_status(self):
        status = {
            "rnafm_enabled": bool(self.mirna_rnafm.enabled),
            "moldeberta_enabled": bool(self.drug_molecular_lm.enabled),
            "moldeberta_source": self.drug_molecular_lm.source,
        }
        return status

    def encode_nodes(self, bundle, device):
        mirna_tokens = bundle.mirna_tokens.to(device)

        m_cnn = self.mirna_cnn(mirna_tokens)
        m_rnafm = self.mirna_rnafm(bundle.mirna_sequences, fallback_tensor=m_cnn)
        m_seq = self.seq_fusion(m_cnn, m_rnafm)

        drug_batch = Batch.from_data_list(bundle.drug_graphs).to(device)
        d_graph = self.drug_graph_encoder(drug_batch)

        d_llm = self.drug_molecular_lm(bundle.drug_smiles, fallback_tensor=d_graph)
        d_struct = self.drug_struct_fusion(d_graph, d_llm)

        x0 = torch.cat([m_seq, d_struct], dim=0)

        x = self.kg_encoder(
            x=x0,
            edge_index=bundle.attn_edge_index.to(device),
            edge_type=bundle.attn_edge_type.to(device),
            edge_sp=bundle.attn_edge_sp.to(device),
        )

        m_kg = x[: bundle.num_mirna]
        d_kg = x[bundle.num_mirna :]

        m_final = self.mirna_fusion(m_seq, m_kg)
        d_final = self.drug_fusion(d_struct, d_kg)

        return m_final, d_final, m_seq, d_struct, m_kg, d_kg, x

    def forward(self, pair_idx, bundle, return_aux=False):
        device = next(self.parameters()).device
        m_final, d_final, m_seq, d_struct, m_kg, d_kg, x_global = self.encode_nodes(bundle, device)

        m_idx = pair_idx[:, 0]
        d_idx = pair_idx[:, 1]

        m_vec = m_final[m_idx]
        d_vec = d_final[d_idx]
        m_ctx, d_ctx, m_path_ctx, d_path_ctx = self.pair_readout(m_vec, d_vec, pair_idx, x_global, bundle)
        m_vec = self.m_context_fusion(m_vec, m_ctx)
        d_vec = self.d_context_fusion(d_vec, d_ctx)
        d_vec = self.d_path_fusion(d_vec, m_path_ctx)
        m_vec = self.m_path_fusion(m_vec, d_path_ctx)

        logits = self.classifier(m_vec, d_vec)

        if not return_aux:
            return logits

        aux = {
            "m_final": m_vec,
            "d_final": d_vec,
            "m_seq": m_seq[m_idx],
            "m_kg": m_kg[m_idx],
            "d_struct": d_struct[d_idx],
            "d_kg": d_kg[d_idx],
            "m_ctx": m_ctx,
            "d_ctx": d_ctx,
            "m_path_ctx": m_path_ctx,
            "d_path_ctx": d_path_ctx,
        }
        return logits, aux
