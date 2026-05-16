import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from collections import deque
import json


ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se", "Na", "K", "Li", "Ca", "Mg", "Zn", "Fe", "Cu", "Mn", "Co", "Ni", "Al", "Unknown"
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


@dataclass
class GraphDataBundle:
    mirna_tokens: torch.Tensor
    mirna_sequences: list
    drug_graphs: list
    drug_smiles: list
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    attn_edge_index: torch.Tensor
    attn_edge_type: torch.Tensor
    attn_edge_sp: torch.Tensor
    train_pairs: torch.Tensor
    train_labels: torch.Tensor
    test_pairs: torch.Tensor
    test_labels: torch.Tensor
    num_mirna: int
    num_drug: int
    num_relations: int
    split_mode: str
    split_summary: dict


def _one_hot_unk(value, choices):
    if value not in choices:
        value = choices[-1]
    return [1.0 if value == c else 0.0 for c in choices]


def _encode_sequence(seq: str, max_len: int = 24):
    mapping = {"A": 1, "U": 2, "C": 3, "G": 4}
    arr = [mapping.get(ch, 0) for ch in str(seq).upper()]
    if len(arr) < max_len:
        arr = arr + [0] * (max_len - len(arr))
    else:
        arr = arr[:max_len]
    return arr


def _kmer_vector(seq: str, k: int = 3):
    alphabet = ["A", "U", "C", "G"]
    kmers = [a + b + c for a in alphabet for b in alphabet for c in alphabet]
    idx = {kmer: i for i, kmer in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    s = str(seq).upper()
    if len(s) < k:
        return vec
    for i in range(len(s) - k + 1):
        token = s[i : i + k]
        if token in idx:
            vec[idx[token]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _morgan_fp(smiles: str, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for i in range(n_bits):
        arr[i] = float(fp[i])
    return arr


def _bond_feature(bond):
    bond_type = bond.GetBondType()
    feat = _one_hot_unk(bond_type, BOND_TYPES + ["OTHER"])
    feat.append(float(bond.GetIsConjugated()))
    feat.append(float(bond.IsInRing()))
    return feat


def _node_feature(atom):
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES[:-1] else "Unknown"
    feat = _one_hot_unk(symbol, ATOM_TYPES)
    feat += _one_hot_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    feat += _one_hot_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feat += _one_hot_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feat.append(float(atom.GetIsAromatic()))
    return feat


def _shortest_path_hist(num_nodes: int, edges: list, max_dist: int = 8):
    neigh = [[] for _ in range(num_nodes)]
    for u, v in edges:
        neigh[u].append(v)

    out = np.zeros((num_nodes, max_dist + 1), dtype=np.float32)
    for start in range(num_nodes):
        dist = np.full(num_nodes, -1, dtype=np.int64)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt in neigh[cur]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        for d in dist:
            if d >= 0:
                out[start, min(int(d), max_dist)] += 1.0
        total = out[start].sum()
        if total > 0:
            out[start] /= total
    return out


def _smiles_to_graph(smiles: str, max_sp: int = 8):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    if n == 0:
        return None

    x_base = [_node_feature(a) for a in mol.GetAtoms()]

    directed_edges = []
    edge_attr = []
    undirected_pairs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_feature(bond)
        directed_edges.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])
        undirected_pairs.extend([(i, j), (j, i)])

    if len(directed_edges) == 0:
        directed_edges = [[0, 0]]
        edge_attr = [[0.0] * (len(BOND_TYPES) + 1 + 2)]
        undirected_pairs = [(0, 0)]

    sp_hist = _shortest_path_hist(n, undirected_pairs, max_dist=max_sp)
    x = np.concatenate([np.array(x_base, dtype=np.float32), sp_hist], axis=1)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32), dtype=torch.float32)

    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)


def _topk_edges_from_similarity(sim: np.ndarray, topk: int):
    n = sim.shape[0]
    edges = []
    k = min(max(1, topk), max(1, n - 1))
    for i in range(n):
        row = sim[i].copy()
        row[i] = -1
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]
        for j in idx:
            if row[j] > 0:
                edges.append((i, int(j)))
    return edges


def _build_mirna_similarity(sequences, topk: int):
    kmers = np.stack([_kmer_vector(s, k=3) for s in sequences], axis=0)
    sim = kmers @ kmers.T
    return _topk_edges_from_similarity(sim, topk=topk)


def _build_drug_similarity(fps: np.ndarray, topk: int):
    inter = fps @ fps.T
    counts = fps.sum(axis=1, keepdims=True)
    denom = counts + counts.T - inter
    denom[denom == 0] = 1.0
    tanimoto = inter / denom
    return _topk_edges_from_similarity(tanimoto, topk=topk)


def _filter_pairs(arr, mirna_map, drug_map):
    out = []
    for m_idx, d_idx in arr:
        if int(m_idx) in mirna_map and int(d_idx) in drug_map:
            out.append([mirna_map[int(m_idx)], drug_map[int(d_idx)]])
    if len(out) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(out, dtype=np.int64)


def _safe_load_pairs(path):
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :2]


def _resolve_fixed_split_dir(fixed_split_dir, split_mode, seed, pos_sample):
    candidates = []
    if fixed_split_dir:
        candidates.append(Path(fixed_split_dir))

    pos_parent = Path(pos_sample).resolve().parent
    candidates.append(pos_parent / "fixed_splits" / f"seed_{seed}" / split_mode)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _load_fixed_split_pairs(split_dir, mirna_map, drug_map):
    required = {
        "train_pos": split_dir / "train_pos.edgelist",
        "train_neg": split_dir / "train_neg.edgelist",
        "test_pos": split_dir / "test_pos.edgelist",
        "test_neg": split_dir / "test_neg.edgelist",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fixed split directory is missing required files: {', '.join(missing)}"
        )

    loaded = {
        name: _filter_pairs(_safe_load_pairs(path), mirna_map, drug_map)
        for name, path in required.items()
    }
    for name, arr in loaded.items():
        if len(arr) == 0:
            raise ValueError(f"Fixed split file is empty after filtering valid IDs: {split_dir / (name + '.edgelist')}")

    metadata_path = split_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return loaded, metadata


def _select_group_ids_for_target(
    group_ids,
    group_counts,
    target_count,
    total_count,
    rng,
    group_entity_sizes=None,
    target_entity_count=None,
    tie_break_entity_count=None,
):
    if len(group_ids) == 0:
        raise ValueError("Cannot split an empty set of groups.")

    if group_entity_sizes is None:
        group_entity_sizes = np.ones(len(group_ids), dtype=np.int64)

    order = np.arange(len(group_ids))
    rng.shuffle(order)
    group_ids = np.asarray(group_ids, dtype=np.int64)[order]
    group_counts = np.asarray(group_counts, dtype=np.int64)[order]
    group_entity_sizes = np.asarray(group_entity_sizes, dtype=np.int64)[order]

    if target_entity_count is None:
        target_entity_count = int(group_entity_sizes.sum())
    if tie_break_entity_count is None:
        tie_break_entity_count = target_entity_count

    # For each attainable held-out edge count, keep the subset whose entity
    # total is closest to the requested target. This makes edge-count proximity
    # the primary objective and entity-ratio proximity the secondary objective.
    best_states = {0: (0, 0, 0, 0)}  # held_out_edges -> (entity_total, tie_entity_total, group_total, mask)

    for idx, (count, entity_size) in enumerate(zip(group_counts.tolist(), group_entity_sizes.tolist())):
        current_states = list(best_states.items())
        for held_out, (entity_total, tie_entity_total, group_total, mask) in current_states:
            next_held_out = held_out + int(count)
            if next_held_out >= total_count:
                continue

            next_state = (
                entity_total + int(entity_size),
                tie_entity_total + 1,
                group_total + 1,
                mask | (1 << idx),
            )
            existing_state = best_states.get(next_held_out)
            if existing_state is None or (
                abs(next_state[0] - target_entity_count),
                abs(next_state[1] - tie_break_entity_count),
                -next_state[2],
            ) < (
                abs(existing_state[0] - target_entity_count),
                abs(existing_state[1] - tie_break_entity_count),
                -existing_state[2],
            ):
                best_states[next_held_out] = next_state

    candidate_states = [
        (held_out, entity_total, tie_entity_total, group_total, mask)
        for held_out, (entity_total, tie_entity_total, group_total, mask) in best_states.items()
        if held_out > 0 and mask != 0
    ]
    if not candidate_states:
        raise ValueError("Failed to choose any held-out groups for the requested split.")

    best_held_out, _, _, _, best_mask = min(
        candidate_states,
        key=lambda item: (
            abs(item[0] - target_count),
            abs(item[1] - target_entity_count),
            abs(item[2] - tie_break_entity_count),
            -item[3],
            item[0],
        ),
    )
    del best_held_out

    selected = [int(group_ids[idx]) for idx in range(len(group_ids)) if (best_mask >> idx) & 1]
    return np.array(selected, dtype=np.int64)


def _select_rows_by_entity(pos, test_ratio, entity_axis, rng):
    entities, counts = np.unique(pos[:, entity_axis], return_counts=True)
    target_test = max(1, int(round(len(pos) * test_ratio)))
    target_entities = max(1, int(round(len(entities) * test_ratio)))
    test_entities = _select_group_ids_for_target(
        group_ids=entities,
        group_counts=counts,
        target_count=target_test,
        total_count=len(pos),
        rng=rng,
        group_entity_sizes=np.ones(len(entities), dtype=np.int64),
        target_entity_count=target_entities,
    )
    test_mask = np.isin(pos[:, entity_axis], test_entities)
    train_pos = pos[~test_mask]
    test_pos = pos[test_mask]
    if len(train_pos) == 0 or len(test_pos) == 0:
        raise ValueError("Cold-start split produced an empty train or test set. Please adjust test_ratio.")
    return train_pos, test_pos


def _scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def _scaffold_labels_for_drugs(drug_smiles):
    labels = []
    for drug_idx, smiles in enumerate(drug_smiles):
        labels.append(_scaffold_from_smiles(smiles) or f"__invalid_{drug_idx}")
    return labels


def _select_rows_by_scaffold(pos, drug_smiles, test_ratio, rng):
    scaffold_groups = {}
    for drug_idx, smiles in enumerate(drug_smiles):
        scaffold = _scaffold_from_smiles(smiles)
        if scaffold is None:
            scaffold = f"__invalid_{drug_idx}"
        scaffold_groups.setdefault(scaffold, []).append(drug_idx)

    target_test = max(1, int(round(len(pos) * test_ratio)))
    group_labels = []
    group_counts = []
    group_members = []
    for _, members in scaffold_groups.items():
        count = int(np.isin(pos[:, 1], np.array(members, dtype=np.int64)).sum())
        if count == 0:
            continue
        group_labels.append(len(group_members))
        group_counts.append(count)
        group_members.append([int(d) for d in members])

    if not group_members:
        raise ValueError("Scaffold split could not find any scaffold with positive samples.")

    selected_group_ids = _select_group_ids_for_target(
        group_ids=np.array(group_labels, dtype=np.int64),
        group_counts=np.array(group_counts, dtype=np.int64),
        target_count=target_test,
        total_count=len(pos),
        rng=rng,
        group_entity_sizes=np.ones(len(group_members), dtype=np.int64),
        target_entity_count=max(1, int(round(len(group_members) * test_ratio))),
        tie_break_entity_count=max(1, int(round(sum(len(members) for members in group_members) * test_ratio))),
    )
    test_drugs = set()
    for group_id in selected_group_ids.tolist():
        test_drugs.update(group_members[group_id])

    test_drugs = np.array(sorted(test_drugs), dtype=np.int64)
    test_mask = np.isin(pos[:, 1], test_drugs)
    train_pos = pos[~test_mask]
    test_pos = pos[test_mask]
    if len(train_pos) == 0 or len(test_pos) == 0:
        raise ValueError("Scaffold split produced an empty train or test set. Please adjust test_ratio.")
    return train_pos, test_pos


def _split_positive_pairs(pos, split_mode, test_ratio, rng, drug_smiles):
    if split_mode == "mirna_cold_start":
        train_pos, test_pos = _select_rows_by_entity(pos, test_ratio=test_ratio, entity_axis=0, rng=rng)
        return train_pos, test_pos, np.unique(train_pos[:, 0]), np.unique(pos[:, 1])
    if split_mode == "drug_cold_start":
        train_pos, test_pos = _select_rows_by_entity(pos, test_ratio=test_ratio, entity_axis=1, rng=rng)
        return train_pos, test_pos, np.unique(pos[:, 0]), np.unique(train_pos[:, 1])
    if split_mode == "drug_scaffold":
        train_pos, test_pos = _select_rows_by_scaffold(pos, drug_smiles=drug_smiles, test_ratio=test_ratio, rng=rng)
        return train_pos, test_pos, np.unique(pos[:, 0]), np.unique(train_pos[:, 1])
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _dedupe_negative_candidates(neg, reference_pos):
    pos_set = {tuple(map(int, row)) for row in reference_pos}
    deduped = []
    seen = set()
    for row in neg:
        key = (int(row[0]), int(row[1]))
        if key in pos_set or key in seen:
            continue
        seen.add(key)
        deduped.append([key[0], key[1]])

    if len(deduped) == 0:
        raise ValueError("No valid negatives remain after removing positive overlaps.")

    return np.array(deduped, dtype=np.int64)


def _partition_negative_pool(neg, reference_pos, entity_axis, train_entities, test_entities):
    candidate_neg = _dedupe_negative_candidates(neg, reference_pos)

    if entity_axis is None:
        return candidate_neg, candidate_neg

    train_entities = np.array(sorted(set(int(x) for x in train_entities)), dtype=np.int64)
    test_entities = np.array(sorted(set(int(x) for x in test_entities)), dtype=np.int64)
    if len(train_entities) == 0 or len(test_entities) == 0:
        raise ValueError("Train/test entity pools must be non-empty for negative sampling.")
    if np.intersect1d(train_entities, test_entities).size > 0:
        raise ValueError("Train and test negative entity pools overlap in a cold-start split.")

    values = candidate_neg[:, entity_axis]
    train_neg = candidate_neg[np.isin(values, train_entities)]
    test_neg = candidate_neg[np.isin(values, test_entities)]
    if len(train_neg) == 0 or len(test_neg) == 0:
        raise ValueError("Negative pool partitioning produced an empty train or test subset.")
    return train_neg, test_neg


def _partition_negative_pool_by_scaffold(neg, reference_pos, drug_smiles, train_drug_ids, test_drug_ids):
    candidate_neg = _dedupe_negative_candidates(neg, reference_pos)
    scaffold_labels = _scaffold_labels_for_drugs(drug_smiles)

    train_scaffolds = {scaffold_labels[int(drug_id)] for drug_id in train_drug_ids}
    test_scaffolds = {scaffold_labels[int(drug_id)] for drug_id in test_drug_ids}
    if len(train_scaffolds) == 0 or len(test_scaffolds) == 0:
        raise ValueError("Train/test scaffold pools must be non-empty for scaffold negative sampling.")
    if train_scaffolds & test_scaffolds:
        raise ValueError("Train and test scaffold pools overlap in a scaffold split.")

    neg_scaffolds = np.array([scaffold_labels[int(drug_id)] for drug_id in candidate_neg[:, 1]], dtype=object)
    train_neg = candidate_neg[np.isin(neg_scaffolds, list(train_scaffolds))]
    test_neg = candidate_neg[np.isin(neg_scaffolds, list(test_scaffolds))]
    if len(train_neg) == 0 or len(test_neg) == 0:
        raise ValueError("Scaffold negative pool partitioning produced an empty train or test subset.")
    return train_neg, test_neg


def _sample_negatives(candidate_neg, sample_size, rng):
    if sample_size <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    candidate_neg = np.array(candidate_neg, dtype=np.int64, copy=True)
    rng.shuffle(candidate_neg)
    if len(candidate_neg) < sample_size:
        raise ValueError(
            f"Not enough negatives for split: required {sample_size}, available {len(candidate_neg)}."
        )
    return candidate_neg[:sample_size]


def _resolve_negative_sample_size(num_pos, neg_ratio):
    if neg_ratio < 0:
        raise ValueError("neg_ratio must be non-negative.")
    if num_pos <= 0 or neg_ratio == 0:
        return 0
    return max(1, int(round(num_pos * neg_ratio)))


def _apply_sample_limit(train_pos, test_pos, sample_limit, rng):
    if sample_limit is None:
        return train_pos, test_pos
    total_pos = len(train_pos) + len(test_pos)
    if total_pos <= sample_limit:
        return train_pos, test_pos

    train_target = max(1, int(round(sample_limit * len(train_pos) / total_pos)))
    test_target = max(1, sample_limit - train_target)
    train_target = min(train_target, len(train_pos))
    test_target = min(test_target, len(test_pos))

    if train_target + test_target < min(sample_limit, total_pos):
        remaining = min(sample_limit, total_pos) - (train_target + test_target)
        train_room = len(train_pos) - train_target
        add_train = min(remaining, train_room)
        train_target += add_train
        remaining -= add_train
        if remaining > 0:
            test_room = len(test_pos) - test_target
            test_target += min(remaining, test_room)

    train_idx = rng.permutation(len(train_pos))[:train_target]
    test_idx = rng.permutation(len(test_pos))[:test_target]
    return train_pos[train_idx], test_pos[test_idx]


def _build_split_summary(train_pos, test_pos, split_mode, drug_smiles, train_neg=None, test_neg=None):
    train_scaffolds = set()
    test_scaffolds = set()
    if len(train_pos):
        train_scaffolds = {_scaffold_from_smiles(drug_smiles[int(d_idx)]) or f"__invalid_{int(d_idx)}" for d_idx in np.unique(train_pos[:, 1])}
    if len(test_pos):
        test_scaffolds = {_scaffold_from_smiles(drug_smiles[int(d_idx)]) or f"__invalid_{int(d_idx)}" for d_idx in np.unique(test_pos[:, 1])}

    summary = {
        "split_mode": split_mode,
        "train_pos": int(len(train_pos)),
        "test_pos": int(len(test_pos)),
        "actual_test_ratio": float(len(test_pos) / max(1, len(train_pos) + len(test_pos))),
        "train_mirna": int(len(np.unique(train_pos[:, 0]))) if len(train_pos) else 0,
        "test_mirna": int(len(np.unique(test_pos[:, 0]))) if len(test_pos) else 0,
        "train_drug": int(len(np.unique(train_pos[:, 1]))) if len(train_pos) else 0,
        "test_drug": int(len(np.unique(test_pos[:, 1]))) if len(test_pos) else 0,
        "train_scaffold": int(len(train_scaffolds)),
        "test_scaffold": int(len(test_scaffolds)),
        "mirna_overlap": int(len(np.intersect1d(np.unique(train_pos[:, 0]), np.unique(test_pos[:, 0]))))
        if len(train_pos) and len(test_pos)
        else 0,
        "drug_overlap": int(len(np.intersect1d(np.unique(train_pos[:, 1]), np.unique(test_pos[:, 1]))))
        if len(train_pos) and len(test_pos)
        else 0,
        "scaffold_overlap": int(len(train_scaffolds & test_scaffolds)),
    }
    if train_neg is not None and test_neg is not None:
        summary["train_neg"] = int(len(train_neg))
        summary["test_neg"] = int(len(test_neg))
        summary["train_neg_ratio"] = float(len(train_neg) / max(1, len(train_pos)))
        summary["test_neg_ratio"] = float(len(test_neg) / max(1, len(test_pos)))
    return summary


def _enrich_graph_split_summary(summary):
    summary = dict(summary)
    summary["graph_context"] = "target_edge_unseen"
    return summary


def _build_rel_spatial_edges(edge_index, edge_type, num_nodes, max_hop=3, max_neighbors=64, virtual_rel=4):
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    rel = edge_type.tolist()

    direct_rel = {(u, v): r for u, v, r in zip(src, dst, rel)}

    undirected_adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        undirected_adj[u].append(v)
        undirected_adj[v].append(u)

    out_src, out_dst, out_rel, out_sp = [], [], [], []

    for target in range(num_nodes):
        dist = {target: 0}
        q = deque([target])

        while q:
            cur = q.popleft()
            if dist[cur] >= max_hop:
                continue
            for nxt in undirected_adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)

        candidates = [(node, d) for node, d in dist.items() if node != target]
        candidates.sort(key=lambda x: (x[1], x[0]))
        if len(candidates) > max_neighbors:
            candidates = candidates[:max_neighbors]

        for node, d in candidates:
            out_src.append(node)
            out_dst.append(target)
            out_sp.append(int(d))
            out_rel.append(int(direct_rel.get((node, target), virtual_rel)))

    return (
        torch.tensor([out_src, out_dst], dtype=torch.long),
        torch.tensor(out_rel, dtype=torch.long),
        torch.tensor(out_sp, dtype=torch.long),
    )


def load_mda_graph_data(
    pos_sample,
    neg_sample,
    mirna_file,
    drug_file,
    seed=0,
    test_ratio=0.2,
    split_mode="drug_cold_start",
    sample_limit=None,
    mm_topk=10,
    dd_topk=10,
    kg_max_hop=3,
    kg_max_neighbors=64,
    fixed_split_dir=None,
    train_neg_ratio=1.0,
    test_neg_ratio=1.0,
):
    rng = np.random.default_rng(seed)

    mirna_df = pd.read_excel(mirna_file)
    drug_df = pd.read_excel(drug_file)

    mirna_sequences = {}
    mirna_tokens = {}
    for idx, row in mirna_df.iterrows():
        seq = str(row["miRNA_Sequence"])
        if len(seq) > 0 and seq != "nan":
            mirna_sequences[int(idx)] = seq
            mirna_tokens[int(idx)] = _encode_sequence(seq)

    drug_graph_by_orig = {}
    drug_smiles_by_orig = {}
    drug_fps = {}
    for idx, row in drug_df.iterrows():
        smiles = str(row["SMILES"])
        graph = _smiles_to_graph(smiles)
        fp = _morgan_fp(smiles)
        if graph is not None and fp is not None:
            drug_graph_by_orig[int(idx)] = graph
            drug_smiles_by_orig[int(idx)] = smiles
            drug_fps[int(idx)] = fp

    mirna_ids = sorted(mirna_sequences.keys())
    drug_ids = sorted(drug_graph_by_orig.keys())

    mirna_map = {orig: new for new, orig in enumerate(mirna_ids)}
    drug_map = {orig: new for new, orig in enumerate(drug_ids)}

    mirna_seq_list = [mirna_sequences[i] for i in mirna_ids]
    mirna_token_tensor = torch.tensor([mirna_tokens[i] for i in mirna_ids], dtype=torch.long)

    drug_graphs = [drug_graph_by_orig[i] for i in drug_ids]
    drug_smiles = [drug_smiles_by_orig[i] for i in drug_ids]
    drug_fp_mat = np.stack([drug_fps[i] for i in drug_ids], axis=0)

    pos = _safe_load_pairs(pos_sample)
    neg = _safe_load_pairs(neg_sample)
    pos = _filter_pairs(pos, mirna_map, drug_map)
    neg = _filter_pairs(neg, mirna_map, drug_map)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Positive or negative sample file is empty after filtering valid miRNA/drug IDs.")

    resolved_fixed_split_dir = _resolve_fixed_split_dir(
        fixed_split_dir=fixed_split_dir,
        split_mode=split_mode,
        seed=seed,
        pos_sample=pos_sample,
    )

    fixed_split_metadata = None
    if resolved_fixed_split_dir is not None:
        if sample_limit is not None:
            raise ValueError("sample_limit is not supported when using fixed split files.")
        fixed_pairs, fixed_split_metadata = _load_fixed_split_pairs(
            split_dir=resolved_fixed_split_dir,
            mirna_map=mirna_map,
            drug_map=drug_map,
        )
        train_pos = fixed_pairs["train_pos"]
        train_neg = fixed_pairs["train_neg"]
        test_pos = fixed_pairs["test_pos"]
        test_neg = fixed_pairs["test_neg"]
    elif split_mode == "drug_scaffold":
        train_pos, test_pos, _, _ = _split_positive_pairs(
            pos=pos,
            split_mode=split_mode,
            test_ratio=test_ratio,
            rng=rng,
            drug_smiles=drug_smiles,
        )
        train_pos, test_pos = _apply_sample_limit(train_pos, test_pos, sample_limit=sample_limit, rng=rng)
        train_neg_pool, test_neg_pool = _partition_negative_pool_by_scaffold(
            neg=neg,
            reference_pos=pos,
            drug_smiles=drug_smiles,
            train_drug_ids=np.unique(train_pos[:, 1]).tolist(),
            test_drug_ids=np.unique(test_pos[:, 1]).tolist(),
        )
        train_neg = _sample_negatives(
            train_neg_pool,
            sample_size=_resolve_negative_sample_size(len(train_pos), train_neg_ratio),
            rng=rng,
        )
        test_neg = _sample_negatives(
            test_neg_pool,
            sample_size=_resolve_negative_sample_size(len(test_pos), test_neg_ratio),
            rng=rng,
        )
    else:
        train_pos, test_pos, _, _ = _split_positive_pairs(
            pos=pos,
            split_mode=split_mode,
            test_ratio=test_ratio,
            rng=rng,
            drug_smiles=drug_smiles,
        )
        train_pos, test_pos = _apply_sample_limit(train_pos, test_pos, sample_limit=sample_limit, rng=rng)
        train_neg_pool, test_neg_pool = _partition_negative_pool(
            neg=neg,
            reference_pos=pos,
            entity_axis=1,
            train_entities=np.unique(train_pos[:, 1]).tolist(),
            test_entities=np.unique(test_pos[:, 1]).tolist(),
        )
        train_neg = _sample_negatives(
            train_neg_pool,
            sample_size=_resolve_negative_sample_size(len(train_pos), train_neg_ratio),
            rng=rng,
        )
        test_neg = _sample_negatives(
            test_neg_pool,
            sample_size=_resolve_negative_sample_size(len(test_pos), test_neg_ratio),
            rng=rng,
        )

    train_pairs = np.vstack([train_pos, train_neg])
    train_labels = np.concatenate(
        [np.ones(len(train_pos), dtype=np.int64), np.zeros(len(train_neg), dtype=np.int64)]
    )
    test_pairs = np.vstack([test_pos, test_neg])
    test_labels = np.concatenate(
        [np.ones(len(test_pos), dtype=np.int64), np.zeros(len(test_neg), dtype=np.int64)]
    )

    train_perm = rng.permutation(len(train_pairs))
    test_perm = rng.permutation(len(test_pairs))
    train_pairs, train_labels = train_pairs[train_perm], train_labels[train_perm]
    test_pairs, test_labels = test_pairs[test_perm], test_labels[test_perm]

    num_m = len(mirna_ids)

    runtime_split_summary = _build_split_summary(
        train_pos=train_pos,
        test_pos=test_pos,
        split_mode=split_mode,
        drug_smiles=drug_smiles,
        train_neg=train_neg,
        test_neg=test_neg,
    )
    if fixed_split_metadata is not None and "split_summary" in fixed_split_metadata:
        runtime_split_summary.update(fixed_split_metadata.get("split_summary", {}))

    mm_edges = _build_mirna_similarity(mirna_seq_list, topk=mm_topk)
    dd_edges = _build_drug_similarity(drug_fp_mat, topk=dd_topk)

    src, dst, rel = [], [], []

    # R_md
    for m_idx, d_idx in train_pos:
        d_global = num_m + int(d_idx)
        src.extend([int(m_idx), d_global])
        dst.extend([d_global, int(m_idx)])
        rel.extend([0, 1])

    # R_mm
    for u, v in mm_edges:
        src.append(u)
        dst.append(v)
        rel.append(2)

    # R_dd
    for u, v in dd_edges:
        src.append(num_m + u)
        dst.append(num_m + v)
        rel.append(3)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    attn_edge_index, attn_edge_type, attn_edge_sp = _build_rel_spatial_edges(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_m + len(drug_ids),
        max_hop=kg_max_hop,
        max_neighbors=kg_max_neighbors,
        virtual_rel=4,
    )

    split_summary = _enrich_graph_split_summary(runtime_split_summary)

    return GraphDataBundle(
        mirna_tokens=mirna_token_tensor,
        mirna_sequences=mirna_seq_list,
        drug_graphs=drug_graphs,
        drug_smiles=drug_smiles,
        edge_index=edge_index,
        edge_type=edge_type,
        attn_edge_index=attn_edge_index,
        attn_edge_type=attn_edge_type,
        attn_edge_sp=attn_edge_sp,
        train_pairs=torch.tensor(train_pairs, dtype=torch.long),
        train_labels=torch.tensor(train_labels, dtype=torch.float32),
        test_pairs=torch.tensor(test_pairs, dtype=torch.long),
        test_labels=torch.tensor(test_labels, dtype=torch.float32),
        num_mirna=len(mirna_ids),
        num_drug=len(drug_ids),
        num_relations=5,
        split_mode=split_mode,
        split_summary=split_summary,
    )
