import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import load_mda_graph_data
from src.model import RASPMDAModel


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "aupr": average_precision_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def info_nce(anchor, positive, temperature=0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t()
    logits = logits / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def info_nce_deduplicated(anchor_all, positive_all, entity_idx, temperature=0.2):
    keep = []
    seen = set()
    for i, entity in enumerate(entity_idx.detach().cpu().tolist()):
        entity = int(entity)
        if entity in seen:
            continue
        seen.add(entity)
        keep.append(i)

    if len(keep) < 2:
        return anchor_all.new_tensor(0.0)

    keep_idx = torch.tensor(keep, device=anchor_all.device, dtype=torch.long)
    return info_nce(anchor_all[keep_idx], positive_all[keep_idx], temperature=temperature)


def evaluate(model, loader, bundle, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []

    with torch.no_grad():
        for pair_idx, labels in loader:
            pair_idx = pair_idx.to(device)
            labels = labels.to(device)
            logits = model(pair_idx, bundle)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    return total_loss / len(loader.dataset), metrics(y_true, y_prob)


def format_metrics(prefix, metric_dict):
    return (
        f"{prefix} "
        f"Acc {metric_dict['acc']:.4f} "
        f"AUC {metric_dict['auc']:.4f} "
        f"AUPR {metric_dict['aupr']:.4f} "
        f"Precision {metric_dict['precision']:.4f} "
        f"Recall {metric_dict['recall']:.4f} "
        f"F1 {metric_dict['f1']:.4f} "
        f"MCC {metric_dict['mcc']:.4f}"
    )


def build_model(args, bundle):
    drug_node_dim = bundle.drug_graphs[0].x.shape[1]
    drug_edge_dim = bundle.drug_graphs[0].edge_attr.shape[1]
    model = MDAGraphModel(
        drug_node_dim=drug_node_dim,
        drug_edge_dim=drug_edge_dim,
        hidden_dim=args.hidden_dim,
        num_relations=bundle.num_relations,
        kg_max_sp=args.kg_max_hop,
        moldeberta_dir=args.moldeberta_dir,
    )
    return model, drug_node_dim, drug_edge_dim


def print_backend_status(model):
    status = model.backend_status()
    print(
        "Feature backends | "
        f"rnafm_enabled={status['rnafm_enabled']} "
        f"moldeberta_enabled={status['moldeberta_enabled']} "
        f"moldeberta_source={status['moldeberta_source']}"
    )


def checkpoint_payload(model, args, epoch, metric_dict):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "args": vars(args).copy(),
        "metrics": metric_dict,
    }


def load_checkpoint_args(ckpt_args, cli_args):
    merged = vars(cli_args).copy()
    merged.update(ckpt_args)
    merged["checkpoint"] = getattr(cli_args, "checkpoint", None)
    merged["batch_size"] = cli_args.batch_size
    return argparse.Namespace(**merged)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bundle = load_mda_graph_data(
        pos_sample=args.pos_sample,
        neg_sample=args.neg_sample,
        mirna_file=args.mirna_file,
        drug_file=args.drug_file,
        seed=args.seed,
        test_ratio=args.test_ratio,
        split_mode=args.split_mode,
        sample_limit=args.sample_limit,
        mm_topk=args.mm_topk,
        dd_topk=args.dd_topk,
        kg_max_hop=args.kg_max_hop,
        kg_max_neighbors=args.kg_max_neighbors,
        fixed_split_dir=args.fixed_split_dir,
        train_neg_ratio=args.train_neg_ratio,
        test_neg_ratio=args.test_neg_ratio,
    )

    train_ds = TensorDataset(bundle.train_pairs, bundle.train_labels)
    test_ds = TensorDataset(bundle.test_pairs, bundle.test_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model, drug_node_dim, drug_edge_dim = build_model(args, bundle)
    model = model.to(device)
    print_backend_status(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_score = float("-inf")
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    print(
        f"Loaded data | split={bundle.split_mode} train={len(train_ds)} test={len(test_ds)} mirna={bundle.num_mirna} "
        f"drug={bundle.num_drug} kg_edges={bundle.edge_index.shape[1]} attn_edges={bundle.attn_edge_index.shape[1]} "
        f"node_dim={drug_node_dim} edge_dim={drug_edge_dim}"
    )
    print(
        "Split stats | "
        f"graph_context={bundle.split_summary.get('graph_context', 'target_edge_unseen')} "
        f"train_pos={bundle.split_summary['train_pos']} test_pos={bundle.split_summary['test_pos']} "
        f"train_neg={bundle.split_summary.get('train_neg', 0)} test_neg={bundle.split_summary.get('test_neg', 0)} "
        f"actual_test_ratio={bundle.split_summary.get('actual_test_ratio', 0.0):.4f} "
        f"train_neg_ratio={bundle.split_summary.get('train_neg_ratio', 0.0):.2f} "
        f"test_neg_ratio={bundle.split_summary.get('test_neg_ratio', 0.0):.2f} "
        f"train_mirna={bundle.split_summary['train_mirna']} test_mirna={bundle.split_summary['test_mirna']} "
        f"train_drug={bundle.split_summary['train_drug']} test_drug={bundle.split_summary['test_drug']} "
        f"train_scaffold={bundle.split_summary['train_scaffold']} test_scaffold={bundle.split_summary['test_scaffold']} "
        f"mirna_overlap={bundle.split_summary['mirna_overlap']} drug_overlap={bundle.split_summary['drug_overlap']} "
        f"scaffold_overlap={bundle.split_summary['scaffold_overlap']}"
    )


    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_mi = 0.0
        y_true, y_prob = [], []

        for pair_idx, labels in train_loader:
            pair_idx = pair_idx.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, aux = model(pair_idx, bundle, return_aux=True)

            loss_cls = criterion(logits, labels)
            loss_mi_m = info_nce_deduplicated(
                aux["m_seq"],
                aux["m_kg"],
                pair_idx[:, 0],
                temperature=args.mi_temp,
            )
            loss_mi_d = info_nce_deduplicated(
                aux["d_struct"],
                aux["d_kg"],
                pair_idx[:, 1],
                temperature=args.mi_temp,
            )
            loss_mi = args.lambda_mi_m * loss_mi_m + args.lambda_mi_d * loss_mi_d

            loss = loss_cls + loss_mi
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls += loss_cls.item() * bs
            total_mi += loss_mi.item() * bs

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

        train_metrics = metrics(y_true, y_prob)
        tr_loss = total_loss / len(train_loader.dataset)
        tr_cls = total_cls / len(train_loader.dataset)
        tr_mi = total_mi / len(train_loader.dataset)

        te_loss, test_metrics = evaluate(model, test_loader, bundle, device, criterion)


        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss {tr_loss:.4f} (cls {tr_cls:.4f} + mi {tr_mi:.4f}) "
            f"{format_metrics('Train', train_metrics)} | "
            f"Test Loss {te_loss:.4f} {format_metrics('Test', test_metrics)}"
        )
            
        monitor_value = test_metrics[args.monitor]
        if monitor_value > best_score:
            best_score = monitor_value
            torch.save(checkpoint_payload(model, args, epoch, test_metrics), best_path)
            print(f"Saved best checkpoint to {best_path} (epoch={epoch}, {args.monitor}={monitor_value:.4f})")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument(
        "--split_mode",
        type=str,
        default="drug_cold_start"
    )

    p.add_argument("--mm_topk", type=int, default=8)
    p.add_argument("--dd_topk", type=int, default=3)
    p.add_argument("--kg_max_hop", type=int, default=3)
    p.add_argument("--kg_max_neighbors", type=int, default=32)

    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--fixed_split_dir", type=str, default=None)
    p.add_argument("--train_neg_ratio", type=float, default=1.0)
    p.add_argument("--test_neg_ratio", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument(
        "--monitor",
        type=str,
        default="aupr",
        choices=["acc", "auc", "aupr", "precision", "recall", "f1", "mcc"],
    )

    p.add_argument("--moldeberta_dir", type=str, default="./src/MolDeBERTa-small")

    p.add_argument("--lambda_mi_m", type=float, default=0.06)
    p.add_argument("--lambda_mi_d", type=float, default=0.06)
    p.add_argument("--mi_temp", type=float, default=0.15)

    p.add_argument("--pos_sample", type=str, default="./data/MDS/pos_MDA_S.edgelist")
    p.add_argument("--neg_sample", type=str, default="./data/MDS/neg_MDA_S.edgelist")
    p.add_argument("--mirna_file", type=str, default="./data/MDS/miRNA_ID_S.xlsx")
    p.add_argument("--drug_file", type=str, default="./data/MDS/drug_ID_S.xlsx")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
