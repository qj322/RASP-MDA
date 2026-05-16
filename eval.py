import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import load_mda_graph_data
from train import build_model, build_parser, evaluate, format_metrics, load_checkpoint_args


def resolve_checkpoint_path(args):
    if args.checkpoint is not None:
        return args.checkpoint
    return os.path.join(args.checkpoint_dir, "best_model.pt")


def main():
    parser = build_parser()
    parser.description = "Evaluate a saved RASP-MDA checkpoint."
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    eval_args = load_checkpoint_args(checkpoint.get("args", {}), args)

    bundle = load_mda_graph_data(
        pos_sample=eval_args.pos_sample,
        neg_sample=eval_args.neg_sample,
        mirna_file=eval_args.mirna_file,
        drug_file=eval_args.drug_file,
        seed=eval_args.seed,
        test_ratio=eval_args.test_ratio,
        split_mode=eval_args.split_mode,
        sample_limit=eval_args.sample_limit,
        mm_topk=eval_args.mm_topk,
        dd_topk=eval_args.dd_topk,
        kg_max_hop=eval_args.kg_max_hop,
        kg_max_neighbors=eval_args.kg_max_neighbors,
        fixed_split_dir=eval_args.fixed_split_dir,
        train_neg_ratio=eval_args.train_neg_ratio,
        test_neg_ratio=eval_args.test_neg_ratio,
    )

    test_ds = TensorDataset(bundle.test_pairs, bundle.test_labels)
    test_loader = DataLoader(test_ds, batch_size=eval_args.batch_size, shuffle=False)

    model, _, _ = build_model(eval_args, bundle)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_metrics = evaluate(model, test_loader, bundle, device, criterion)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Test Loss {test_loss:.4f} {format_metrics('Test', test_metrics)}")


if __name__ == "__main__":
    main()
