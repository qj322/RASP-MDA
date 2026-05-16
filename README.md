# RASP-MDA


## Drug-structure channel

Input: SMILES

- Molecular graph features:
  - atom features (type/degree/charge/H/aromatic)
  - bond features (bond type/conjugation/ring)
  - shortest-path distance encoding (node-wise distance histogram)
- Encoder: lightweight GraphTransformer-mol (`TransformerConv` + residual + mean/max pooling)
- Language-model branch: MolDeBERTa embedding from SMILES
- Fusion: gated fusion of `GraphTransformer-mol` and MolDeBERTa outputs

## miRNA channel

- CNN embedding (local motifs)
- RNA-FM embedding
- Gated fusion of CNN and RNA-FM

## Heterogeneous graph channel

RASP-MDA uses a target-edge unseen graph: candidate miRNA/drug nodes are visible through sequence, structure, and similarity context, while target miRNA-drug association edges are built from train positives only.

Relations built from current dataset:
- `R_md`: miRNA-drug edges (train positives only)
- `R_mm`: miRNA-miRNA similarity
- `R_dd`: drug-drug similarity

Entity features are propagated by the relation-aware spatial attention encoder and fused back with sequence/structure embeddings.

## Training

Default quick smoke test:

```bash
/opt/anaconda3/envs/drug/bin/python train.py --epochs 2
```

The default `split_mode` is `drug_cold_start`, and both `RNA-FM` and `MolDeBERTa` are enabled by default.

Run the generalized split settings:

```bash
/opt/anaconda3/envs/drug/bin/python train.py --split_mode drug_scaffold
```

Default local MolDeBERTa directory:

```bash
/opt/anaconda3/envs/drug/bin/python train.py --epochs 1 --moldeberta_dir ./src/MolDeBERTa-small
```

The graph remains target-edge unseen under fixed splits: train/test association pairs are fixed by the split files, and similarity edges are rebuilt from available miRNA sequences and drug structures.

Do not set `--sample_limit` together with `--fixed_split_dir`.


Evaluate the saved best checkpoint:

```bash
/opt/anaconda3/envs/drug/bin/python eval.py
```

## Main files

- `src/dataset.py`: graph/data construction
- `src/model.py`: RASP-MDA model
- `train.py`: train/eval entrypoint
- `eval.py`: load best checkpoint and print evaluation metrics
