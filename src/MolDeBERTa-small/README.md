---
license: cc-by-nc-nd-4.0
datasets:
- SaeedLab/MolDeBERTa
tags:
- chemistry
- bioinformatics
- drug-discovery
- feature-extraction
- transformers
---

# MolDeBERTa-small-123M-contrastive_mtr

This model corresponds to the MolDeBERTa small architecture pretrained on the 123M dataset using the contrastive MTR pretraining objective.

\[[Github Repo](https://github.com/pcdslab/MolDeBERTa)\] | \[[Dataset on HuggingFace](https://huggingface.co/datasets/SaeedLab/MolDeBERTa)\] | \[[Model Collection](https://huggingface.co/collections/SaeedLab/moldeberta)\] | \[[Cite](#citation)\]

## Abstract

Encoder-based molecular transformer foundation models for SMILES strings have become the dominant paradigm for learning molecular representations, achieving substantial progress across a wide range of downstream chemical tasks. Despite these advances, most existing models rely on first-generation transformer architectures and are predominantly pretrained using masked language modeling—a generic objective that fails to explicitly encode physicochemical or structural information. In this work, we introduce MolDeBERTa, an encoder-based molecular framework built upon the DeBERTaV2 architecture and pretrained on large-scale SMILES data. We systematically investigate the interplay between model scale, pretraining dataset size, and pretraining objective by training 30 MolDeBERTa variants across three architectural scales, two dataset sizes, and five distinct pretraining objectives. Crucially, we introduce three novel pretraining objectives designed to inject strong inductive biases regarding molecular properties and structural similarity directly into the model's latent space. Across nine downstream benchmarks from MoleculeNet, MolDeBERTa achieves state-of-the-art performance on 7 out of 9 tasks under a rigorous fine-tuning protocol. Our results demonstrate that chemically grounded pretraining objectives consistently outperform standard masked language modeling. Finally, based on atom-level interpretability analyses, we provide qualitative evidence that MolDeBERTa learns task-specific molecular representations, highlighting chemically relevant substructures in a manner consistent with known physicochemical principles. These results establish MolDeBERTa as a robust encoder-based foundation model for chemistry-informed representation learning. 

## Model Details

MolDeBERTa is a family of encoder-based molecular foundation models built upon the DeBERTaV2 encoder architecture and pretrained on large-scale SMILES data. The framework was evaluated across three architectural scales (tiny, small, and base), pretrained on two datasets of substantially different sizes (10M and 123M molecules), and optimized using five distinct pretraining objectives, resulting in a total of 30 pretrained model variants.

## Model Selection Guide

Unsure which of the 30 models to use for your task? Based on our benchmark results, we recommend the following configurations:

* **Tiny Models:** [MolDeBERTa-tiny-123M-contrastive_mtr](https://huggingface.co/SaeedLab/MolDeBERTa-tiny-123M-contrastive_mtr) or [MolDeBERTa-tiny-123M-mtr](https://huggingface.co/SaeedLab/MolDeBERTa-tiny-123M-mtr)
* **Small Models:** [MolDeBERTa-small-123M-contrastive_mlc](https://huggingface.co/SaeedLab/MolDeBERTa-small-123M-contrastive_mlc) or [MolDeBERTa-small-123M-contrastive_mtr](https://huggingface.co/SaeedLab/MolDeBERTa-small-123M-contrastive_mtr)
* **Base Models:** [MolDeBERTa-base-123M-contrastive_mtr](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-contrastive_mtr) or [MolDeBERTa-base-123M-mtr](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr) (Overall SOTA)

## Model Usage

You can use this model for feature extraction (embeddings) or fine-tune it for downstream prediction tasks (such as property prediction or sequence classification). The embeddings may be used for similarity measurements, visualization, or training predictor models.

### Usage Example

Use the code below to get started with the model:

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model_name = "SaeedLab/MolDeBERTa-small-123M-contrastive_mtr"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input
smiles = ["CC(=O)Oc1ccccc1C(=O)O"]

# Tokenize and extract embeddings
inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
  outputs = model(**inputs)

# Access the last hidden state
embeddings = outputs.last_hidden_state
print(embeddings.shape)
```

## Citation

The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).