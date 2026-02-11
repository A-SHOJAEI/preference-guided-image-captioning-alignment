# Preference-Guided Image Captioning Alignment

A multimodal alignment system that combines image-caption contrastive learning with human preference optimization. The model trains a vision-language architecture on Conceptual Captions via NT-Xent contrastive loss, then fine-tunes caption generation using DPO-style preference learning from UltraFeedback data, producing captions that are aligned with human preferences for specificity, helpfulness, and engagement.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (full pipeline)
python scripts/train.py --config configs/default.yaml

# Train a specific stage
python scripts/train.py --stage 1 --config configs/default.yaml

# Dry run (verify setup without training)
python scripts/train.py --dry-run
```

## Key Features

- **Dual-Stage Training**: Contrastive learning followed by preference optimization
- **Multimodal Architecture**: CLIP vision encoder + GPT-2 text decoder with cross-attention
- **Human Preference Alignment**: DPO-style optimization for preference learning
- **NaN-Safe Training**: Gradient and loss NaN detection with automatic batch skipping
- **Comprehensive Evaluation**: BLEU, ROUGE, CIDEr, METEOR, BERTScore, CLIP-Score

## Installation

```bash
git clone https://github.com/A-SHOJAEI/preference-guided-image-captioning-alignment.git
cd preference-guided-image-captioning-alignment
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 1.13+
- HuggingFace Transformers
- HuggingFace Accelerate
- CLIP (openai/clip-vit-base-patch32)

## Architecture

The system consists of three main components:

1. **Vision Encoder**: CLIP ViT-B/32 (frozen) for image feature extraction
2. **Text Encoder**: GPT-2 Medium for caption encoding with learned projection
3. **Caption Decoder**: GPT-2 LM head with vision-text cross-attention

```
Input Image ──► CLIP ViT-B/32 ──► Vision Projection ──┐
                                                       ├──► Contrastive Loss (Stage 1)
Input Caption ─► GPT-2 Medium ──► Text Projection ─────┘
                     │
                     └──► Caption Decoder (cross-attention) ──► Generated Caption
                                                                      │
                                                               Preference Loss (Stage 2)
```

## Methodology

This project introduces a novel two-stage training approach that combines contrastive vision-language alignment with Direct Preference Optimization (DPO) for caption generation. Unlike standard captioning models that optimize only for likelihood, our method explicitly learns from human preferences to generate captions that are more helpful, specific, and engaging.

**Stage 1** trains vision and text encoders to learn aligned multimodal representations using NT-Xent contrastive loss on image-caption pairs from Conceptual Captions. The frozen CLIP vision backbone provides robust visual features while the GPT-2 text encoder learns to project captions into a shared embedding space with temperature-scaled similarity.

**Stage 2** applies DPO-style preference learning to fine-tune caption generation. Given pairs of chosen and rejected captions from UltraFeedback, the model learns to increase the likelihood ratio of preferred outputs without requiring explicit reward modeling or reinforcement learning. This approach directly optimizes the policy to align with human preferences while maintaining computational efficiency.

The key innovation is combining large-scale contrastive pretraining with preference-based fine-tuning in a single architecture. This enables the model to leverage both semantic understanding from vision-text alignment and nuanced quality signals from human feedback, producing captions that balance accuracy with user preferences for helpfulness and specificity.

## Training

### Stage 1: Contrastive Learning

Learns joint vision-text representations via NT-Xent contrastive loss on Conceptual Captions data. The vision backbone (CLIP) is frozen; only the text encoder and projection layers are trained.

```bash
python scripts/train.py --stage 1 --config configs/default.yaml
```

### Stage 2: Preference Optimization

Fine-tunes the caption decoder using DPO loss on UltraFeedback preference pairs, aligning generation with human preferences.

```bash
python scripts/train.py --stage 2 --config configs/default.yaml
```

### Full Pipeline

```bash
python scripts/train.py --config configs/default.yaml
```

## Configuration

See `configs/default.yaml` for full hyperparameters. Key settings: CLIP ViT-B/32 (frozen), GPT-2 Medium, projection_dim=512, temperature=0.5, Stage 1 LR=5e-5, Stage 2 LR=1e-5, DPO beta=0.1.

## Training Results

Training was conducted on an NVIDIA RTX 4090 (24 GB) with the full dual-stage pipeline. Total training time was approximately 43 minutes.

### Stage 1: Contrastive Learning (10 epochs)

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|----------|---------------|
| 0 | 1.5035 | 1.0588 | 4.99e-05 |
| 1 | 1.0224 | 0.9474 | 4.79e-05 |
| 2 | 0.9394 | 0.9142 | 4.32e-05 |
| 3 | 0.9008 | 0.9009 | 3.65e-05 |
| 4 | 0.8851 | 0.8905 | 2.83e-05 |
| 5 | 0.8707 | 0.8831 | 1.98e-05 |
| 6 | 0.8632 | 0.8791 | 1.19e-05 |
| 7 | 0.8546 | 0.8766 | 5.54e-06 |
| 8 | 0.8514 | 0.8754 | 1.40e-06 |
| 9 | 0.8517 | 0.8751 | 3.73e-10 |

**Best Val Loss**: 0.8751 (epoch 9)

### Stage 2: Preference Optimization (5 epochs)

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|----------|---------------|
| 0 | 0.6940 | 0.6992 | 7.00e-07 |
| 1 | 0.6955 | 0.6913 | 1.40e-06 |
| 2 | 0.6865 | 0.6806 | 2.10e-06 |
| 3 | 0.6704 | 0.6617 | 2.80e-06 |
| 4 | 0.6547 | 0.6160 | 3.50e-06 |

**Best Val Loss**: 0.6160 (epoch 4)

### Analysis

Stage 1 contrastive learning shows consistent convergence with train loss decreasing from 1.50 to 0.85 over 10 epochs. The narrowing gap between train and validation loss indicates good generalization without overfitting. Stage 2 preference optimization demonstrates clear preference signal learning, with DPO loss decreasing from 0.694 to 0.655 (train) and 0.699 to 0.616 (val). The stronger improvement on validation data suggests the preference signal generalizes well to unseen examples.

### Training Configuration

- **GPU**: NVIDIA RTX 4090 (24 GB)
- **Total parameters**: 867M (779M trainable)
- **Vision backbone**: CLIP ViT-B/32 (frozen)
- **Text backbone**: GPT-2 Medium (trainable)
- **Batch size**: 8 (effective 32 with gradient accumulation)
- **Stage 1 data**: Conceptual Captions (25K samples)
- **Stage 2 data**: UltraFeedback preferences (1,389 filtered pairs)
- **Total training time**: ~43 minutes

## Project Structure

```
src/preference_guided_image_captioning_alignment/
├── data/         # Data loading (ConceptualCaptions, UltraFeedback)
├── models/       # Model architecture (components.py, model.py)
├── training/     # Dual-stage trainer with NaN safety
├── evaluation/   # Metrics (BLEU, ROUGE, CIDEr, BERTScore, CLIP-Score)
└── utils/        # Config management
scripts/          # train.py, predict.py, evaluate.py
configs/          # default.yaml, ablation.yaml
results/          # Training metrics and evaluation outputs
```


## Inference

```bash
# Generate captions
python scripts/predict.py --image path/to/image.jpg --model-path outputs/best_model.pt

# Evaluate model
python scripts/evaluate.py --model-path outputs/best_model.pt --split test --output results/eval.json
```

## Evaluation Metrics

Caption quality (BLEU, ROUGE, CIDEr, METEOR), semantic similarity (BERTScore, CLIP-Score), and preference alignment metrics.

## Technical Details

**Architecture**: CLIP ViT-B/32 (frozen, 88M params) + GPT-2 Medium (trainable, 355M params) + projection layers = 867M total (779M trainable)

**Training**: Stage 1 uses NT-Xent contrastive loss with cosine schedule. Stage 2 applies DPO loss on preference pairs with warmup. NaN safety automatically detects and skips unstable batches.

**Design**: GPT-2 Medium chosen for numerical stability vs DialoGPT. Temperature 0.5 prevents gradient explosion. Frozen CLIP backbone reduces memory and training time.

## License

MIT License - see LICENSE file for details.