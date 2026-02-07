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

Key configuration parameters in `configs/default.yaml`:

```yaml
model:
  vision_model: "openai/clip-vit-base-patch32"
  text_model: "gpt2-medium"
  projection_dim: 512
  temperature: 0.5
  freeze_vision_backbone: true

training:
  stage1:
    batch_size: 8
    learning_rate: 5.0e-5
    num_epochs: 10
    gradient_accumulation_steps: 4
  stage2:
    batch_size: 8
    learning_rate: 1.0e-5
    num_epochs: 5
    dpo_beta: 0.1
```

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
preference-guided-image-captioning-alignment/
├── src/preference_guided_image_captioning_alignment/
│   ├── data/              # Data loading and preprocessing
│   │   ├── loader.py      # ConceptualCaptions & UltraFeedback datasets
│   │   └── preprocessing.py  # Image and text processors
│   ├── models/            # Model architecture and losses
│   │   └── model.py       # Vision/text encoders, contrastive & DPO losses
│   ├── training/          # Training pipeline
│   │   └── trainer.py     # Dual-stage trainer with NaN safety
│   ├── evaluation/        # Metrics and evaluation
│   │   └── metrics.py     # BLEU, ROUGE, CIDEr, BERTScore, CLIP-Score
│   └── utils/             # Configuration and utilities
│       └── config.py      # YAML config management
├── scripts/
│   ├── train.py           # Main training script
│   └── run_evaluation.py  # Evaluation script
├── tests/                 # Unit and integration tests
├── configs/               # Training configurations
│   └── default.yaml       # Default hyperparameters
└── requirements.txt       # Python dependencies
```

## API Usage

```python
from preference_guided_image_captioning_alignment.models.model import (
    PreferenceGuidedCaptioningModel,
)
from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
)

# Initialize model
model = PreferenceGuidedCaptioningModel(
    vision_model="openai/clip-vit-base-patch32",
    text_model="gpt2-medium",
    projection_dim=512,
)

# Process inputs
image_processor = ImageProcessor(image_size=224)
text_processor = TextProcessor(model_name="gpt2-medium", max_length=128)

image_tensor = image_processor.process_image("image.jpg")
caption_encoding = text_processor.encode_caption("A cat sitting on a chair")
```

## Evaluation Metrics

- **Caption Quality**: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, CIDEr
- **Semantic Similarity**: BERTScore, CLIP-Score
- **Preference Alignment**: Win rate vs. baseline captions
- **Diversity**: Unique n-grams, caption variety

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ --cov=src/preference_guided_image_captioning_alignment
```

## Technical Details

### Model Architecture
- **Vision encoder**: CLIP ViT-B/32 (88M parameters, frozen)
- **Text encoder**: GPT-2 Medium (355M parameters, trainable)
- **Caption decoder**: GPT-2 LM head with cross-attention
- **Projection layers**: Vision and text projections to shared 512-dim space
- **Total**: 867M parameters (779M trainable)

### Training Strategy
1. **Stage 1**: Joint vision-text representation learning via NT-Xent contrastive loss with cosine learning rate schedule
2. **Stage 2**: Preference alignment via DPO loss on chosen/rejected caption pairs with warmup schedule
3. **NaN Safety**: Automatic detection and skipping of batches producing NaN loss or gradients

### Design Decisions
- **GPT-2 Medium over DialoGPT**: DialoGPT-medium produces NaN gradients in LayerNorm backward passes; GPT-2 Medium is numerically stable with identical architecture
- **Temperature 0.5**: Lower temperatures (e.g. 0.07) amplify gradients ~14x through the contrastive loss, causing instability; 0.5 provides stable 2x amplification
- **Frozen vision backbone**: CLIP's pretrained representations are strong enough; freezing reduces memory usage and training time while preventing catastrophic forgetting

## License

MIT License - see LICENSE file for details.