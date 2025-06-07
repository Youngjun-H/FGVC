import torch
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights
import torchvision.models as models

# --- Basic Settings ---
TRAIN_DIR = 'efficientnet/dataset/train'  # Modify with actual training data folder path
NUM_CLASSES = 396
VALIDATION_SPLIT = 0.2

# --- Model Settings ---
MODEL_VARIANT = 'B0'  # Choose from 'B0', 'B1', 'B2', 'B3'

# Model configuration based on variant
MODEL_CONFIG = {
    'B0': {
        'img_size': (224, 224),
        'weights': EfficientNet_B0_Weights.IMAGENET1K_V1,
        'model_loader': models.efficientnet_b0
    },
    'B1': {
        'img_size': (240, 240),
        'weights': EfficientNet_B1_Weights.IMAGENET1K_V1,
        'model_loader': models.efficientnet_b1
    },
    'B2': {
        'img_size': (260, 260),
        'weights': EfficientNet_B2_Weights.IMAGENET1K_V1,
        'model_loader': models.efficientnet_b2
    },
    'B3': {
        'img_size': (300, 300),
        'weights': EfficientNet_B3_Weights.IMAGENET1K_V1,
        'model_loader': models.efficientnet_b3
    }
}

if MODEL_VARIANT not in MODEL_CONFIG:
    raise ValueError(f"Unsupported EfficientNet variant: {MODEL_VARIANT}")

# Get model configuration
IMG_SIZE = MODEL_CONFIG[MODEL_VARIANT]['img_size']
WEIGHTS = MODEL_CONFIG[MODEL_VARIANT]['weights']
MODEL_LOADER = MODEL_CONFIG[MODEL_VARIANT]['model_loader']

# --- Training Settings ---
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 50
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5
EARLY_STOPPING_PATIENCE = 10

# --- Device Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 