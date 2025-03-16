import torch

from dataset import create_training_datasets
from model import MoEModel
from utils import num_classes_dict

# Run this AFTER the model has been trained
# --------------
# Set seeds for reproducibility (same as training)
SEED = 42
train_datasets = create_training_datasets()
num_classes_dict = num_classes_dict(train_datasets)
# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoEModel(num_classes_dict).to(device)
model.set_seed(SEED)

# Load the saved weights
model.load_weights()

# Set model to evaluation mode
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Same dummy input as in training code
model.export(dummy_input)
