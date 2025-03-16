import json
import random
from os import path

import torch

from dataset import create_training_datasets, create_label_mappings, \
    create_validation_datasets
from model import MoEModel
from utils import num_classes_dict, log_best

# PARAMETERS
# >========<
FULLY_BAKED = False
"""Control export of model to Onnx format"""

NUM_EPOCHS = 20
"""Controls the number of epochs to train the model"""

weights_path = "weights/moe_efficientnet_b3.pth"
"""path where the model's weights will be saved."""

label_mappings_path = "label_mappings.json"
"""path where the labels will be saved."""
# ========<

# Set Seed(s)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# set BATCH_SIZE here (if needed, default is 16)
training_datasets = create_training_datasets()
val_datasets = create_validation_datasets()
num_classes_dict = num_classes_dict(training_datasets)
# Create model and load weights and load to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoEModel(num_classes_dict).to(device)
if path.exists(model.weights_path):
    model.load_weights()

# Save the mappings for reference at inference time
label_mappings = create_label_mappings(training_datasets)

with open(label_mappings_path, 'w') as f:
    json.dump(label_mappings, f)
best_acc = model.train_for(NUM_EPOCHS, device)

# Training Finished

# Only export to Onnx when you're happy with the model
if FULLY_BAKED:
    torch.onnx.export(model, torch.randn(1, 3, 224, 224).to(device),
                      "moe_efficientnet_b3.onnx",
                      input_names=["input"],
                      output_names=["family_output", "manufacturer_output", "variant_output"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "family_output": {0: "batch_size"},
                                    "manufacturer_output": {0: "batch_size"},
                                    "variant_output": {0: "batch_size"}},
                      opset_version=11)

log_best(best_acc)