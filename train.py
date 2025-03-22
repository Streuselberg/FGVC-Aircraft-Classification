import json
import random
from os import path

import torch
from rich.console import Console

from fgvc_config import Config
from dataset import create_training_datasets, create_label_mappings, \
    create_validation_datasets
from model import MoEModel
from utils import num_classes_dict, log_best


def main():
    config = Config()
    # PARAMETERS
    # >========<
    fully_baked = config.EXPORT_MODEL_TO_ONNX
    """Control export of model to Onnx format"""

    num_epochs = config.EPOCHS
    """Controls the number of epochs to train the model"""

    label_mappings_path = config.LABEL_MAPPINGS_FILE_NAME
    """path where the labels will be saved."""
    # ========<

    # Set Seed(s)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set BATCH_SIZE here (if needed, default is 16)
    training_datasets: dict = create_training_datasets(config)
    val_datasets: dict = create_validation_datasets(config)
    num_classes_map = num_classes_dict(training_datasets)
    # Create model and load weights and load to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoEModel(num_classes_map, config).to(device)
    if path.exists(config.MODEL_WEIGHTS_PATH):
        model.load_weights()

    # Save the mappings for reference at inference time
    label_mappings = create_label_mappings(training_datasets)

    with open(label_mappings_path, 'w', encoding='utf-8') as f:
        json.dump(label_mappings, f)
    console = Console()
    best_acc = model.train_for(num_epochs, device, console)

    # Training Finished

    # Only export to Onnx when you're happy with the model
    if fully_baked:
        torch.onnx.export(model, torch.randn(1, 3, 224, 224).to(device),
                          config.EXPORT_PATH_FILE_NAME,
                          input_names=["input"],
                          output_names=["family_output", "manufacturer_output", "variant_output"],
                          dynamic_axes={"input": {0: "batch_size"},
                                        "family_output": {0: "batch_size"},
                                        "manufacturer_output": {0: "batch_size"},
                                        "variant_output": {0: "batch_size"}},
                          opset_version=11)

    log_best(best_acc)


if __name__ == "__main__":
    main()
