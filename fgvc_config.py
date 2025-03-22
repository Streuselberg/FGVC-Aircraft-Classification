"""
NAME: fgvc_config.py
houses utility / configuration files for the project. Change these to adapt this model to your needs
"""
from dataclasses import dataclass
from enum import auto, Enum, IntEnum
from os import path
import torch


class ExpertName(IntEnum):
    family = auto()
    manufacturer = auto()
    variant = auto()


@dataclass(frozen=True)
class Config:
    """All the settings for the project"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TRAINING SETTINGS
    BATCH_SIZE = 16
    EPOCHS = 20
    EXPORT_MODEL_TO_ONNX = False
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    STEP_SIZE = 7
    GAMMA = 0.1
    # Filenames
    MODEL_FILE_NAME = "moe_efficientnet_b3"
    LABEL_MAPPINGS_FILE_NAME = "label_mappings.json"
    EXPORT_PATH_FILE_NAME = f"{MODEL_FILE_NAME}.onnx"
    # Directories
    DATA_DIR = r'..\Engine\Data\FGVC'
    IMAGES_DIR = path.join(DATA_DIR, 'images')
    # Paths
    MODEL_WEIGHTS_PATH = path.join("models", f"{MODEL_FILE_NAME}.pth")
    EXPORT_PATH = path.join("models", EXPORT_PATH_FILE_NAME)
    expert_weights_paths = {
        name.name: path.join("weights", name.name + "_expert.pth") for name in ExpertName
    }
