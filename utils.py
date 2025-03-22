"""
NAME: Utils
Contains utility functions for use in multiple scripts
"""
from json import load
from os import path
from numpy import ndarray
import onnxruntime as ort

from fgvc_config import ExpertName


def load_label_mappings(label_mappings_path: str) -> dict:
    """
    Load lapel mappings from disk
    :param label_mappings_path: str path for data
    :return: label mappings dictionary
    """
    if not path.exists(label_mappings_path):
        raise FileNotFoundError(f"Label mappings file not found at {label_mappings_path}")
    with open(label_mappings_path, 'r', encoding='utf-8') as f:
        return load(f)


def run_inference(session: ort.InferenceSession, input_name: str, input_numpy: ndarray) -> tuple:
    """
    runs inference on an input tensor
    :param session: the instance of the model you want to use for inference
    :param input_name: the name of the input tensor
    :param input_numpy: the input numpy array
    :return:
    """
    output_names = [expert.name + "_output" for expert in ExpertName]
    try:
        return session.run(output_names, {input_name: input_numpy})
    except Exception as e:
        raise RuntimeError(f"Failed to run inference with ONNX model: {e}") from e


def load_onnx_model(onnx_model_path: str):
    """
    Loads the ONNX model from disk and returns it as an ONNX runtime inference session
    :param onnx_model_path:
    :return:
    """
    try:
        return ort.InferenceSession(onnx_model_path)
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model from {onnx_model_path}: {e}") from e


def num_classes_dict(train_datasets: dict):
    """generate a dictionary mapping of expert name to number of classes used in training"""
    return {
        expert.name: len(train_datasets[expert.name]) for expert in ExpertName
    }


def log_best(best_accuracy: dict[ExpertName, float]) -> None:
    """
    log the best accuracies seen so far for each expert
    """
    accuracies = ", ".join(f"{name.name.capitalize()}: {best_accuracy[name]:.2f}%"
                           for name in ExpertName)
    print("Best: " + accuracies)
