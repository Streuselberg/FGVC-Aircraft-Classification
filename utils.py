import json
import os
from numpy import ndarray
import onnxruntime as ort
family = 'family'
manufacturer = 'manufacturer'
variant = 'variant'


def load_label_mappings(label_mappings_path: str):
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"Label mappings file not found at {label_mappings_path}")
    with open(label_mappings_path, 'r') as f:
        return json.load(f)


def run_inference(session: ort.InferenceSession, input_name: str, input_numpy: ndarray):
    output_names = ["family_output", "manufacturer_output", "variant_output"]
    try:
        return session.run(output_names, {input_name: input_numpy})
    except Exception as e:
        raise RuntimeError(f"Failed to run inference with ONNX model: {e}")


def load_onnx_model(onnx_model_path: str):
    try:
        return ort.InferenceSession(onnx_model_path)
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model from {onnx_model_path}: {e}")


def num_classes_dict(train_datasets):
    return {
        family: len(train_datasets[family]),
        manufacturer: len(train_datasets[manufacturer]),
        variant: len(train_datasets[variant])
    }


def log_best(best_accuracy: dict[str, float]) -> None:
    print(
        f"Best: Family: {best_accuracy[family]:.2f}%,"
        f" Manufacturer: {best_accuracy[manufacturer]:.2f}%,"
        f" Variant: {best_accuracy[variant]:.2f}%")
