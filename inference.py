# NAME: Inference.py
# Load an already trained model and run inference on an input image
# -----------------------------------------------------------------
import argparse
import json
import os
import numpy as np
from transforms import val_transform, load_and_preprocess_image
from utils import load_label_mappings, run_inference, load_onnx_model

label_mappings_path_leaf = "label_mappings.json"
onnx_model_path_leaf = "moe_efficientnet_b3.onnx"

parser = argparse.ArgumentParser(description="Run inference on an ONNX model.")
parser.add_argument("image_path",
                    help="Path to the image file to run inference on.",
                    type=str)
parser.add_argument("--deploy",  # flag for if script is being
                    action="store_true",  # take flag's presence as True
                    help="Configure the script for use as a script outside the directory in which the model"
                         "was trained.")
parser.add_argument("--label_mappings_path",
                    default=label_mappings_path_leaf,
                    help="Path to label_mappings.json in deployment mode")
args = parser.parse_args()
# Determine mode
deploy_mode = args.deploy or os.environ.get("RUN_MODE", "").lower() == "deploy"

# DEBUG:

# Set file paths based on mode
# In deployment mode, use the provided label_mappings_path or assume it's in the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
label_mappings_path = os.path.join(os.getcwd(), "Data",
                                   "label_mappings.json") if deploy_mode and args.label_mappings_path \
    else args.label_mappings_path
onnx_model_path = os.path.join(script_dir, onnx_model_path_leaf)

label_mappings = load_label_mappings(label_mappings_path)

# Load and preprocess the image
image = load_and_preprocess_image(args.image_path)
input_tensor = val_transform(image).unsqueeze(0)
# Convert to numpy for ONNX runtime
input_numpy = input_tensor.numpy()
# create/run model session
session = load_onnx_model(onnx_model_path)

# Get the input name for the model
input_name = session.get_inputs()[0].name
# Run inference and extract outputs
family_output, manufacturer_output, variant_output = run_inference(session, input_name, input_numpy)

# Get predicted Class (Family, Manufacturer, Variant) indices
family_pred = np.argmax(family_output, axis=1)[0]
manufacturer_pred = np.argmax(manufacturer_output, axis=1)[0]
variant_pred = np.argmax(variant_output, axis=1)[0]

# map only predicted index to label, use generator to avoid memory of list comprehension
result = {
    'family': next(k for k, v in label_mappings['family'].items() if v == family_pred),
    'manufacturer': next(k for k, v in label_mappings['manufacturer'].items() if v == manufacturer_pred),
    'variant': next(k for k, v in label_mappings['variant'].items() if v == variant_pred)
}

# Print result as JSON (to be captured from stdout)
print(json.dumps(result))
