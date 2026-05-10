import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model(
    "saved_models/best_model.keras",
    compile=False
)

# Basic info
print("Output shape:", model.output_shape)
print("Input shape:", model.input_shape)

# Find convolution-like feature map layers
conv_layers = []

for layer in model.layers:
    try:
        if len(layer.output.shape) == 4:
            conv_layers.append(layer.name)
    except:
        pass

print("Last conv layer:", conv_layers[-1])

# Verify important layers
layer_names = [l.name for l in model.layers]

print("se_combine_relu exists:", "se_combine_relu" in layer_names)
print("gap exists:", "gap" in layer_names)