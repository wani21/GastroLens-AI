import traceback
import sys
import os

from backend.model_service import model_service
import numpy as np
from PIL import Image
import io
import shap

def run_test():
    print("Loading model...")
    model_service.load()
    
    test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(test_img).save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    x = model_service.preprocess_image(image_bytes)
    background = np.repeat(x, 10, axis=0) + np.random.normal(0, 0.1, (10, 224, 224, 3))
    
    print("Explainer...")
    explainer = shap.GradientExplainer(model_service.model, background)
    shap_values = explainer.shap_values(x)
    
    print("SHAP VALUES TYPE:", type(shap_values))
    if isinstance(shap_values, list):
        print("LIST LEN:", len(shap_values))
        for i, sv in enumerate(shap_values):
            print(f"[{i}] SHAPE:", np.array(sv).shape)
    else:
        print("SHAPE:", shap_values.shape)

if __name__ == "__main__":
    run_test()
