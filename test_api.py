import requests
import io
import numpy as np
from PIL import Image

def test_endpoints():
    test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(test_img).save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    files = {'file': ('test.png', image_bytes, 'image/png')}
    print("Testing SHAP...")
    res = requests.post("http://127.0.0.1:8000/shap", files=files)
    print("SHAP Status:", res.status_code)
    if res.status_code == 200:
        print("SHAP keys:", res.json().keys())
    else:
        print("SHAP error:", res.text)
        
    files = {'file': ('test.png', image_bytes, 'image/png')}
    print("Testing Segment...")
    res = requests.post("http://127.0.0.1:8000/segment", files=files)
    print("Segment Status:", res.status_code)
    if res.status_code == 200:
        print("Segment keys:", res.json().keys())
    else:
        print("Segment error:", res.text)

if __name__ == "__main__":
    test_endpoints()
