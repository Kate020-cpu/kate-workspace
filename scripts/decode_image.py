import base64
import cv2
import numpy as np
from pathlib import Path

def load_base64_image(base64_path, output_path):
    \"\"\"Load base64 encoded image and save as actual image file\"\"\"
    with open(base64_path, 'r') as f:
        img_data = f.read().strip()
    
    # Decode base64 to image
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save as actual image
    cv2.imwrite(str(output_path), img)
    print(f"Image saved to {output_path}")
    return img

if __name__ == "__main__":
    base64_file = Path("data/raw/maize/leaf_sample_base64.txt")
    output_file = Path("data/raw/maize/healthy_maize_001.jpg")
    load_base64_image(base64_file, output_file)
