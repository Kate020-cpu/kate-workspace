import cv2
import numpy as np
import sys

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Create a simple test image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (25, 25), (75, 75), (0, 255, 0), -1)
cv2.imwrite('test_output.jpg', img)
print("✅ Test image created: test_output.jpg")

# Try to read our leaf image (if it exists)
try:
    leaf = cv2.imread('data/raw/maize/healthy_maize_001.jpg')
    if leaf is not None:
        print(f"✅ Leaf image loaded successfully! Shape: {leaf.shape}")
    else:
        print("⚠️ Leaf image not found yet - will be created when we decode")
except Exception as e:
    print(f"⚠️ Could not load leaf image: {e}")

print('Testing caching mechanism...')
