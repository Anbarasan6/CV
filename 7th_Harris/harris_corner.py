import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
filename = 'chess.jpg'
img = cv2.imread(filename)
img_corners = img.copy()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate to make corners more pronounced
dst = cv2.dilate(dst, None)

# Draw larger red circles on corners
corner_points = np.argwhere(dst > 0.01 * dst.max())
for pt in corner_points:
    y, x = pt
    cv2.circle(img_corners, (x, y), 8, (0, 0, 255), -1)  # radius=8, red, filled

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_corners_rgb = cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB)

# Plot original vs corners detected
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_corners_rgb)
plt.title("Harris Corners (Large Red)")
plt.axis("off")

plt.tight_layout()
plt.show()
