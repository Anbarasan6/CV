import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread("bird_2.jpeg", 0)

# Histogram equalization
equalized = cv2.equalizeHist(img)

# Calculate histogram for original image
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

# Compute CDF
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()  # normalize for plotting

# Plotting
plt.figure(figsize=(12, 4))

# Left: Original and Equalized images side by side
plt.subplot(1, 2, 1)
combined = np.hstack((img, equalized))
plt.imshow(combined, cmap='gray')
plt.title('Original (Left) & Equalized (Right)')
plt.axis('off')

# Right: Histogram and CDF
plt.subplot(1, 2, 2)
plt.plot(cdf_normalized, color='blue', label='CDF')
plt.hist(img.ravel(), bins=256, range=(0, 256), color='red', alpha=0.4, label='Histogram')
plt.title('Histogram & CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
