import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load Image ---
# Replace 'your_image.jpg' with the actual path to your image file.
# The 'cameraman.tif' is a standard test image.
img = cv2.imread('zebra.jpg')
if img is None:
    print("Error: Could not find or read the image file.")
    exit()

# --- Step 2: Grayscale Conversion ---
# Convert the image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 3: Edge Detection Algorithms ---

# 1. Canny Edge Detection
# Canny is a multi-stage algorithm that uses hysteresis thresholding.
edges_canny = cv2.Canny(gray, 100, 200)

# 2. Sobel Edge Detection
# The Sobel operator uses two kernels to compute gradients in the x and y directions.
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobelx, sobely)

# 3. Prewitt Edge Detection
# Prewitt is similar to Sobel but uses a different set of kernels.
# OpenCV does not have a built-in Prewitt function, so we define the kernels manually.
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewittx = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
prewitty = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
edges_prewitt = cv2.magnitude(prewittx, prewitty)

# 4. Laplacian Edge Detection (LoG - Laplacian of Gaussian)
# LoG is sensitive to noise, so we first blur the image and then apply the Laplacian.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges_laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# 5. LoG (Laplacian of Gaussian)
# This is a direct implementation of LoG using a larger filter.
# It is often done by convolving with a LoG kernel.
# We'll use a direct Laplacian function on a blurred image.
# A more explicit LoG would require a custom kernel, but this is a common approximation.
edges_log = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_64F)

# 6. DoG (Difference of Gaussians)
# This approximates the LoG by subtracting two Gaussian-filtered images.
gaussian1 = cv2.GaussianBlur(gray, (5, 5), 1)
gaussian2 = cv2.GaussianBlur(gray, (5, 5), 2)
dog = gaussian1.astype(np.float32) - gaussian2.astype(np.float32)
# Normalize for display
edges_dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
edges_dog = edges_dog.astype(np.uint8)

# --- Step 4: Display Results ---
plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(edges_sobel, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(edges_prewitt, cmap='gray')
plt.title('Prewitt')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(edges_laplacian, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(edges_log, cmap='gray')
plt.title('LoG')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(edges_dog, cmap='gray')
plt.title('DoG')
plt.axis('off')

plt.tight_layout()
plt.show()
