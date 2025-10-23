import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Step 1: Load sample face images
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data 
image_shape = (64, 64)

# Step 2: Apply PCA for dimensionality reduction
model = PCA(n_components=100)
X_pca = model.fit_transform(X)

X_reconstructed = model.inverse_transform(X_pca)

# Helper function to plot original and reconstructed images
def plot_images(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(image_shape), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(image_shape), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Step 5: Plot results
plot_images(X, X_reconstructed)
