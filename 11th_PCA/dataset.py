# First, ensure you have the necessary libraries installed:
# pip install scikit-learn scipy numpy

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from scipy.io import savemat

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)

# The data is a NumPy array where each row is a flattened image
X = faces.data

# You can also get the images as 64x64 arrays
images = faces.images

# The target labels (person IDs)
y = faces.target

# Save the data to a .mat file
# The keys in the dictionary will be the variable names in MATLAB
mat_dict = {
    'X': X,
    'images': images,
    'y': y
}

savemat('olivetti_faces_data.mat', mat_dict)

print("Successfully created olivetti_faces_data.mat")
