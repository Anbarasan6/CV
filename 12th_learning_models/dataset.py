# First, ensure you have the necessary libraries installed:
# pip install scikit-learn scipy numpy

import numpy as np
from sklearn.datasets import load_digits
from scipy.io import savemat

# Load the digits dataset
digits = load_digits()

# The image data has a shape of (1797, 8, 8)
X_images = digits.images

# The corresponding labels
y_labels = digits.target

# Save the data to a .mat file
# The keys in the dictionary will become the variable names in MATLAB
mat_dict = {
    'X_images': X_images,
    'y_labels': y_labels
}

savemat('digits_data.mat', mat_dict)

print("Successfully created digits_data.mat")
