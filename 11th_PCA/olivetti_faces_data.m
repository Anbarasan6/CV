% Clear workspace and command window
clc;
clear;

% --- Step 1: Load sample face images ---
try
    load("D:\Sem_3_Lab\CV\11th_PCA\olivetti_faces_data.mat");
    image_shape = [64, 64];
catch
    error('olivetti_faces_data.mat not found. Please create this file from the Python dataset.');
end

% --- Step 2: Apply PCA for dimensionality reduction ---
n_components_pca = 100;
% The 'pca' function returns the principal components (coeff), the reduced-
% dimension data (score), and the mean (mu).
[coeff, score, ~, ~, ~, mu] = pca(X, 'NumComponents', n_components_pca);

% --- Step 3: Reconstruct the image using PCA components ---
% Reconstruct the data by multiplying the scores by the components and adding the mean.
X_reconstructed = score * coeff' + mu;

% --- Helper function to plot original and reconstructed images ---
function plot_images(original, reconstructed, n)
    figure('Name', 'Image Reconstruction (PCA Only)', 'NumberTitle', 'off', 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    
    for i = 1:n
        % Original
        subplot(2, n, i);
        imshow(reshape(original(i, :), 64, 64), []);
        title('Original');
        
        % Reconstructed
        subplot(2, n, i + n);
        imshow(reshape(reconstructed(i, :), 64, 64), []);
        title('Reconstructed');
    end
end

% --- Step 4: Plot results ---
n_to_plot = 5;
plot_images(X, X_reconstructed, n_to_plot);