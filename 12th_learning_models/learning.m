% Clear workspace and command window
clc;
clear;

% --- Step 1: Load sample image dataset (digits) ---
try
    % Use a specific path if the file is not in the current directory.
    % The 'load' command is the correct way to load a .mat file.
    load("D:\Sem_3_Lab\CV\12th_learning_models\digits_data.mat");
    
    % Ensure data is of a suitable type.
    X_images = double(X_images); 
    
    % Ensure labels are a numerical vector, as TreeBagger requires this.
    % The 'digits' dataset in Python loads labels as integers.
    if iscell(y_labels)
        y_labels = cellfun(@str2double, y_labels);
    end
    y_labels = double(y_labels);
    
catch
    error('digits_data.mat not not found at the specified path.');
end

% Flatten images for classifier input
num_samples = size(X_images, 1);
X_flat = reshape(X_images, num_samples, []);

% --- Step 2: Train-test split ---
cv = cvpartition(num_samples, 'HoldOut', 0.3);
idx_train = training(cv);
idx_test = test(cv);
X_train = X_flat(idx_train, :);
y_train = y_labels(idx_train);
X_test = X_flat(idx_test, :);
y_test = y_labels(idx_test);

% --- Step 3: Train a classifier (e.g., Random Forest) ---
% TreeBagger is the MATLAB equivalent of a Random Forest Classifier.
num_estimators = 100;
model = TreeBagger(num_estimators, X_train, y_train, 'Method', 'classification');

% --- Step 4: Predict on test set ---
% 'predict' returns a cell array of strings by default for classification
y_pred_str = predict(model, X_test);

% --- Step 5: Evaluate the model ---
% Convert the predicted labels from a cell array of strings to a numerical array.
y_pred = str2double(y_pred_str);

% Calculate accuracy.
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Accuracy: %.4f\n', accuracy);

% Generate the confusion matrix.
% Both inputs must be numerical and have the same data type.
conf_mat = confusionmat(y_test, y_pred);

% Display confusion matrix using a heatmap.
figure;
heatmap(conf_mat, 'XData', 0:9, 'YData', 0:9, 'ColorbarVisible', 'off');
title('Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');