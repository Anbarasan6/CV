% Clear workspace and command window
clc;
clear;

% --- Step 1: Load Image ---
% Replace 'bird_2.jpeg' with the actual path to your image file.
img = imread("D:\Sem_3_Lab\CV\5th_Histogram\bird_2.jpeg");

% Ensure the image is grayscale if it's not already
if size(img, 3) == 3
    img = rgb2gray(img);
end

% --- Step 2: Histogram Equalization ---
equalized = histeq(img);

% --- Step 3: Combine Images for Display ---
% Concatenate the original and equalized images side by side.
combined = [img, equalized];

% --- Step 4: Calculate Histogram and CDF ---
% The 'imhist' function returns the histogram counts.
[hist_counts, ~] = imhist(img);
% The 'cumsum' function calculates the cumulative sum to get the CDF.
cdf = cumsum(hist_counts);
% Normalize the CDF for plotting.
cdf_normalized = cdf * max(hist_counts) / max(cdf);

% --- Step 5: Plotting ---
figure('Name', 'Histogram Equalization', 'NumberTitle', 'off');

% Left Subplot: Original and Equalized images side by side
subplot(1, 2, 1);
imshow(combined);
title('Original (Left) & Equalized (Right)');

% Right Subplot: Histogram and CDF
subplot(1, 2, 2);
% Use 'yyaxis' to plot two different y-axes on the same plot.
yyaxis left;
bar(hist_counts, 'r');
ylabel('Frequency');
hold on;

yyaxis right;
plot(cdf_normalized, 'b', 'LineWidth', 1.5);
ylabel('Normalized CDF');

hold off;
title('Histogram & CDF');
xlabel('Pixel Intensity');
legend('Histogram', 'CDF');

% Ensure a tight layout for better visualization
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);