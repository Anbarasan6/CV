% Clear workspace and command window
clc;
clear;

% --- Step 1: Load Images ---
% Load the main image (larger image)
img = imread("D:\Sem_3_Lab\CV\6th_Templet\dhoni-virat.png");
% Load the template (smaller image)
template = imread("D:\Sem_3_Lab\CV\6th_Templet\virat.png");

% Check if images were loaded successfully
if isempty(img) || isempty(template)
    disp('Error: Could not open or find the image/template.');
    return;
end

% --- Step 2: Convert to Grayscale ---
% MATLAB's template matching function 'normxcorr2' works best with grayscale images.
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

if size(template, 3) == 3
    template_gray = rgb2gray(template);
else
    template_gray = template;
end

% --- Step 3: Perform Template Matching ---
% Use the normalized 2-D cross-correlation function.
% It returns a correlation matrix 'C'.
C = normxcorr2(template_gray, img_gray);

% --- Step 4: Find the Best Match Location ---
% Find the peak correlation value and its location.
[max_val, max_loc_idx] = max(C(:));
[ypeak, xpeak] = ind2sub(size(C), max_loc_idx);

% Calculate the top-left corner coordinates of the matched area.
% The output of normxcorr2 is padded, so we adjust the coordinates.
[h, w] = size(template_gray);
top_left_y = ypeak - h + 1;
top_left_x = xpeak - w + 1;

% --- Step 5: Draw a Rectangle Around the Matched Area ---
% Create an RGB version of the original image for drawing the rectangle in color.
result_img = img;
result_img = insertShape(result_img, 'Rectangle', [top_left_x, top_left_y, w, h], 'Color', 'red', 'LineWidth', 3);

% --- Step 6: Display Results ---
figure('Name', 'Template Matching Results', 'NumberTitle', 'off');

subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(result_img);
title('Template Matched');