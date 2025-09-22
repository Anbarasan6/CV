% --- Step 1: Load Image ---
clc;
clear;
img_filename = "D:\Sem_3_Lab\CV\7th_Harris\chess.png";
img = imread(img_filename);

% --- Step 2: Grayscale Conversion ---
% The detectHarrisFeatures function requires a grayscale image.
if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end

% --- Step 3: Harris Corner Detection ---
% Use detectHarrisFeatures to find corners.
% We can adjust 'MinQuality' to control the number of detected corners.
% The output is a cornerPoints object.
corners = detectHarrisFeatures(gray_img, 'MinQuality', 0.01);

% --- Step 4: Draw Corners on a New Image ---
% Create a copy of the original image to draw on.
img_corners = img;

% Convert the cornerPoints object to a list of coordinates
corner_locations = corners.Location;

% Use insertShape to draw filled circles on the image.
% [x y radius]
radii = 8;
circle_positions = [corner_locations(:,1), corner_locations(:,2), repmat(radii, size(corner_locations, 1), 1)];
img_corners = insertShape(img_corners, 'FilledCircle', circle_positions, 'Color', 'red', 'Opacity', 1);


% --- Step 5: Display Results ---
figure('Name', 'Harris Corner Detection', 'NumberTitle', 'off');

% Subplot 1: Original Image
subplot(1, 2, 1);
imshow(img);
title('Original Image');

% Subplot 2: Harris Corners
subplot(1, 2, 2);
imshow(img_corners);
title('Harris Corners (Large Red)');

% Adjust layout for better display
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);