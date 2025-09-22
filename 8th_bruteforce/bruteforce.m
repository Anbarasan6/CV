% Clear workspace and command window
clc;
clear;

% --- Step 1: Load Images ---
% Load the two images to be matched.
img1 = imread("D:\Sem_3_Lab\CV\8th_bruteforce\high-school.jpg");
img2 = imread("D:\Sem_3_Lab\CV\8th_bruteforce\school.jpg");

% Check if images were loaded successfully
if isempty(img1) || isempty(img2)
    disp('Error: Could not open or find the images.');
    return;
end

% --- Step 2: Convert to Grayscale ---
% Feature detection algorithms generally work on grayscale images.
img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);

% --- Step 3: Detect ORB Features ---
% Use the 'detectORBFeatures' function to find keypoints.
% Then use 'extractFeatures' to compute the descriptors.
orb_features1 = detectORBFeatures(img1_gray);
orb_features2 = detectORBFeatures(img2_gray);

[descriptors1, valid_points1] = extractFeatures(img1_gray, orb_features1);
[descriptors2, valid_points2] = extractFeatures(img2_gray, orb_features2);

% --- Step 4: Match Features ---
% The 'matchFeatures' function is the equivalent of the Brute-Force Matcher.
indexPairs = matchFeatures(descriptors1, descriptors2, 'MatchThreshold', 50, 'MaxRatio', 0.8);

% Retrieve the locations of the matched points.
matchedPoints1 = valid_points1(indexPairs(:, 1), :);
matchedPoints2 = valid_points2(indexPairs(:, 2), :);

% --- Step 5: Visualize Matches ---
% Use 'showMatchedFeatures' to display the images and the matched points.
figure('Name', 'ORB Feature Match', 'NumberTitle', 'off');
showMatchedFeatures(img1, img2, matchedPoints1, matchedPoints2, 'montage');
title('Matched Features (ORB)');

% To display only a subset of matches, you can take the first N matches:
% N = 20;
% figure;
% showMatchedFeatures(img1, img2, matchedPoints1(1:N), matchedPoints2(1:N), 'montage');
% title('Top 20 Matched Features (ORB)');