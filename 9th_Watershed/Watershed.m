clc; clear; close all;
% --------- Load image ----------
img = imread("D:\Sem_3_Lab\CV\9th_Watershed\fruits.png");
img = im2double(img);
[h, w, ~] = size(img);
% --------- Gradient for watershed ----------
I = rgb2gray(img);
I = imgaussfilt(I, 1.0);
G = imgradient(I);
% --------- Ask user how many regions ----------
numRegions = input('Enter the number of regions to segment: ');
% --------- Interactive marker drawing ----------
figure('Name','Draw markers (double-click to finish each)');
imshow(img);
title('Draw ROI markers (double-click each to finish)');
hold on;
markerMask = zeros(h, w, 'uint16');
markerColors = containers.Map('KeyType','double','ValueType','any');
for markerID = 1:numRegions
    % assign random bright color
    c = 0.2 + 0.8*rand(1,3);
    markerColors(markerID) = c;
    % let user draw ROI
    disp(['Draw marker: ' num2str(markerID)]);
    hRoi = drawfreehand('Color', c, 'FaceAlpha', 0.15, 'LineWidth', 1.5);
    mask = createMask(hRoi);
    markerMask(mask) = markerID;
    % print marker ID on image
    pos = mean(hRoi.Position,1);
    text(pos(1), pos(2), ['Marker: ' num2str(markerID)], ...
        'Color','w','FontSize',12,'FontWeight','bold','BackgroundColor','k');
end
% --------- Show markers ----------
allIDs = 1:numRegions;
colorArray = zeros(numRegions, 3);
for k = allIDs
    if isKey(markerColors, k)
        colorArray(k, :) = markerColors(k);
    else
        colorArray(k, :) = [0 0 0]; % unused ID is black
    end
end
figure, imshow(label2rgb(markerMask, colorArray, 'w')), title('User Markers');
% --------- Watershed segmentation ----------
G_forced = imimposemin(G, markerMask > 0);
Lw = watershed(G_forced);
assignedMap = zeros(h, w, 'uint16');
basins = unique(Lw); basins(basins == 0) = [];
for k = 1:numel(basins)
    lab = basins(k);
    pix = (Lw == lab);
    mvals = markerMask(pix);
    mvals = mvals(mvals > 0);
    if ~isempty(mvals)
        assignedMap(pix) = mode(mvals);
    end
end
% Fill unassigned basins by nearest marker
unassigned = (assignedMap == 0);
if any(unassigned(:))
    [~, idx] = bwdist(markerMask > 0);
    nearestMarkerId = markerMask(idx);
    assignedMap(unassigned) = nearestMarkerId(unassigned);
end
% --------- Build segmentation ----------
segRGB = zeros(h, w, 3);
for id = 1:numRegions
    if isKey(markerColors, id)
        c = markerColors(id);
        m = (assignedMap == id);
        segRGB(:,:,1) = segRGB(:,:,1) + m * c(1);
        segRGB(:,:,2) = segRGB(:,:,2) + m * c(2);
        segRGB(:,:,3) = segRGB(:,:,3) + m * c(3);
    end
end
% --------- Overlay ----------
alpha = 0.4;
overlay = (1 - alpha) * img + alpha * segRGB;
boundaries = (Lw == 0);
overlay(repmat(boundaries, [1 1 3])) = 1;
figure, imshow(overlay), title('Watershed Segmentation with Markers');