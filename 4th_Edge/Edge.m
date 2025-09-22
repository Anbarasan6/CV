% Read Image
img = imread("zebra.png"); % You can change this to any image file
grayImg = rgb2gray(img); % Convert to grayscale if input is RGB

% Show Original Image
figure, imshow(grayImg), title('Original Image');

% Sobel Edge Detection
sobelEdges = edge(grayImg, 'sobel');
figure, imshow(sobelEdges), title('Sobel Edge Detection');

% Prewitt Edge Detection
prewittEdges = edge(grayImg, 'prewitt');       
figure, imshow(prewittEdges), title('Prewitt Edge Detection');

% Roberts Edge Detection
robertsEdges = edge(grayImg, 'roberts');
figure, imshow(robertsEdges), title('Roberts Edge Detection');

% Canny Edge Detection
cannyEdges = edge(grayImg, 'canny');
figure, imshow(cannyEdges), title('Canny Edge Detection');

% Laplacian of Gaussian (LoG) Edge Detection
logEdges = edge(grayImg, 'log');
figure, imshow(logEdges), title('LoG Edge Detection');
