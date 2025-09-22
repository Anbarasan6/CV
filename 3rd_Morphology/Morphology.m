% Load image
img1 =  imread("D:\Sem_3_Lab\CV\3rd_Morphology\tiger.png");
original_img = img1;
gray = rgb2gray(img1);
img = gray;
kernel = ones(5,5);

% Morphological operations
erosion = imerode(img, kernel);
dilation = imdilate(img, kernel);
opening = imopen(img, kernel);
closing = imclose(img, kernel);
gradient = imsubtract(imdilate(img, kernel), imerode(img, kernel));
tophat = imtophat(img, kernel);
blackhat = imbothat(img, kernel);

rect_kernel = strel('rectangle', [5,5]);
rectangle_op = imclose(img, rect_kernel);

ellipse_kernel = strel('disk', 5);  % Using disk as a replacement for ellipse
ellipse_op = imclose(img, ellipse_kernel);

cross_kernel = strel('line', 5, 45);
cross_op = imclose(img, cross_kernel);

% Plot results
figure('Position', [100, 100, 1200, 900]);
subplot(3,4,1), imshow(original_img), title('Original Image'), axis off;
subplot(3,4,2), imshow(gray), title('Gray Scale Image'), axis off;
subplot(3,4,3), imshow(erosion), title('Erosion Image'), axis off;
subplot(3,4,4), imshow(dilation), title('Dilation Image'), axis off;
subplot(3,4,5), imshow(opening), title('Open operation Image'), axis off;
subplot(3,4,6), imshow(closing), title('Close operation Image'), axis off;
subplot(3,4,7), imshow(gradient), title('Gradient operation Image'), axis off;
subplot(3,4,8), imshow(tophat), title('Top Hat operation Image'), axis off;
subplot(3,4,9), imshow(blackhat), title('Black Hat operation Image'), axis off;
subplot(3,4,10), imshow(rectangle_op), title('Rectangular operation Image'), axis off;
subplot(3,4,11), imshow(ellipse_op), title('Ellipse operation Image'), axis off;
subplot(3,4,12), imshow(cross_op), title('Cross operation Image'), axis off;