import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img1 = cv.imread('tiger.jpg')
original_img = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)


# Morphological operations
erosion = cv.erode(img,kernel,iterations = 1)
dilation = cv.dilate(img,kernel,iterations = 1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
rectangle = cv.morphologyEx(img, cv.MORPH_RECT, kernel)
ellipse = cv.morphologyEx(img, cv.MORPH_ELLIPSE, kernel)
cross = cv.morphologyEx(img, cv.MORPH_CROSS, kernel)

# Plot results
plt.figure(figsize=(12, 9))
plt.subplot(3,4,1),plt.imshow(original_img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,2),plt.imshow(gray, cmap = 'gray')
plt.title('Gray Scale Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,3),plt.imshow(erosion, cmap = 'gray')
plt.title('erosion Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,4),plt.imshow(dilation, cmap = 'gray')
plt.title('dilation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,5),plt.imshow(opening, cmap = 'gray')
plt.title('Open operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,6),plt.imshow(closing, cmap = 'gray')
plt.title('Close operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,7),plt.imshow(gradient, cmap = 'gray')
plt.title('Gradient operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,8),plt.imshow(tophat, cmap = 'gray')
plt.title('Top Hat operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,9),plt.imshow(blackhat, cmap = 'gray')
plt.title('Black Hat operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,10),plt.imshow(rectangle, cmap = 'gray')
plt.title('Rectangular operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,11),plt.imshow(ellipse, cmap = 'gray')
plt.title('Ellipse operation Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,4,12),plt.imshow(cross, cmap = 'gray')
plt.title('Cross operation Image'), plt.xticks([]), plt.yticks([])

plt.show()