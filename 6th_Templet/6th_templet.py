import cv2
import matplotlib.pyplot as plt

# Load main image and template
img = cv2.imread('dhoni-virat.jpg', 0)   # larger image (grayscale)
template = cv2.imread('virat.jpg', 0)  # template (grayscale)

if img is None or template is None:
    print("Could not open or find the image/template.")
    exit()

# Template width and height
w, h = template.shape[::-1]

# Perform template matching (using correlation coefficient method)
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find location of best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Top-left and bottom-right coordinates of matched area
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw rectangle on match
result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 2)

# Plot results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(result_img)
plt.title("Template Matched")
plt.axis("off")

plt.tight_layout()
plt.show()
