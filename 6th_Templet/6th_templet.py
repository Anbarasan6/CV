import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and template in grayscale
img_path = r"D:\Sem_3_Lab\CV\6th_Templet\rohit.png"
template_path = r"D:\Sem_3_Lab\CV\6th_Templet\rohit1.png"

img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

th, tw = template_gray.shape

methods = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
}

plt.figure(figsize=(15, 12))  # Large figure for all subplots

for idx, (method_name, method) in enumerate(methods.items(), 1):
    result = cv2.matchTemplate(img_gray, template_gray, method)

    # For SQDIFF methods, minimum value is better; otherwise maximum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc

    # Draw rectangle on a copy of the image
    img_copy = img_gray.copy()
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    img_color = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img_color, top_left, bottom_right, (255, 0, 0), 2)

    # Plot in subplot
    plt.subplot(3, 2, idx)  # 3 rows, 2 columns
    plt.imshow(img_color, cmap='gray')
    plt.title(method_name)
    plt.axis('off')

plt.suptitle("Template Matching - All Methods", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
