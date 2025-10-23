import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images in grayscale
img = cv2.imread(r"D:\Sem_3_Lab\CV\6th_Templet\rohit.png",0)
template= cv2.imread(r"D:\Sem_3_Lab\CV\6th_Templet\rohit1.png",0)
if img is None or template is None:
    raise IOError("Error: Could not read one or both images. Check file paths.")

# Resize template if larger than image
if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
    scale = min(img.shape[0]/template.shape[0], img.shape[1]/template.shape[1]) * 0.9
    template = cv2.resize(template, (int(template.shape[1]*scale), int(template.shape[0]*scale)))

h, w = template.shape

# Define methods
methods = ['normxcorr2','ssd','sad','correlation','normalized_correlation','normalized_difference']

plt.figure(figsize=(15, 18))

for i, method in enumerate(methods, 1):
    if method == 'normxcorr2':
        # Normalized cross-correlation using cv2.matchTemplate
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        score = max_val
    else:
        # Custom methods
        result = np.zeros((img.shape[0]-h+1, img.shape[1]-w+1))
        T = template.astype(np.float64)
        Tn = (T - T.mean()) / (T.std() + 1e-5)
        
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                P = img[y:y+h, x:x+w].astype(np.float64)
                if method == 'ssd':
                    result[y,x] = np.sum((P-T)**2)
                elif method == 'sad':
                    result[y,x] = np.sum(np.abs(P-T))
                elif method == 'correlation':
                    result[y,x] = np.sum(P*T)
                elif method == 'normalized_correlation':
                    Pn = (P - P.mean()) / (P.std() + 1e-5)
                    result[y,x] = np.sum(Pn*Tn)
                elif method == 'normalized_difference':
                    result[y,x] = np.sum((P-T)*2) / np.sum(T*2)
        
        if method in ['ssd','sad','normalized_difference']:
            score = result.min()
            top_left = np.unravel_index(np.argmin(result), result.shape)[::-1]
        else:
            score = result.max()
            top_left = np.unravel_index(np.argmax(result), result.shape)[::-1]

    # Plot heatmap
    plt.subplot(6, 3, (i-1)*3 + 1)
    plt.imshow(result, cmap='gray')
    plt.title(f'{method} result')
    plt.axis('off')

    # Plot score text
    plt.subplot(6, 3, (i-1)*3 + 2)
    plt.text(0.5, 0.5, f'{method}\nScore: {score:.4f}', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    # Plot detected rectangle on image
    plt.subplot(6, 3, (i-1)*3 + 3)
    plt.imshow(img, cmap='gray')
    rect = plt.Rectangle(top_left, w, h, edgecolor='r', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title('Detected Template')
    plt.axis('off')

plt.tight_layout()
plt.show()
