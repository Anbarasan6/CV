
"""
Keyboard shortcuts:
    ESC   - exit
    SPACE - generate new distribution
"""

import numpy as np
import cv2 as cv

def make_gaussians(cluster_n, img_size):
    points = []
    for _ in range(cluster_n):
        mean = (0.1 + 0.8 * np.random.rand(2)) * img_size
        a = (np.random.rand(2, 2) - 0.5) * img_size * 0.1
        cov = np.dot(a.T, a) + img_size * 0.05 * np.eye(2)
        n = 100 + np.random.randint(900)
        pts = np.random.multivariate_normal(mean, cov, n)
        points.append(pts)
    points = np.float32(np.vstack(points))
    return points

def main():
    cluster_n = 5
    img_size = 512

    # Generating bright palette
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0,:] = 255
    colors[0,:,0] = np.arange(0, 180, 180.0 / cluster_n)
    colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]

    print(__doc__)

    while True:
        print('sampling distributions...')
        points = make_gaussians(cluster_n, img_size)

        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
        _ret, labels, _centers = cv.kmeans(points, cluster_n, None, term_crit, 10, cv.KMEANS_RANDOM_CENTERS)

        img = np.zeros((img_size, img_size, 3), np.uint8)
        for (x, y), label in zip(np.int32(points), labels.ravel()):
            c = tuple(map(int, colors[label]))
            cv.circle(img, (x, y), 1, c, -1)

        cv.imshow('kmeans', img)
        ch = cv.waitKey(0)
        if ch == 27:   # ESC
            break

    print('Done')
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
