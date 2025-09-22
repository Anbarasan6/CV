#!/usr/bin/env python

'''
Watershed segmentation
=========

This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().

Usage
-----
watershed.py [image filename]

Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# -------------------------------
# BEGIN: Sketcher class (from common.py)
# -------------------------------
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self.on_mouse)
        self.show()

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            cv.line(self.dests[0], self.prev_pt, pt, self.colors_func()[0], 5)
            cv.line(self.dests[1], self.prev_pt, pt, self.colors_func()[1], 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()
# -------------------------------
# END: Sketcher class
# -------------------------------

class App:
    def __init__(self, fn):
        self.img = cv.imread(fn)
        if self.img is None:
            raise Exception('Failed to load image file: %s' % fn)

        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32(list(np.ndindex(2, 2, 2))) * 255

        self.auto_update = True
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)

    def get_colors(self):
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv.watershed(self.img, m)
        overlay = self.colors[np.maximum(m, 0)]
        vis = cv.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
        cv.imshow('watershed', vis)

    def run(self):
        while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
            ch = cv.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update is', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
        cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'
    App(cv.samples.findFile(fn)).run()
