"""
many images in the IR durectory have regions completely white, because the original image was translated/rotated.
Here we identify those regions and we make them as the same color of the background
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

GLOBAL_PATH = os.getcwd()
IR_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR')
DST_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR_no_artifacts')

if not os.path.exists(DST_PATH):
        os.mkdir(DST_PATH)

if __name__ == '__main__':

    for fname in sorted(os.listdir(IR_PATH)):
        
        IMG_PATH = os.path.join(IR_PATH,fname)
        img = cv.imread(IMG_PATH,0)

        ret, mask_artifacts = cv.threshold(img,253,255,cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8) 
        mask_dilated = cv.dilate(mask_artifacts, kernel, iterations=2)

        retina_pixels = img[mask_dilated == 0]
        background_gray = np.median(retina_pixels)

        img_cleaned = img.copy()
        img_cleaned[mask_dilated == 255] = background_gray
        DST_IMG_PATH = os.path.join(DST_PATH,fname)
        cv.imwrite(DST_IMG_PATH,img_cleaned)