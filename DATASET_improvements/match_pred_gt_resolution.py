"""
All IR images have the resolution 1908 x 1908
Here you take one path as input (results of a segmentation pipeline) e you transform each image in the same resolution as the resolution
of the associated label
"""

import os
import cv2 as cv

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice1','segmentation_results_3')
DST_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice1','segmentation_results_3_resized')
LABELS_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')

if not os.path.exists(DST_PATH):
        os.mkdir(DST_PATH)

if __name__ == '__main__':

    
    for fname in sorted(os.listdir(SRC_PATH)):
        

        if 'PRE' in fname:
            subject = fname.split('PRE')[0]
            phase = 'PRE'
            target = 'total_1.png'

        elif 'POST' in fname:
            subject = fname.split('POST')[0]
            phase = 'POST'
            target = 'total_2.png'

        IMG_PATH = os.path.join(SRC_PATH,fname)
        LABEL_PATH = os.path.join(LABELS_PATH,subject,target)
        if not os.path.exists(LABEL_PATH):
            continue

        img = cv.imread(IMG_PATH,0)
        label = cv.imread(LABEL_PATH,0)

        if img.shape[:2] != label.shape[:2]:
            img = cv.resize(img, (label.shape[1],label.shape[0]), interpolation=cv.INTER_NEAREST)
        
        OUTPUT_PATH = os.path.join(DST_PATH,fname)
        cv.imwrite(OUTPUT_PATH, img)

        




    