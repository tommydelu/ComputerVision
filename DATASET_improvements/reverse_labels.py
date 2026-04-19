"""
Since i performed the segmentation having bright vessels on a dark background this code is used to reverse the labels
"""

import os
import cv2 as cv

GLOBAL_PATH = os.getcwd()
LABELS_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')
DST_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels_reversed')

if not os.path.exists('labels_reversed'):
        os.mkdir('labels_reversed')

if __name__ == '__main__':

    for dir in sorted(os.listdir(LABELS_PATH)):

        SUBJECT_PATH = os.path.join(LABELS_PATH,dir)

        output_path1 = os.path.join(DST_PATH,dir)
        output_path2 = os.path.join(DST_PATH,dir)

        if not os.path.exists(output_path1):
            os.mkdir(output_path1)
        elif not os.path.exists(output_path2):
            os.mkdir(output_path2)

        img1_path = os.path.join(SUBJECT_PATH,'total_1.png')
        img2_path = os.path.join(SUBJECT_PATH,'total_2.png')

        img1 = cv.imread(img1_path,0)
        img2 = cv.imread(img2_path,0)

        img1 = 255 - img1
        img2 = 255 - img2

        new_img1_path = os.path.join(output_path1,'total_1.png')
        new_img2_path = os.path.join(output_path1,'total_2.png')

        cv.imwrite(new_img1_path,img1)
        cv.imwrite(new_img2_path,img2)






















    



