from functions import *
import os


GLOBAL_PATH = os.getcwd()
IMG_PATH1 = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_bigVessels_1')
IMG_PATH2 = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_thinVessels_1')
DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', "segmentation_results_1")

if not os.path.exists(DST_PATH):
    os.mkdir(DST_PATH)

if __name__ == "__main__":

    for fname in sorted(os.listdir(IMG_PATH1)):

        path1 = os.path.join(IMG_PATH1,fname)
        path2 = os.path.join(IMG_PATH2,fname)

        imgBig = cv.imread(path1,0)
        imgThin = cv.imread(path2,0)

        final_segmentation = cv.bitwise_or(imgBig, imgThin)

        save_path = os.path.join(DST_PATH, fname)
        cv.imwrite(save_path, final_segmentation)




