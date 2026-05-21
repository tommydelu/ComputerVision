import cv2 as cv
import numpy as np
from skimage import filters
from skimage import morphology
from OPTICAL_FLOW.utils.functions import wavelet_filtering, clahe, create_linear_structuring_elements, create_1D_gaussian_ker, create_1D_laplacian_gaussian_ker, compose_kernels, apply_filters, final_filtering, get_percentile_thresholds
import os
from OPTICAL_FLOW.utils.classes import Subject

GLOBAL_PATH = os.getcwd()
SRC_PATH    = os.path.join(GLOBAL_PATH,'DATA','DATASET1','raw','IR')
DST_PATH    = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation','results','figures',"bigVessels_1")
LABELS_PATH = os.path.join(GLOBAL_PATH,'DATA','DATASET1','processed','results','figures','labels_reversed')

os.makedirs(DST_PATH, exist_ok=True)


if __name__ == '__main__':


    for fname in sorted(os.listdir(SRC_PATH)):

        if not os.path.exists(SRC_PATH):
            continue

        subject = Subject(fname,SRC_PATH,LABELS_PATH)


        wavelet_filtered_img = wavelet_filtering(subject.img)

        median_blur = cv.medianBlur(wavelet_filtered_img, 5)

        bilateral_filtered_img = cv.bilateralFilter(src=median_blur, d=31, sigmaColor=35, sigmaSpace=35) 

        clahe_img = clahe(bilateral_filtered_img,clipLimit=1,gridSize=4)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
        tophat = cv.morphologyEx(clahe_img,cv.MORPH_BLACKHAT,kernel)

        linear_structuring_elements = create_linear_structuring_elements(element_len=21, angle_step=15)

        r1,kernel1 = create_1D_gaussian_ker(sigma=5, 
                                        kernel_len=39)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma=5, 
                                                    kernel_len=39,
                                                    scaling=10)     

        resulting_kernel = compose_kernels(r1,kernel1,kernel2)
        Idiff = apply_filters(tophat,resulting_kernel)

        If = final_filtering(Idiff,linear_structuring_elements)
        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        If_closed = cv.morphologyEx(If, cv.MORPH_CLOSE, kernel_close)
        If_norm = cv.normalize(If_closed,None,0,255,cv.NORM_MINMAX).astype(np.uint8) # bring the intensity range in 0-255

        t_low, t_high = get_percentile_thresholds(If_norm,85,70)
        hyst = filters.apply_hysteresis_threshold(If_norm,t_low,t_high)
        clean_hyst = morphology.remove_small_objects(hyst.astype(bool), min_size=3000)
        result = (clean_hyst.astype(int) * 255)

        result_clean = subject.correct_artifacts(result)

        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result_clean.astype(np.uint8))

        











