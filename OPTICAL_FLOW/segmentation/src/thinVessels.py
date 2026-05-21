import os
import cv2 as cv
from skimage import filters
from skimage import morphology
import numpy as np
from OPTICAL_FLOW.utils.functions import wavelet_filtering, clahe, create_linear_structuring_elements, create_1D_gaussian_ker, create_1D_laplacian_gaussian_ker, compose_kernels, apply_filters, final_filtering, get_percentile_thresholds
from OPTICAL_FLOW.utils.classes import Subject



GLOBAL_PATH = os.getcwd()
SRC_PATH    = os.path.join(GLOBAL_PATH,'DATA','DATASET1','raw','IR')
DST_PATH    = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation','results','figures',"thinVessels_1")
LABELS_PATH = os.path.join(GLOBAL_PATH,'DATA','DATASET1','processed','results','figures','labels_reversed')


os.makedirs(DST_PATH, exist_ok=True)




if __name__ == '__main__':

    for fname in sorted(os.listdir(SRC_PATH)):

        if not os.path.exists(SRC_PATH):
            continue

        subject = Subject(fname,SRC_PATH,LABELS_PATH)


        wavelet_filtered_img = wavelet_filtering(subject.img)

        median_blur = cv.medianBlur(wavelet_filtered_img,3)

        bilateral_filtered_img = cv.bilateralFilter(src=median_blur,d=9,sigmaColor=10,sigmaSpace=10) 

        clahe_img = clahe(bilateral_filtered_img,clipLimit=2,gridSize=32)
        
        kernel_thin = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
        tophat_thin = cv.morphologyEx(clahe_img, cv.MORPH_BLACKHAT, kernel_thin)

        linear_structuring_elements = create_linear_structuring_elements(element_len=11, angle_step=15)
        r1,kernel1 = create_1D_gaussian_ker(sigma=5, 
                                        kernel_len=33)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma=5, 
                                                    kernel_len=33, 
                                                    scaling=10)

        resulting_kernel = compose_kernels(r1,kernel1,kernel2)

        Idiff = apply_filters(tophat_thin,resulting_kernel)

        If = final_filtering(Idiff,linear_structuring_elements)
        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        If_closed = cv.morphologyEx(If, cv.MORPH_OPEN, kernel_close)
        If_norm = cv.normalize(If_closed,None,0,255,cv.NORM_MINMAX).astype(np.uint8) # bring the intensity range in 0-255

        t_low, t_high = get_percentile_thresholds(If_norm,90,80)
        hyst = filters.apply_hysteresis_threshold(If_norm,t_low,t_high)
        clean_hyst = morphology.remove_small_objects(hyst.astype(bool), min_size=1000)
        result = (clean_hyst.astype(int) * 255)

        result_clean = subject.correct_artifacts(result)
        
        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result_clean.astype(np.uint8))
