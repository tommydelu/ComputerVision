import os
import cv2 as cv
from functions import *
from skimage import filters
from skimage import morphology


GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','aligned_images_optic_nerve')
DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', "segmentation_bigVessels_1")
LABEL_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','aligned_labels')

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

        IMG_PATH = os.path.join(SRC_PATH, fname)
        img = cv.imread(IMG_PATH,0)

        label_path = os.path.join(LABEL_PATH,subject,target)
        if not os.path.exists(label_path):
            continue

        label = cv.imread(label_path,0)

        _, mask_artifacts = cv.threshold(img, 253, 255, cv.THRESH_BINARY)
        _, mask_black_zones = cv.threshold(img,2,255,cv.THRESH_BINARY_INV) # Aggiunta a segmentation_results_2

        kernel_mask = np.ones((15, 15), np.uint8) 
        forbidden_zone = cv.dilate(mask_artifacts, kernel_mask, iterations=3)

        wavelet_filtered_img = wavelet_filtering(img)

        median_blur = cv.medianBlur(wavelet_filtered_img, 5)

        bilateral_filtered_img = cv.bilateralFilter(src=median_blur,d=31, sigmaColor=35, sigmaSpace=35) 

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

        result[forbidden_zone == 255] = 0
        result[mask_black_zones == 255] = 0 # Aggiunta a segmentation_results_2

        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)

        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result.astype(np.uint8))
