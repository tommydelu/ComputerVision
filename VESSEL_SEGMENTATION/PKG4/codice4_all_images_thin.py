from VESSEL_SEGMENTATION.PKG4.functions import *
import os
from skimage import filters
from skimage import morphology


GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET',"IR")
DST_PATH = os.path.join(GLOBAL_PATH, 'PACKAGE_codice4', "segmentation_thin_results_5")
LABEL_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')

if not os.path.exists(DST_PATH):
    os.mkdir(DST_PATH)

if __name__ == "__main__":

    for fname in os.listdir(SRC_PATH):

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
        kernel_mask = np.ones((15, 15), np.uint8) 
        forbidden_zone = cv.dilate(mask_artifacts, kernel_mask, iterations=3)

        wavelet_filtered_img = wavelet_filtering(img)

        medianKsize = 3
        median_blur = cv.medianBlur(wavelet_filtered_img,medianKsize)

        d = 9
        sigmaC = 10
        sigmaS = 10
        bilateral_filtered_img = cv.bilateralFilter(src=median_blur,d=d,sigmaColor=sigmaC,sigmaSpace=sigmaS) 

        clipLimit = 2
        gridSize = 32
        clahe_img = clahe(bilateral_filtered_img,clipLimit=clipLimit,gridSize=gridSize)
        
        tophatKsize = (15,15)    
        kernel_thin = cv.getStructuringElement(cv.MORPH_ELLIPSE,tophatKsize)
        tophat_thin = cv.morphologyEx(clahe_img, cv.MORPH_BLACKHAT, kernel_thin)

        element_len = 11

        linear_structuring_elements = create_linear_structuring_elements(element_len=element_len, angle_step=15)
        sigma = 5
        klen = 33

        r1,kernel1 = create_1D_gaussian_ker(sigma=sigma, 
                                        kernel_len=klen)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma=sigma, 
                                                    kernel_len=klen, 
                                                    scaling=10)

        resulting_kernel = compose_kernels(r1,kernel1,kernel2)

        Idiff = apply_filters(tophat_thin,resulting_kernel)

        If = final_filtering(Idiff,linear_structuring_elements)

        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        If_closed = cv.morphologyEx(If, cv.MORPH_OPEN, kernel_close)

        If_norm = cv.normalize(If_closed,None,0,255,cv.NORM_MINMAX).astype(np.uint8) # bring the intensity range in 0-255
        phigh = 90
        plow = 80
        t_low, t_high = get_percentile_thresholds(If_norm,phigh,plow)
        hyst = filters.apply_hysteresis_threshold(If_norm,t_low,t_high)

        clean_hyst = morphology.remove_small_objects(hyst.astype(bool), min_size=1000)
        result = (clean_hyst.astype(int) * 255)

        result[forbidden_zone == 255] = 0


        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)
        
        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result.astype(np.uint8))

    row_name = os.path.basename(DST_PATH)
    row_content = {"Name": row_name,
                   "median ksize": medianKsize,
                   "tophat ksize":tophatKsize,
                   "d":d,
                   "SigmaC":sigmaC,
                   "SigmaS":sigmaS,
                   "element len":element_len,
                   "Cliplimit":clipLimit,
                   "sigma":sigma,
                   "klen":klen,
                   "gridsize": gridSize,
                   "phigh":phigh,
                   "plow":plow}

    with open("PACKAGE_codice4/parameters_used_thin.txt", "a", encoding="utf-8") as file:
        file.write(str(row_content) + '\n')