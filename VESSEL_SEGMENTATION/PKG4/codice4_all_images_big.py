from VESSEL_SEGMENTATION.PKG4.functions import *
import os
from skimage import filters
from skimage import morphology


# !!!! in segmentation_big_results_1 stavo passando al bilateral la wavelet_img e non la media_blur, il median blur non è stato applicato
# !!!! nei risultati 2 dei big vessels stavo passando al kernel laplaciano invece che sigma = sigma sigma = klen quindi ho i vasi giganti di spessore 

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET',"IR")
DST_PATH = os.path.join(GLOBAL_PATH, 'PACKAGE_codice4', "segmentation_big_results_5")
LABEL_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')

if not os.path.exists(DST_PATH):
    os.mkdir(DST_PATH)

if __name__ == "__main__":

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
        kernel_mask = np.ones((15, 15), np.uint8) 
        forbidden_zone = cv.dilate(mask_artifacts, kernel_mask, iterations=3)


        wavelet_filtered_img = wavelet_filtering(img)

        medianKsize = 5
        median_blur = cv.medianBlur(wavelet_filtered_img,medianKsize)

        d = 31
        sigmaC = 35
        sigmaS = 35
        bilateral_filtered_img = cv.bilateralFilter(src=median_blur,d=d,sigmaColor=sigmaC,sigmaSpace=sigmaS) 

        clipLimit = 1
        gridSize = 4
        clahe_img = clahe(bilateral_filtered_img,clipLimit=clipLimit,gridSize=gridSize)

        tophatKsize = (30,30)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,tophatKsize)
        tophat = cv.morphologyEx(clahe_img,cv.MORPH_BLACKHAT,kernel)

        element_len = 21
        linear_structuring_elements = create_linear_structuring_elements(element_len=element_len, angle_step=15)
       
        sigma = 5
        klen=39
        r1,kernel1 = create_1D_gaussian_ker(sigma=sigma, 
                                        kernel_len=klen)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma=sigma, 
                                                    kernel_len=klen,
                                                    scaling=10)

        resulting_kernel = compose_kernels(r1,kernel1,kernel2)

        Idiff = apply_filters(tophat,resulting_kernel)

        If = final_filtering(Idiff,linear_structuring_elements)

        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        If_closed = cv.morphologyEx(If, cv.MORPH_CLOSE, kernel_close)
        
        If_norm = cv.normalize(If_closed,None,0,255,cv.NORM_MINMAX).astype(np.uint8) # bring the intensity range in 0-255

        phigh = 85
        plow = 70
        t_low, t_high = get_percentile_thresholds(If_norm,phigh,plow)
        hyst = filters.apply_hysteresis_threshold(If_norm,t_low,t_high)

        clean_hyst = morphology.remove_small_objects(hyst.astype(bool), min_size=3000)
        result = (clean_hyst.astype(int) * 255)

        result[forbidden_zone == 255] = 0

        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)

        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result.astype(np.uint8))


    row_name = os.path.basename(DST_PATH)
    row_content = {"Name": row_name,
                   "median ksize": medianKsize,
                   "d": d,
                   "sigmaC":sigmaC,
                   "sigmaS":sigmaS,
                   "tophat ksize":tophatKsize,
                   "element len":element_len,
                   "sigma":sigma,
                   "klen":klen,
                   "Cliplimit":clipLimit,
                   "gridsize": gridSize,
                   "phigh":phigh,
                   "plow":plow}

    with open("PACKAGE_codice4/parameters_used_big.txt", "a", encoding="utf-8") as file:
        file.write(str(row_content) + '\n')