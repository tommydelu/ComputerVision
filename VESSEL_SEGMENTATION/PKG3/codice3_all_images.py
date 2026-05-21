from VESSEL_SEGMENTATION.PKG3.functions import *
import os

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET',"IR_no_artifacts")
DST_PATH = os.path.join(GLOBAL_PATH, 'PACKAGE_codice3', "segmentation_results_1")
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

        # FILTRAGGIO CON WAVELET
        wavelet_img = wavelet_filtering(img)

        clipLimit = 2
        gridSize = (16,16)

        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=gridSize)
        clahe_img = clahe.apply(wavelet_img)

        element_len = 21
        linear_structuring_elements = create_linear_structuring_elements(element_len=element_len, angle_step=15)
        Ic = opening(clahe_img, linear_structuring_elements)
        Ic_optimized = dilation_reconstruction(Ic,clahe_img)

        median_ksize = 61

        background = cv.medianBlur(src=clahe_img,ksize=median_ksize)
        Iv = cv.subtract(background, Ic_optimized)

        If = final_filtering(Iv,linear_structuring_elements)

        phigh = 98
        plow = 85
        t_low, t_high = get_percentile_thresholds(If,phigh,plow)
        result = thresholding(If,t_low,t_high)

        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)

        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result)


    row_name = os.path.basename(DST_PATH)
    row_content = {"Name": row_name,
                   "element len":element_len,
                   "Median filter size": median_ksize,
                   "Cliplimit":clipLimit,
                   "gridsize": gridSize,
                   "phigh":phigh,
                   "plow":plow}
    
    with open("PACKAGE_codice3/parameters_used.txt", "a", encoding="utf-8") as file:
        file.write(str(row_content) + '\n')