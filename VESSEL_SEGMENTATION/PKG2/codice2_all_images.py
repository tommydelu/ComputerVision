from VESSEL_SEGMENTATION.PKG2.functions import *
import os

file_path = os.path.abspath(__file__)
GLOBAL_PATH = os.path.dirname(os.path.dirname(file_path))
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR_no_artifacts')
LABEL_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')
DST_PATH = os.path.join(GLOBAL_PATH, 'PACKAGE_codice2', "segmentation_results_1")

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
        original_img = cv.imread(IMG_PATH,0)

        label_path = os.path.join(LABEL_PATH,subject,target)
        if not os.path.exists(label_path):
            continue

        label = cv.imread(label_path,0)

        d = 9
        sigmaColor = 50
        sigmaSpace = 75

        blurring = cv.bilateralFilter(original_img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

        median_filter_size = 11
        clipLimit = 2
        grid_size = (16,16)
        amax = 255
        amin = 0

        clahe = pre_processing_attempt1(blurring, median_filter_size, clip_limit=clipLimit, grid_size=grid_size, a_min=amin, a_max=amax)
        clahe = cv.bitwise_not(clahe)

        sigma = 4
        ksize = 31

        r1,kernel1 = create_1D_gaussian_ker(sigma, ksize)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma, ksize,scaling=10)
        final_kernels = compose_kernels(r1,kernel1,kernel2)
        Idiff = apply_filters(clahe,final_kernels)

        element_len = 31
        linear_structuring_elements = create_linear_structuring_elements(element_len=element_len, angle_step=15)
        Ic = opening(Idiff, linear_structuring_elements)
        Ic_optimized = dilation_reconstruction(Ic,Idiff)
        Ib = extract_background(Idiff,linear_structuring_elements)
        Iv = cv.subtract(Ic_optimized,Ib)

        If = final_filtering(Iv,linear_structuring_elements)

        Ic1 = opening(If, linear_structuring_elements)
        Ic_optimized1 = dilation_reconstruction(Ic1,If)

        # Adaptive thresholding --> find a threshold based on a local region of pixels
        maxValue = 255
        blockSize = 5
        offsetC = 3

        result = cv.adaptiveThreshold(Ic_optimized1,maxValue,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,blockSize,offsetC)

        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)

        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result)


    row_name = os.path.basename(DST_PATH)
    row_content = {"Name": row_name,
                   "d": d,
                   "sigmaColor":sigmaColor,
                   "sigmaSpace":sigmaSpace,
                   "Median filter size": median_filter_size,
                   "Cliplimit":clipLimit,
                   "gridsize": grid_size,
                   "amax": amax,
                   "amin":amin,
                    "sigma":sigma,
                    "ksize":ksize,
                    "maxValue":maxValue,
                    "blockSize":blockSize,
                    "offSetC":offsetC}
    
    with open("PACKAGE_codice2/parameters_used.txt", "a", encoding="utf-8") as file:
        file.write(str(row_content) + '\n')

    

    
    
    
