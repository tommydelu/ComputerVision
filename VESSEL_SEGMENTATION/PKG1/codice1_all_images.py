from VESSEL_SEGMENTATION.PKG1.functions import *
import os

file_path = os.path.abspath(__file__)
GLOBAL_PATH = os.path.dirname(os.path.dirname(file_path))
SRC_PATH = os.path.join(GLOBAL_PATH, "DATASET",'IR_no_artifacts')
LABEL_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')
DST_PATH = os.path.join(GLOBAL_PATH, "PACKAGE_codice1", "segmentation_results_3")

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

        median_filter_size = 15
        clipLimit = 2
        grid_size = (16,16)
        amax = 255
        amin = 0

        clahe = pre_processing_attempt1(original_img, median_filter_size, clipLimit, grid_size, amin, amax)
        clahe = cv.bitwise_not(clahe)

        element_len = 21
    
        linear_structuring_elements = create_linear_structuring_elements(element_len=element_len,angle_step=15)
        Ic = opening(clahe, linear_structuring_elements)
        Ic_optimized = dilation_reconstruction(Ic,clahe)
        Ib = extract_background(clahe,linear_structuring_elements)
        Iv = cv.subtract(Ic_optimized,Ib)
        
        sigma = 5
        ksize = 51

        r1,kernel1 = create_1D_gaussian_ker(sigma, ksize)
        r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma,ksize,scaling=10)
        final_kernels = compose_kernels(r1,kernel1,kernel2)
        max_response = apply_filters(Iv,final_kernels)

        phigh = 97
        plow = 85

        t_low, t_high = get_percentile_thresholds(max_response,p_high=phigh,p_low=plow)
        If = final_filtering(max_response,linear_structuring_elements)
        result = thresholding(If,t_low,t_high)

        if result.shape[:2] != label.shape[:2]:
            result = cv.resize(result,(label.shape[1],label.shape[0]),interpolation=cv.INTER_NEAREST)
        
        
        filename = os.path.join(DST_PATH,fname)
        cv.imwrite(filename,result)

    
    row_name = os.path.basename(DST_PATH)
    row_content = {"Name": row_name,
                    "Median filter size": median_filter_size,
                   "Cliplimit":clipLimit,
                   "gridsize": grid_size,
                   "amax": amax,
                   "amin":amin,
                    "sigma":sigma,
                    "ksize":ksize,
                    "phigh":phigh,
                    "plow":plow}
    

    # with open("PACKAGE_codice1/parameters_used.txt", "a", encoding="utf-8") as file:
    #     file.write(str(row_content) + '\n')