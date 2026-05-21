from VESSEL_SEGMENTATION.PKG4.functions import *
import os
from skimage import filters
from skimage import morphology

GLOBAL_PATH = os.getcwd()
IMG_PATH = os.path.join(GLOBAL_PATH,'DATASET',"IR","L_07POST.jpg")
LABEL_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels_reversed',"L_07","total_2.png")

if __name__ == "__main__":

    img = cv.imread(IMG_PATH,0)
    label = cv.imread(LABEL_PATH,0) # label may be flipped with respect to img, i take into account this problem when i compute the metrics

    _, mask_artifacts = cv.threshold(img, 253, 255, cv.THRESH_BINARY)
    kernel_mask = np.ones((15, 15), np.uint8) 
    forbidden_zone = cv.dilate(mask_artifacts, kernel_mask, iterations=3)

    ################### PLOT 1 ###################
    plt.figure(figsize=(15,15))
    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.title("ORIGINAL IMAGE")
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(label,cmap='gray')
    plt.title("GROUND TRUTH IMAGE")
    plt.axis('off')

    # Let's start by filtering the image with the wavelet: we erase the high frequency noise
    wavelet_filtered_img = wavelet_filtering(img)
    noise_removed = cv.subtract(wavelet_filtered_img,img)
    plt.subplot(243)
    plt.imshow(noise_removed,cmap='gray')
    plt.title("Noise removed thanks to wavelet filtering")
    plt.axis('off')

    median_blur = cv.medianBlur(wavelet_filtered_img,3)

    bilateral_filtered_img = cv.bilateralFilter(src=median_blur,d=9,sigmaColor=10,sigmaSpace=10) 

    plt.subplot(244)
    plt.imshow(bilateral_filtered_img,cmap='gray')
    plt.title("Median Blurring")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # ################### PLOT 2 ###################

    plt.figure(figsize=(15,15))
    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.title("ORIGINAL IMAGE")
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(label,cmap='gray')
    plt.title("GROUND TRUTH IMAGE")
    plt.axis('off')

    clahe_img = clahe(bilateral_filtered_img,clipLimit=2,gridSize=32)
    plt.subplot(243)
    plt.imshow(clahe_img,cmap='gray')
    plt.title("Contrast enhancement with CLAHE")
    plt.axis('off')

    kernel_thin = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))
    tophat_thin = cv.morphologyEx(clahe_img, cv.MORPH_BLACKHAT, kernel_thin)
        
    plt.subplot(244)
    plt.imshow(tophat_thin,cmap='gray')
    plt.title("Tophat using small parameters")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # # #################### PLOT 3 ###################

    plt.figure(figsize=(15,15))
    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.title("ORIGINAL IMAGE")
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(label,cmap='gray')
    plt.title("GROUND TRUTH IMAGE")
    plt.axis('off')

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

    plt.subplot(243)
    plt.imshow(If_closed,cmap='gray')
    plt.title("After morpholohy")
    plt.axis('off')

    If_norm = cv.normalize(If_closed,None,0,255,cv.NORM_MINMAX).astype(np.uint8) # bring the intensity range in 0-255
    t_low, t_high = get_percentile_thresholds(If_norm,90,80)
    hyst = filters.apply_hysteresis_threshold(If_norm,t_low,t_high)

    clean_hyst = morphology.remove_small_objects(hyst.astype(bool), min_size=1000)
    result = (clean_hyst.astype(int) * 255)

    result[forbidden_zone == 255] = 0

    plt.subplot(244)
    plt.imshow(result,cmap='gray')
    plt.title("Hystheresis thresholding")
    plt.axis('off')

    plt.tight_layout()
    plt.show()






