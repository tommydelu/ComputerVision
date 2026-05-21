from VESSEL_SEGMENTATION.PKG2.functions import *
import matplotlib.pyplot as plt
import os


def plot_img(img, title = "", cmap=None):

    plt.imshow(img,cmap=cmap)
    plt.axis("off")
    plt.title(title)


file_path = os.path.abspath(__file__)
GLOBAL_PATH = os.path.dirname(os.path.dirname(file_path))
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR_no_artifacts')
IMG_PATH = os.path.join(SRC_PATH,'S_27PRE.jpg')

if __name__ == "__main__":

    img = cv.imread(IMG_PATH,0)

    plt.figure(figsize=(15,15))
    plt.subplot(2,4,1)
    plot_img(img, "Original Image",cmap='gray')

    # Let's try to add a low-pass filter to the original image
    blurring = cv.bilateralFilter(img,9,50,50)
    plt.subplot(2,4,2)
    plot_img(blurring,"Bilateral filter",cmap='gray')

    # Increase the contrast
    clahe = pre_processing_attempt1(blurring, 9, 2, (2,2), 0, 255)
    clahe = cv.bitwise_not(clahe)
    plt.subplot(2,4,3)
    plot_img(clahe,"CLAHE", "gray")


    # Let's try to switch the gaussian convolution with the morphology as a suggestion of the professor

    r1,kernel1 = create_1D_gaussian_ker(4,31)
    r2,kernel2 = create_1D_laplacian_gaussian_ker(4,31,scaling=10)
    final_kernels = compose_kernels(r1,kernel1,kernel2)
    Idiff = apply_filters(clahe,final_kernels)
    plt.subplot(2,4,4)
    plot_img(Idiff,"After convolution with the two orthogonal kernels", cmap="gray")



    linear_structuring_elements = create_linear_structuring_elements(element_len=21, angle_step=15)
    Ic = opening(Idiff, linear_structuring_elements)
    Ic_optimized = dilation_reconstruction(Ic,Idiff)
    Ib = extract_background(Idiff,linear_structuring_elements)
    Iv = cv.subtract(Ic_optimized,Ib)
    plt.subplot(2,4,5)
    plot_img(Iv,"After morphology", cmap="gray")

    If = final_filtering(Iv,linear_structuring_elements)
    plt.subplot(2,4,6)
    plot_img(If,"After final stage of morphology","gray")


    Ic1 = opening(If, linear_structuring_elements)
    Ic_optimized1 = dilation_reconstruction(Ic1,If)
    plt.subplot(2,4,7)
    plot_img(Ic_optimized1,"After second morphology", "gray")

    # Adaptive thresholding --> find a threshold based on a local region of pixels
    maxValue = 255
    blockSize = 5
    offsetC = 5

    # result_mean_thresh = cv.adaptiveThreshold(If,maxValue,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,blockSize,offsetC)

    result_gauss_thresh = cv.adaptiveThreshold(Ic_optimized1,maxValue,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,blockSize,offsetC)
    plt.subplot(248)
    plot_img(result_gauss_thresh, "SEGMENTED IMAGE after adaptive thresholding", cmap='gray')

    plt.show()


    
    
    
