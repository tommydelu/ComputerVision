from VESSEL_SEGMENTATION.PKG1.functions import *
import os

"""
To start this project, first of all, since I did not know any methodology for the segmentation of the retinal vessels, I looked for some
references on the net

References
[1] https://www.nature.com/articles/s41598-020-71010-0
[2] https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/

[3] https://pubmed.ncbi.nlm.nih.gov/12426111/ ---> the methodology I tried to implement comes from this paper
"""

file_path = os.path.abspath(__file__)
GLOBAL_PATH = os.path.dirname(os.path.dirname(file_path))
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR_no_artifacts')
IMG_PATH = os.path.join(SRC_PATH,'L_06PRE.jpg')


if __name__ == "__main__":

    original_img = cv.imread(IMG_PATH,0)

    plt.figure(figsize=(15,15))
    plt.subplot(231)
    plt.imshow(original_img,cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    """
    # From [1] I tried the so called CLAHE (to enhance img contrast), in [2] I looked for the definitions of the parameters
    """

    # See utilities/functions/pre_processing_attempt1
    clahe = pre_processing_attempt1(original_img, 
                                    median_ksize=15, 
                                    clip_limit=2, 
                                    grid_size=(16,16), 
                                    a_min=0, 
                                    a_max=255)
    
    clahe = cv.bitwise_not(clahe) # [3] worked with bright vessels on a darker background so I did this inversion

    plt.subplot(232)
    plt.imshow(clahe,cmap='gray')
    plt.title("After brightness enhancement with CLAHE")
    plt.axis('off')

    """
    Here I start to apply the methodology of [3]
    1) Create 12 rotated kernels (linear structuring elements)
    2) Apply morphology operations
    """

    linear_structuring_elements = create_linear_structuring_elements(element_len=21, angle_step=15)

    Ic = opening(clahe, linear_structuring_elements)
    plt.subplot(233)
    plt.imshow(Ic,cmap='gray')
    plt.title("After opening")
    plt.axis('off')

    Ic_optimized = dilation_reconstruction(Ic, clahe)
    plt.subplot(234)
    plt.imshow(Ic_optimized,cmap='gray')
    plt.title("Optimized opening")
    plt.axis('off')

    Ib = extract_background(clahe, linear_structuring_elements)
    plt.subplot(235)
    plt.imshow(Ib,cmap='gray')
    plt.title("Bacground extracted")
    plt.axis('off')

    Iv = cv.subtract(Ic_optimized, Ib)
    plt.subplot(236)
    plt.imshow(Iv,cmap='gray')
    plt.title("Final image after morphology")
    plt.axis('off')

    plt.show()


    """
    End of first morphological filtering stage
    Starting of filtering with gaussian and laplacian kernel (enhancement of vasculature)
    """

    r1,kernel1 = create_1D_gaussian_ker(sigma=5, 
                                        kernel_len=51)
    r2,kernel2 = create_1D_laplacian_gaussian_ker(sigma=5, 
                                                  kernel_len=51, 
                                                  scaling=10)

    resulting_kernel = compose_kernels(r1,kernel1,kernel2)

    Idiff = apply_filters(Iv,resulting_kernel)

    plt.figure(figsize=(15,15))
    plt.subplot(221)
    plt.imshow(original_img,cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(Idiff,cmap='gray')
    plt.title("After gaussian + laplacian filtering")
    plt.axis('off')

    """
    End of Gaussian + Laplacian filtering stage
    Now: final morphology operation
    """

    If = final_filtering(Idiff,linear_structuring_elements)
    plt.subplot(223)
    plt.imshow(If,cmap='gray')
    plt.title("After final morphology stage")
    plt.axis('off')


    """
    Final stage: segmentation of the image
    """

    t_low, t_high = get_percentile_thresholds(If, p_high=95, p_low=85)
    final_result = thresholding(If, t_low, t_high)

    plt.subplot(224)
    plt.imshow(final_result,cmap='gray')
    plt.title("SEGMENTED IMAGE")
    plt.axis('off')

    plt.show()















   


    

    

    

    


    






