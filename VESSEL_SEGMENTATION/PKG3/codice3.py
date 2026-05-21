from VESSEL_SEGMENTATION.PKG3.functions import *
import os

file_path = os.path.abspath(__file__)
GLOBAL_PATH = os.path.dirname(os.path.dirname(file_path))
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR_no_artifacts')
IMG_PATH = os.path.join(SRC_PATH,'L_06PRE.jpg')

"""
In questo codice ho provato a introdurre come fase iniziale del pre processing un filtraggio con le wavelet
E nella brightness correction con il clahe, ho tolto il median filter (magari lo faccio poi a parte)

Qui non sto usando il filtro gaussiano + laplacian perchè ho visto che introduceva molte curvature randomiche, volevo vedere cosa succedeva
a toglierlo
"""

if __name__ == "__main__":

    img = cv.imread(IMG_PATH,0)
    
    plt.figure(figsize=(15,15))

    plt.subplot(241)
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.title("original image")


    wavelet_img = wavelet_filtering(img)

    plt.subplot(242)
    plt.imshow(cv.subtract(wavelet_img,img),cmap='gray')
    plt.axis('off')
    plt.title("Rumore tolto col wavelet filtering")


    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(16,16))
    clahe_img = clahe.apply(wavelet_img)
   
    plt.subplot(243)
    plt.imshow(clahe_img,cmap='gray')
    plt.axis('off')
    plt.title("After CLAHE")

    linear_structuring_elements = create_linear_structuring_elements(element_len=21, angle_step=15)
    Ic = opening(clahe_img, linear_structuring_elements)
    Ic_optimized = dilation_reconstruction(Ic,clahe_img)

    plt.subplot(244)
    plt.imshow(Ic_optimized,cmap='gray')
    plt.axis('off')
    plt.title("IC OPTIMIZED")

    background = cv.medianBlur(src=clahe_img,ksize=69)
    Iv = cv.subtract(background,Ic_optimized)

    plt.subplot(245)
    plt.imshow(Iv,cmap='gray')
    plt.axis('off')
    plt.title("After first step of morphology")
    
    If = final_filtering(Iv,linear_structuring_elements)

    plt.subplot(246)
    plt.imshow(If,cmap='gray')
    plt.axis('off')
    plt.title("after final stage of morphology")
    
    # Segmentation 
    t_low, t_high = get_percentile_thresholds(If)
    res = thresholding(If,t_low,t_high)
    plt.subplot(247)
    plt.imshow(res, cmap='gray')
    plt.axis('off')
    plt.title("segmented image")

    plt.show()
    