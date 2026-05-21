import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import pywt
import os
import pandas as pd

def create_linear_structuring_elements(element_len = 17, angle_step = 15):

    """
    A linear shape is defined as a bright part of an image with a minimum length L and a maximum width W. The vasculature in a retinal
    image is composed of many such connected linear shapes, and the aim of the initial processing is to preserve image structures which
    satisfy the criteria of being at least L pixels long, and no more than W pixels wide.
    The paper works with images with a different resolution (much lower because the paper is very old) so I needed to adapt the parameters
    to the high resolution of my images
    """

    Cx = element_len // 2
    Cy = element_len // 2
    radius = element_len // 2

    linear_structuring_elements = [] # place-holder

    for i in range(0,180,angle_step):

        kernel = np.zeros((element_len, element_len),dtype=np.uint8)
        radians_value = np.radians(i)

        # Rotation of the kernel
        x = radius * np.cos(radians_value)
        y = radius * np.sin(radians_value)

        P1 = (int(round(Cx - x)), int(round(Cy - y)))
        P2 = (int(round(Cx + x)), int(round(Cy + y)))

        cv.line(kernel, P1, P2, 255, 1) # Thickness is 1 pixel
        linear_structuring_elements.append(kernel)

    return linear_structuring_elements


#----------------------------------------------------------#



    """
    Opening the image with a linear structuring element, B, of length L1, and width 1 preserves linear shapes when the structuring
    element and the shape are approximately parallel. If many such structuring elements are used, with different angular rotations,
    then all linear shapes with length greater than or equal to L1 should be preserved by at least one rotation.

    Thus, a cleaner version Ic of the Io (original image), can be obtained by taking the supremum of the openings
    of the image with linear structuring elements with many different rotations.
    """

    max_opening = np.zeros_like(Io)

    for element in linear_elements:
        current_opening = cv.morphologyEx(Io, cv.MORPH_OPEN, element)
        max_opening = np.maximum(max_opening, current_opening)

    return max_opening


#----------------------------------------------------------#


def dilation_reconstruction(marker, mask):
    """
    Much of the small detail that is lost in the previous operation can be recovered if morphological reconstruction is used.
    Morphological reconstruction extracts the peaks in a mask (conditioning) image that are touched by a marker image. If the image
    produced by the opening is used as the marker image and the original image is used as the mask image then an improved version
    of Ic can be obtained (see equation (3.2) in the paper).
    """
    kernel = np.ones((3, 3), np.uint8)

    i=0
    while True: # until convergence

        previous_marker = marker.copy()
        dilated = cv.dilate(marker, kernel, iterations=1)
        marker = np.minimum(dilated, mask)
        i+=1
        if np.array_equal(marker, previous_marker):
            break

    return marker


#----------------------------------------------------------#


def create_1D_gaussian_ker(sigma,kernel_len):

    """
    Nel paper viene usato un sigma di 1.75 ma siccome le nostre immagini sono 3x risoluzione ho fatto un 3x di sigma.
    Di conseguenza ho anche aumentato la lunghezza del kernel facendo un x5 circa

    Sto facendo smoothing dell'immagine lungo la direzione dei vasi sanguigni per evitare che i bordi siano iregolari
    """

    r = kernel_len // 2
    r = np.arange(-r, r + 1)

    formula_first_term = 1 / (math.sqrt(2 * math.pi * (sigma ** 2)))
    formula_second_term = - (r ** 2) / (2 * (sigma ** 2))
    kernel = formula_first_term * np.exp(formula_second_term)

    return r, kernel


#----------------------------------------------------------#


def create_1D_laplacian_gaussian_ker(sigma, kernel_len, scaling):

    r = kernel_len // 2
    r = np.arange(-r, r + 1)

    first_term = (scaling*( (r**2) - (sigma**2) ) ) / (sigma**4)
    second_term = np.exp( -(r**2) / (2 * (sigma ** 2)) )
    kernel = -first_term * second_term

    return r, kernel


#----------------------------------------------------------#


def compose_kernels(r,kernel1,kernel2):

    radius = len(r)//2

    kernels = []
    base_kernel = np.outer(kernel1, kernel2)
    center = (radius,radius)
    angle_step = 15

    for angle in range(0,180,angle_step):
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        kernel_rotated = cv.warpAffine(base_kernel, rot_mat, (len(r), len(r)),
                                       flags=cv.INTER_LINEAR)
        kernels.append(kernel_rotated)

    return kernels


#----------------------------------------------------------#


def apply_filters(img, kernels):

    max_response = np.zeros_like(img, dtype=np.float32)

    for kernel in kernels:
        response = cv.filter2D(img.astype(np.float32), cv.CV_32F, kernel)
        max_response = np.maximum(max_response, response)

    max_response[max_response < 0] = 0
    final_image = cv.normalize(max_response, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return final_image


#----------------------------------------------------------#


"""
Final morphological filtering
"""

#----------------------------------------------------------#


def erosion_reconstruction(marker, mask):

    kernel = np.ones((3, 3), np.uint8)
    i = 0

    while True:
        previous_marker = marker.copy()
        dilated = cv.erode(marker, kernel, iterations=1)
        marker = np.maximum(dilated, mask)
        i += 1

        if np.array_equal(marker, previous_marker):
            break

    return marker


# ----------------------------------------------------------#


def final_filtering(max_response,linear_elements):

    sup = np.zeros_like(max_response)
    for element in linear_elements:
        open_img = cv.morphologyEx(max_response, cv.MORPH_OPEN, element)
        sup = np.maximum(sup, open_img)

    Il = dilation_reconstruction(sup,max_response)

    infimum = np.full_like(Il, 255)

    for element in linear_elements:
        cl = cv.morphologyEx(Il, cv.MORPH_CLOSE, element)
        infimum = np.minimum(infimum, cl)

    If = erosion_reconstruction(infimum, Il)
    return If


#----------------------------------------------------------#

"""
Final thresholding stage
"""

#----------------------------------------------------------#


def thresholding(If,low,high):

    If_norm = cv.normalize(If, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    _, I_low = cv.threshold(If_norm,low,255,cv.THRESH_BINARY)
    _, I_high = cv.threshold(If_norm,high,255,cv.THRESH_BINARY)

    thresh_result = dilation_reconstruction(marker=I_high,mask=I_low)
    return thresh_result


#----------------------------------------------------------#


def get_percentile_thresholds(img_response, p_high=97, p_low=85):

    """
    Per calcolare una soglia adatta a tutte le immagini sfrutto una misura statistica.
    Ipotizzo che in questo tipo di immagini, i vasi non ricoprano più del 3-5% dei pixel totali
    p_high=97: Considera il top 3% dei pixel come "Vasi Sicuri"
    p_low=85: Considera il top 15% dei pixel come "Vasi Possibili"
    """

    valid_pixels = img_response[img_response > 0] # estraggo i pixel con un valore diverso da 0

    if len(valid_pixels) == 0:
        return 20, 80  # Valori di fallback se l'immagine è nera

    t_high = np.percentile(valid_pixels, p_high)
    t_low  = np.percentile(valid_pixels, p_low)

    return t_low, t_high


#----------------------------------------------------------#


def wavelet_filtering(img):

    """
    Discrete wavelet transform multilevel for 2D data
    db3: forma del filtro che faccio scorrere sull'immagine
    level: quante volte rimpicciolisco e filtro l'immagine
    mode: come si comporta il filtro ai bordi
    """

    levels = 4
    coeffs = pywt.wavedec2(img,'db3',level=levels,mode='symmetric') # estrazione coefficienti [cA (cH4,cV4,cD4) (cH3,cV3,cD3) ...]
    details = coeffs[1:] # ignoro cA

    # VisuShrink per calcolare la soglia di filtraggio
    
    cH1, cV1, cD1 = coeffs[-1] # estraggo i coefficienti del primo livello
    mediana = np.median(np.abs(cD1))
    sigma = mediana / 0.6745
    N = img.size
    soglia_T = sigma * np.sqrt(2 * np.log(N))
    coeffs_puliti = [coeffs[0]] # una volta trovata la soglia devo thresholdare i dettagli e aggiungerli alla nuova lista

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cH_pulito = pywt.threshold(cH, value=soglia_T, mode='soft')
        cV_pulito = pywt.threshold(cV, value=soglia_T, mode='soft')
        cD_pulito = pywt.threshold(cD, value=soglia_T, mode='soft')
        
        coeffs_puliti.append((cH_pulito, cV_pulito, cD_pulito))

    img_pulita = pywt.waverec2(coeffs_puliti, 'db3')
    img_pulita = np.clip(img_pulita, 0, 255).astype(np.uint8)
    return img_pulita


#----------------------------------------------------------#

def gamma_correction(img,gamma=1.8):

    """
    Gamma correction adjusts the brightness of an image by applying
    a nonlinear function to the pixel values in the input image. Gamma
    correction improves contrast and detail in an image by making dark
    regions brighter and bright regions darker. The effects in the image
    cause it to be perceived as more realistic and natural.
    """

    gamma_corrected = np.array(255 * (img/255)**gamma, dtype='uint8')
    return gamma_corrected


def histogram_equalization(img):

    equ = cv.equalizeHist(img)
    return equ


def laplacian(img):

    sharpened_img = cv.Laplacian(img, cv.CV_64F)
    return sharpened_img


def sharpen_img(img):

    kernel1 = np.array([ [0,1,0], [1,-2,1], [0,1,0]])
    kernel2 = np.array([ [1,1,1], [1,-6,1], [1,1,1]])

    sharpened_image = cv.filter2D(img, -1, kernel2)

    return sharpened_image


def clahe(img,clipLimit=2,gridSize=2):

    clahe = cv.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridSize,gridSize))
    clahe_img = clahe.apply(img)
    return clahe_img


def create_subject_data(fname: str,best_subjects: list, src_path, masks_path, labels_path, csv_seg_path,draw_src_path,draw_option) -> dict:
    
    subject_dict = dict()

    if 'POST' in fname: # ad ogni iterazione processo la coppia PRE-POST
         return None
    
    subject = fname.split('PRE')[0] # esempio: da L_06PRE.JPG estraggo L_06
    phase = 'PRE'
    subject_phase = subject + phase

    # Se il soggetto non è tra i best non ci interessa
    if subject not in best_subjects:
        return None
    
    fname_post = f"{subject}POST.JPG" # estraggo il nome del file POST


    IMG_PATH1 = os.path.join(src_path,fname)
    IMG_PATH2 = os.path.join(src_path,fname_post)

    MASK_PATH1 = os.path.join(masks_path,fname)
    MASK_PATH2 = os.path.join(masks_path,fname_post)

    LABEL_PATH1 = os.path.join(labels_path,subject,'total_1.png')
    LABEL_PATH2 = os.path.join(labels_path,subject,'total_2.png')

    imgPre  = cv.imread(IMG_PATH1,0)
    imgPre = clahe(imgPre,2,4)

    imgPost = cv.imread(IMG_PATH2,0)
    imgPost = clahe(imgPost,2,4)

    maskPre  = cv.imread(MASK_PATH1,0)
    maskPost = cv.imread(MASK_PATH2,0)

    labelPre  = cv.imread(LABEL_PATH1,0)
    labelPost = cv.imread(LABEL_PATH2,0)

    # Controllo se la label è flippata in modo giusto o no
    flipped_cols_csv = pd.read_csv(csv_seg_path, index_col=False,usecols=['SUBJECT','Flipped'])    
    flipped_subject_state = flipped_cols_csv.loc[flipped_cols_csv['SUBJECT'] == subject_phase,'Flipped'].values[0]

    if flipped_subject_state: # se la label è specchiata rispetto all'immagine
        maskPre = cv.flip(maskPre,1)
        maskPost = cv.flip(maskPost,1)
        labelPre = cv.flip(labelPre,1)
        labelPost = cv.flip(labelPost,1)


    if draw_option == 3:
        fname_colored_superposition = os.path.join(draw_src_path,f'{subject}_confronto_post.png')
        img_colors = cv.imread(fname_colored_superposition)
        img_colors = cv.cvtColor(img_colors,cv.COLOR_BGR2RGB)
        
    elif draw_option == 1:
        img_colors = cv.cvtColor(imgPre,cv.COLOR_GRAY2BGR)
    else:
        img_colors = cv.cvtColor(labelPre,cv.COLOR_GRAY2BGR)

    subject_dict['ID'] = subject
    subject_dict['PHASE'] = phase
    subject_dict['ID and PHASE'] = subject_phase
    subject_dict['IMG PRE'] = imgPre
    subject_dict['IMG POST'] = imgPost
    subject_dict['LABEL PRE'] = labelPre
    subject_dict['LABEL POST'] = labelPost
    subject_dict['DRAW IMG'] = img_colors

    return subject_dict

    

def safe_mean(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Se non c'è nessun elemento di mask = true su cui calcolare la media allora returna 0
    return np.mean(data[mask]) if np.any(mask) else 0

















