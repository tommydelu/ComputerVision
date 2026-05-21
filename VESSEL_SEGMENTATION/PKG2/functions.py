import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import pywt


"""
First pre-processing attempt: use the pre-processing technique of [1] (read in the abstract)
"""

def pre_processing_attempt1(img, median_ksize = 3, clip_limit = 2, grid_size = (2,2), a_min = 0, a_max = 255):

    median_blur = cv.medianBlur(img, median_ksize) # blurring using the median of the pixels in a window of ksize x ksize

    """
    divido in quadratini di grandezza tileGridSize l'immagine, e migliora il contrasto separatamente per ogni quadratino
    esalto i contrasti, ma se il contrasto è troppo alto lo limito con clip limit
    """

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # con np.clip forzo tutti i valori dell'immagine a stare dentro il range a_min a_max
    clahe_img = np.clip(clahe.apply(median_blur), a_min, a_max) # per tenere l'img in un range controllato

    return clahe_img


#----------------------------------------------------------#

"""
FUNCTIONS FOR THE METHODOLOGY OF [3]
The aim of the initial morphological filtering operations is to emphasise the vasculature, preserval vessel crossings and bifurcations.
"""

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


def opening(Io, linear_elements):

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


def extract_background(Io, linear_elements):

    min_opening = np.zeros_like(Io)+255

    for element in linear_elements:
        opening = cv.morphologyEx(Io, cv.MORPH_OPEN, element)
        min_opening = np.minimum(min_opening, opening)

    return min_opening


#----------------------------------------------------------#


"""
Paragrafo 3.3.2, Second derivative properties of the vasculature
Il profilo di luminosità di un vaso sanguigno ha la forma di una curva Gaussiana: al centro del vaso la luminosità è massima,
ai bordi scende gradualmente verso lo sfondo.
La derivata seconda misura la concavità di una funzione, e per una forma a campana la derivata seconda è fortemente negativa proprio
in corrispondenza del picco (dentro il vaso) e vicina a zero nelle zone a pendenza costante --> quindi calcolo il negativo
della derivata seconda, così ottengo valori molto alti al centro del vaso e valori negativi fuori --> sto accendendo i vasi

Vantaggio per vasi sottili: più il vaso è sottile, più forte sarà la risposta del filtro.

Problema del rumore: prima si applica uno smoothing Gaussiano, lo smoothing è direzionale lungo il vaso, se lo facessi in tutte le
direzioni perderei dettagli dei vasi. Poi applico la derivata seconda attraverso il vaso per rilevare i bordi
"""

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