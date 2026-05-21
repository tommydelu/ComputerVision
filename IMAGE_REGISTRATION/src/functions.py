import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pywt


def wavelet_filtering(img,display=False):

    """
    Discrete wavelet transform multilevel for 2D data
    db3: forma del filtro che faccio scorrere sull'immagine
    level: quante volte rimpicciolisco e filtro l'immagine
    mode: come si comporta il filtro ai bordi
    """

    levels = 4
    coeffs = pywt.wavedec2(img,'db3',level=levels,mode='symmetric') # estrazione coefficienti [cA (cH4,cV4,cD4) (cH3,cV3,cD3) ...]
    details = coeffs[1:] # ignoro cA

    if display:
        fig, axes = plt.subplots(levels, 3, figsize=(15, levels*4))
        for i, (cH, cV, cD) in enumerate(details):
            actual_level = levels - i
            
            axes[i, 0].imshow(np.abs(cH), cmap='gray')
            axes[i, 0].set_title(f"Livello {actual_level} - Orizzontale (cH)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(np.abs(cV), cmap='gray')
            axes[i, 1].set_title(f"Livello {actual_level} - Verticale (cV)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(np.abs(cD), cmap='gray')
            axes[i, 2].set_title(f"Livello {actual_level} - Diagonale (cD)")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95) 
        plt.show()

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



def clahe(img,clipLimit=2,gridSize=2):

    clahe = cv.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridSize,gridSize))
    clahe_img = clahe.apply(img)
    return clahe_img


    


















