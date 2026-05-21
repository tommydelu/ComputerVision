#------------------------------ LIBRARIES ------------------------------#


import os
import cv2 as cv
import numpy


#-----------------------------------------------------------------------#

"""
Idea: isolare l'angolo in basso a sx dell'immagine per centrare il tutto sulla barretta che definisce la scala. Poi sfruttare il fatto
che la barretta è completamente bianca (valore 255) e sfruttando i contorni trovare la sua bounding box e quindi la larghezza in pixel
"""


img = cv.imread('DATASET/IR/L_01POST.JPG', 0)
img_color = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
h, w = img.shape

# Centratura sulla barretta
y_offset = h - 100
x_offset = 30

roi = img[y_offset:h, x_offset:200]

# Thresholding per isolare il bianco della barretta
_, thresh = cv.threshold(roi, 240, 255, cv.THRESH_BINARY)

# Troviamo i contorni sull'immagine binaria
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


if contours:
    
    # Trovo il contorno con area massima
    cnt = max(contours, key=cv.contourArea)
    cnt_offset = cnt + [x_offset, y_offset]
    cv.drawContours(img_color, [cnt_offset], -1, (0,0,255), 2)
    x, y, w_bar, h_bar = cv.boundingRect(cnt)
    
    print(f"La barra è lunga {w_bar} pixel")
    lunghezza_barra_px = w_bar


cv.imshow('Immagine con identificata la barretta per la scala',img_color)
cv.waitKey(0)
cv.destroyAllWindows()

pixel_length = 200/43

print("Conversione da pixel a um...")
print(f"Lunghezza in um della barra: 200 um / 43 pixel = {pixel_length}")