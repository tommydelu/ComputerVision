#--------------------- LIBRARIES ---------------------#


import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------------------#


GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR')
DST_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','detected_contours')
txt_path = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','lato_nervo.txt')

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)

# Salvo l'informazione del lato corretto del nervo ottico
info_lati = {}
with open(txt_path, 'r') as f:
    for riga in f:
        parti = riga.strip().split()
        if len(parti) == 2:
            info_lati[parti[0]] = parti[1]


#-----------------------------------------------------#


if __name__ == '__main__':

    for fname in sorted(os.listdir(SRC_PATH)):

        if 'PRE' in fname:
            id = fname.split('PRE')[0]
            phase = 'PRE'
        elif 'POST' in fname:
            id = fname.split('POST')[0]
            phase = 'POST'

        
        lato_nervo = info_lati.get(id) # estraggo il lato corretto del nervo

        if lato_nervo is None:
            print(f"\n[ATTENZIONE] ID {id} non trovato nel file di testo per il file {fname}")
            continue



        #-------------------- CREO Region of Interest --------------------#


        IMG_PATH = os.path.join(SRC_PATH,fname)
        original_img = cv.imread(IMG_PATH,0)      # 1908 x 1908        
        h,w = original_img.shape                  # h = righe (y), w = colonne (x)

        fig, axs = plt.subplots(2,3,figsize=(12,8))
        axs[0,0].imshow(original_img,cmap='gray'), axs[0,0].set_axis_off()

        original_img[original_img > 253] = 0 # Le zone bianco puro (255 che sono artefatti le metto a 0)
        axs[0,1].imshow(original_img,cmap='gray'), axs[0,1].set_axis_off()
        
        blurred_img = cv.GaussianBlur(original_img,(101,101),0) # problema di un gaussian blur forte: crea zone molto scure vicino ai bordi --> devo espandere la zona vietata in cui cercare erodendo la maschera
        
        img_colors = cv.cvtColor(blurred_img,cv.COLOR_GRAY2BGR) # 1908 x 1908 x 3, da usare per visualizzazioni a colori


        # Definisco una x e una y limite entro cui cercare il nervo ottico: ho cercato manualmente questi valori
        x_limit = int(w * 0.23)
        y_limit_sup = int(h * 0.2)
        y_limit_inf = int(h * 0.33)     

        # Riempio di nero le zone di esclusione
        blurred_img[0:y_limit_sup,:] = 0
        blurred_img[h-y_limit_inf:,:] = 0
        if lato_nervo == 'sx':
            blurred_img[y_limit_sup:h-y_limit_inf,x_limit::] = 0
        elif lato_nervo == 'dx':
            blurred_img[y_limit_sup:h-y_limit_inf,:w-x_limit] = 0

        axs[0,2].imshow(blurred_img,cmap='gray'), axs[0,2].set_axis_off()


        #-----------------------------------------------------------------#


        roi_mask = (blurred_img > 0).astype(np.uint8)*255
        kernel_erode = np.ones((201,201), np.uint8)
        roi_mask = cv.erode(roi_mask,kernel_erode)
        roi_pixels = blurred_img[roi_mask > 0]


        """
        Ragionamento che ho fatto: nelle immagini IR spesso il nervo ottico ha l'intensità di pixels più scure/chiare di tutta l'immagine.
        Quindi posso identificare questi valori, capire in che lato dell'immagine si trovano, se dx o sx, ed escludere l'altra parte dell'immagine,
        per ridurre ancora di più la zona di esclusione e trovare meno contorni errati.
        Unico problema: a volte le zone più scure/chiare corrispondono a vasi/background, quindi devo usare un metodo a punteggio, del tipo,
        se il contorno trovato si trova nella parte giusta ha un altopunteggio se no basso, insomma un sistema a punteggi
        """

        low_thresh = np.percentile(roi_pixels,10)  # caso nervo ottico scuro: prendo i pixel più scuri
        high_thresh = np.percentile(roi_pixels,90) # caso nervo ottico chiaro: prendo i pixel chiari

        _, dark_pixels_mask = cv.threshold(blurred_img,low_thresh,255,cv.THRESH_BINARY_INV)
        _, bright_pixels_mask = cv.threshold(blurred_img,high_thresh,255,cv.THRESH_BINARY)

        dark_pixels_mask[roi_mask == 0] = 0
        bright_pixels_mask[roi_mask == 0] = 0

        cnts_dark, _   = cv.findContours(dark_pixels_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts_bright, _ = cv.findContours(bright_pixels_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        all_contours = cnts_dark + cnts_bright

        img_colors_copy1 = img_colors.copy()
        cv.drawContours(img_colors_copy1,all_contours,-1,color=(0,0,255),thickness=10)
        axs[1,0].imshow(img_colors_copy1), axs[1,0].set_axis_off()


        """
        Adesso che ho i contorni devo iniziare a usare delle loro proprietà o anche conoscenze a priori che ho io per filtrare questi
        contorni, e riuscire a ottenere solo quello giusto
        """

        target_cnt = None
        best_x_found = float('inf') if lato_nervo == 'sx' else -1

        most_left_cnt = None
        min_x_coordinate = -1

        for i,cnt in enumerate(all_contours):

            # Area of contour
            area = cv.contourArea(cnt)
            if area < 5000:
                continue

            # Aspect ratio: see open cv documentation
            x,y,rect_w,rect_h = cv.boundingRect(cnt)
            aspect_ratio = float(rect_w)/rect_h

            if lato_nervo == 'sx':
                current_x = np.min(cnt[:, 0, 0])
                if current_x < best_x_found:
                    best_x_found = current_x
                    target_cnt = cnt

            elif lato_nervo == 'dx':
                current_x = np.max(cnt[:, 0, 0])
                if current_x > best_x_found:
                    best_x_found = current_x
                    target_cnt = cnt


        img_colors_copy2 = img_colors.copy()
        cv.drawContours(img_colors_copy2,[target_cnt],-1,color=(0,0,255),thickness=10)
        axs[1,1].imshow(img_colors_copy2), axs[1,1].set_axis_off()
        





        """
        Ragionamento alternativo: visto che so a priori il lato corretto del nervo ottico, e visto che so a priori in che zona è
        situato (zona centrale, praticamente attaccato al bordo), per risparmiare tempo e usare una tecnica più semplice (che si potrà automatizzare)
        disegno semplicemente una maschera rettangolare che vada a coprire il nervo ottico in tutti i casi
        """

        larghezza_roi = int(w * 0.25)
        altezza_roi = int(h * 0.5)
        y_start = (h // 2) - (altezza_roi // 2)
        y_end = y_start + altezza_roi

        if lato_nervo == 'sx':
            cv.rectangle(img_colors, (0, y_start), (larghezza_roi, y_end), (0,0,255), -1)
        elif lato_nervo == 'dx':
            cv.rectangle(img_colors, (w - larghezza_roi, y_start), (w, y_end),  (0,0,255), -1)

        axs[1,2].imshow(img_colors), axs[1,2].set_axis_off()

        file_name = f"{id}{phase}.jpg"
        plt.savefig(os.path.join(DST_PATH,file_name), dpi=300)
        plt.close(fig)





       





       

       





        
    
       

      
       

       
       



        
        