"""
Visto che sto calcolando l'optical flow solo su 10 soggetti, e visto che trovare automaticamente il centro della fovea in queste immagini
infrarosso è difficile, posso cliccare io manualmente il centro della fovea, così posso migliorare i risutlati siccome sono più coerente
con la procedura fatta dall'ospdedale
"""

import cv2 as cv
import os
import numpy as np

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET1','raw','IR',)
DST_PATH = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results','figures','fovea_clicks_visualized')

best_subjects = ['L_03','L_06','L_15','L_16','L_42','L_48','L_63','L_78','S_36','S_46']
coordinate_fovea = {}

os.makedirs(DST_PATH, exist_ok=True)

def click_event(event, x, y, flags, params):
    
    """
    event: è una variabile che OpenCV riempie automaticamente ogni volta che succede qualcosa sopra la finestra dell'immagine (muovi il mouse, clicchi, rilasci il tasto).
    cv.EVENT_LBUTTONDOWN: è una costante di OpenCV che sta per "Event Left Button Down" (Evento Tasto Sinistro Abbassato). 
    Matematicamente ha un valore numerico (solitamente 1), ma si usa il nome per rendere il codice leggibile.
    """

    if event == cv.EVENT_LBUTTONDOWN:

        img_display = params['img_display']
        scale_x = params['scale_x']
        scale_y = params['scale_y']
        subj = params['subject']

        real_x, real_y = int(x * scale_x), int(y * scale_y)
        coordinate_fovea[subj] = (real_x, real_y)

        temp_view = img_display.copy()

        cv.circle(temp_view, (x, y), 5, (0, 255, 0), -1) 
        cv.drawMarker(temp_view, (x, y), (0, 0, 255), cv.MARKER_CROSS, 15, 2)

        cv.imshow(f"Clicca la fovea per {subj}", temp_view)
        print(f"{subj}: Fovea cliccata a schermo ({x}, {y}) -> Reale ({real_x}, {real_y})")
        print("Premi un tasto sulla tastiera per confermare e passare al prossimo...")






for subj in best_subjects:

    fname = f"{subj}PRE.JPG"
    img_path = os.path.join(SRC_PATH, fname)
    img_original = cv.imread(img_path)
    
    if img_original is not None:

        if len(img_original.shape) == 2: # Se è già grigia
            img_original_color = cv.cvtColor(img_original, cv.COLOR_GRAY2BGR)
        else:
            img_original_color = img_original.copy()

        img_show = cv.resize(img_original_color, (800, 800))
        sx = img_original.shape[1] / 800
        sy = img_original.shape[0] / 800

        params = {
            'subject': subj,
            'img_display': img_show,
            'scale_x': sx,
            'scale_y': sy
        }
        
        
        cv.imshow(f"Clicca la fovea per {subj}", img_show)
        
        cv.setMouseCallback(f"Clicca la fovea per {subj}", click_event, params)

        cv.waitKey(0)
        cv.destroyAllWindows()

        if subj in coordinate_fovea:
            real_coords = coordinate_fovea[subj]
            img_to_save = img_original_color.copy()
            cv.drawMarker(img_to_save, real_coords, (0, 0, 255), cv.MARKER_CROSS, 80, 5)
            cv.circle(img_to_save, real_coords, 10, (0, 255, 0), -1)
            text = f"Fovea: {real_coords}"
            cv.putText(img_to_save, text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            save_fname = f"{subj}_fovea_clicked.jpg"
            save_path = os.path.join(DST_PATH, save_fname)
            cv.imwrite(save_path, img_to_save)
            print(f"Immagine salvata in: {save_path}")

            
print("\n--- CICLO COMPLETATO ---")
print("Copia questo dizionario nel tuo script principale dell'Optical Flow:")
print(coordinate_fovea)
        











        

        
        
        
        
        
    
        
            
            
            
            
            


        
       
        
        

    
        
        
       
        

