#--------------------- LIBRARIES ---------------------#

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------#

GLOBAL_PATH = os.getcwd()
PATH_DEF1 = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'IR_Def')
PATH_DEF2 = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'IR_Def2')

DST_PATH = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION','results','figures', 'detected_contours')
PAPER_DST_PATH = os.path.join(GLOBAL_PATH, 'PAPERS','myPaper', 'paper_imgs')

for path in [DST_PATH, PAPER_DST_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

#-----------------------------------------------------#

def process_retina_image(img_path, patient_id, phase, dataset_source):
    """
    Funzione che racchiude la logica di processing per una singola immagine.
    L'optic nerve è assunto essere SEMPRE sul lato destro ('dx').
    Genera sia la griglia di debug che le immagini singole ad alta risoluzione per il paper.
    """
    
    # Lettura immagine
    original_img = cv.imread(img_path, 0)
    if original_img is None:
        print(f"[ERRORE] Impossibile leggere il file: {img_path}")
        return
        
    h, w = original_img.shape # h = righe (y), w = colonne (x)

    # Preparo la griglia di debug (2x3)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(original_img, cmap='gray'), axs[0, 0].set_axis_off()

    # Rimozione artefatti bianco puro
    original_img[original_img > 253] = 0 
    axs[0, 1].imshow(original_img, cmap='gray'), axs[0, 1].set_axis_off()
    
    # Sfocatura e conversione colore per visualizzazione di debug
    blurred_img = cv.GaussianBlur(original_img, (101, 101), 0)
    img_colors = cv.cvtColor(blurred_img, cv.COLOR_GRAY2BGR)

    # Limiti ROI cercati manualmente
    x_limit = int(w * 0.23)
    y_limit_sup = int(h * 0.2)
    y_limit_inf = int(h * 0.33)     

    # Riempimento zone di exclusion (Nervo a DESTRA -> elimino la SINISTRA)
    blurred_img[0:y_limit_sup, :] = 0
    blurred_img[h-y_limit_inf:, :] = 0
    blurred_img[y_limit_sup:h-y_limit_inf, :w-x_limit] = 0

    axs[0, 2].imshow(blurred_img, cmap='gray'), axs[0, 2].set_axis_off()

    # Maschera ROI ed erosione
    roi_mask = (blurred_img > 0).astype(np.uint8) * 255
    kernel_erode = np.ones((201, 201), np.uint8)
    roi_mask = cv.erode(roi_mask, kernel_erode)
    
    roi_pixels = blurred_img[roi_mask > 0]
    if len(roi_pixels) == 0:
        print(f"[ATTENZIONE] ROI vuota per {patient_id} - {phase}. Salto l'immagine.")
        plt.close(fig)
        return

    # Thresholding basato su percentili
    low_thresh = np.percentile(roi_pixels, 10)  
    high_thresh = np.percentile(roi_pixels, 90) 

    _, dark_pixels_mask = cv.threshold(blurred_img, low_thresh, 255, cv.THRESH_BINARY_INV)
    _, bright_pixels_mask = cv.threshold(blurred_img, high_thresh, 255, cv.THRESH_BINARY)

    dark_pixels_mask[roi_mask == 0] = 0
    bright_pixels_mask[roi_mask == 0] = 0

    cnts_dark, _   = cv.findContours(dark_pixels_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnts_bright, _ = cv.findContours(bright_pixels_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    all_contours = cnts_dark + cnts_bright

    img_colors_copy1 = img_colors.copy()
    cv.drawContours(img_colors_copy1, all_contours, -1, color=(0, 0, 255), thickness=10)
    axs[1, 0].imshow(img_colors_copy1), axs[1, 0].set_axis_off()

    # Filtraggio contorni per trovare l'Optic Nerve
    target_cnt = None
    best_x_found = -1

    for cnt in all_contours:
        area = cv.contourArea(cnt)
        if area < 5000:
            continue

        current_x = np.max(cnt[:, 0, 0])
        if current_x > best_x_found:
            best_x_found = current_x
            target_cnt = cnt

    # Disegno del contorno target sulla griglia di debug
    img_colors_copy2 = img_colors.copy()
    if target_cnt is not None:
        cv.drawContours(img_colors_copy2, [target_cnt], -1, color=(0, 0, 255), thickness=10)
    axs[1, 1].imshow(img_colors_copy2), axs[1, 1].set_axis_off()
    
    # Maschera rettangolare fissa a destra sulla griglia di debug
    larghezza_roi = int(w * 0.25)
    altezza_roi = int(h * 0.5)
    y_start = (h // 2) - (altezza_roi // 2)
    y_end = y_start + altezza_roi
    cv.rectangle(img_colors, (w - larghezza_roi, y_start), (w, y_end), (0, 0, 255), -1)
    axs[1, 2].imshow(img_colors), axs[1, 2].set_axis_off()

    # Salvo la griglia di debug completa
    out_name = f"{dataset_source}_{patient_id}_{phase}.jpg"
    plt.savefig(os.path.join(DST_PATH, out_name), dpi=300)
    plt.close(fig)

    # =========================================================================
    # GENERAZIONE IMMAGINI SINGOLE PER IL PAPER (SULL'IMMAGINE INVECE CHE SFOCATA)
    # =========================================================================
    
    # 1) Immagine Singola per il Paper: Optic Nerve Localizzato (Contorno)
    if target_cnt is not None:
        paper_contour_img = cv.cvtColor(original_img, cv.COLOR_GRAY2BGR)
        cv.drawContours(paper_contour_img, [target_cnt], -1, color=(0, 0, 255), thickness=8)
        
        fig_paper1, ax_paper1 = plt.subplots(figsize=(8, 8))
        ax_paper1.imshow(paper_contour_img)
        ax_paper1.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Rimuove i bordi bianchi di matplotlib
        
        paper_contour_name = f"PAPER_{dataset_source}_{patient_id}_{phase}_contour.jpg"
        plt.savefig(os.path.join(PAPER_DST_PATH, paper_contour_name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig_paper1)

    # 2) Immagine Singola per il Paper: Maschera Rettangolare Alternativa Fissa
    paper_rect_img = cv.cvtColor(original_img, cv.COLOR_GRAY2BGR)
    cv.rectangle(paper_rect_img, (w - larghezza_roi, y_start), (w, y_end), (0, 0, 255), -1)
    
    fig_paper2, ax_paper2 = plt.subplots(figsize=(8, 8))
    ax_paper2.imshow(paper_rect_img)
    ax_paper2.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    paper_rect_name = f"PAPER_{dataset_source}_{patient_id}_{phase}_rectROI.jpg"
    plt.savefig(os.path.join(PAPER_DST_PATH, paper_rect_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig_paper2)

    print(f"[INFO] Generati file per debug e paper per: {patient_id} ({phase})")

#-----------------------------------------------------#

if __name__ == '__main__':

    # --- 1) PROCESSING PRIMO PATH: IR_Def (.png diretti nella cartella) ---
    print("\n--- Avvio elaborazione PATH 1 (IR_Def) ---")
    if os.path.exists(PATH_DEF1):
        for fname in sorted(os.listdir(PATH_DEF1)):
            if fname.endswith('.png') and 'IR_' in fname:
                name_clean = fname.replace('.png', '')
                parts = name_clean.split('IR_')
                
                patient_id = parts[0]   
                num_phase = parts[1]    
                
                phase = "PRE" if num_phase == "0" else f"{num_phase}mo"
                
                img_full_path = os.path.join(PATH_DEF1, fname)
                process_retina_image(img_full_path, patient_id, phase, dataset_source="Def1")
    else:
        print(f"[ATTENZIONE] Il path {PATH_DEF1} non esiste.")

    # --- 2) PROCESSING SECONDO PATH: IR_Def2 (Sottocartelle per Cognome, file .tif) ---
    print("\n--- Avvio elaborazione PATH 2 (IR_Def2) ---")
    if os.path.exists(PATH_DEF2):
        for subfolder in sorted(os.listdir(PATH_DEF2)):
            subfolder_path = os.path.join(PATH_DEF2, subfolder)
            
            if os.path.isdir(subfolder_path):
                patient_id = subfolder
                
                for fname in sorted(os.listdir(subfolder_path)):
                    if fname.endswith('.tif'):
                        fname_lower = fname.lower()
                        
                        if 'preop' in fname_lower:
                            phase = "PRE"
                        elif '1 mo' in fname_lower:
                            phase = "1mo"
                        elif '3 mo' in fname_lower:
                            phase = "3mo"
                        elif '12 mo' in fname_lower:
                            phase = "12mo"
                        else:
                            phase = "UNKNOWN"
                        
                        img_full_path = os.path.join(subfolder_path, fname)
                        process_retina_image(img_full_path, patient_id, phase, dataset_source="Def2")
    else:
        print(f"[ATTENZIONE] Il path {PATH_DEF2} non esiste.")

    print("\n--- Elaborazione completata! Controlla la cartella 'paper_figures' ---")