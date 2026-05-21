#--------------------- LIBRARIES ---------------------#

import os
import cv2 as cv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Se hai importato funzioni tue, lascialo così. Altrimenti commentalo se ti dà errore.
from IMAGE_REGISTRATION.src.functions import * 

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) 

SRC_FLAT_PATH = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'IR_Def')
SRC_NESTED_PATH = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'IR_Def2')

# Cartelle di destinazione come da richiesta
DST_FIGURES_PATH = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results', 'figures')
DST_ALIGNED_PATH = os.path.join(DST_FIGURES_PATH, 'DATASET2_aligned')
LOG_PATH = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results', 'logs')

os.makedirs(DST_ALIGNED_PATH, exist_ok=True)
os.makedirs(DST_FIGURES_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# --- COSTANTI PER LA SCALE BAR ---
PIXEL_LENGTH = 4.651162790697675     # in um

# --- UTILS ---
def create_alignment_overlay(fixed_img, moving_img):
    """
    Crea l'overlay Ciano/Magenta (OpenCV format BGR).
    Fixed image  -> Cyan (Blue + Green channels)
    Moving image -> Magenta (Blue + Red channels)
    """
    h, w = fixed_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Canale Blu: fusione bilanciata per ottenere il grigio/bianco nelle aree allineate
    overlay[:, :, 0] = cv.addWeighted(fixed_img, 0.5, moving_img, 0.5, 0)
    # Canale Verde (Cyan component dell'immagine Fixed)
    overlay[:, :, 1] = fixed_img
    # Canale Rosso (Magenta component dell'immagine Moving)
    overlay[:, :, 2] = moving_img
    
    return overlay

def save_paper_pdf(overlay_bgr, out_path, w, h):
    """Genera e salva la figura PDF standalone pixel-perfect con layout 8x8 a 300 DPI."""
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    overlay_rgb = cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)
    ax.imshow(overlay_rgb, aspect='equal')

    # Salvataggio nativo in PDF vettoriale per LaTeX
    plt.savefig(out_path, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == '__main__':
    
    mapping_file = os.path.join(LOG_PATH, 'mapping_pazienti.txt')
    discarded_file = os.path.join(LOG_PATH, 'subjects_discarded.txt')
    
    # Reset dei log
    open(mapping_file, 'w', encoding='utf-8').close() 
    open(discarded_file, 'w', encoding='utf-8').close()
    
    print("Inizio Fase 1: Estrazione pazienti da ENTRAMBE le cartelle...")

    grouped_patients = defaultdict(lambda: {'pre': None, '1mo': None, '3mo': None, '12mo': None})
    
    # LETTURA 1: IR_Def (Piatta)
    if os.path.exists(SRC_FLAT_PATH):
        for fname in sorted(os.listdir(SRC_FLAT_PATH)):
            if fname.startswith('.'): continue
            name_without_ext = os.path.splitext(fname)[0]
            parts = name_without_ext.rsplit('_', 1)
            if len(parts) == 2:
                surname = f"IR_Def_{parts[0]}" 
                timepoint_str = parts[1]
                img_full_path = os.path.join(SRC_FLAT_PATH, fname)
                if timepoint_str == '0': grouped_patients[surname]['pre'] = img_full_path
                elif timepoint_str == '1': grouped_patients[surname]['1mo'] = img_full_path
                elif timepoint_str == '3': grouped_patients[surname]['3mo'] = img_full_path
                elif timepoint_str == '12': grouped_patients[surname]['12mo'] = img_full_path

    # LETTURA 2: IR_Def2 (Sottocartelle)
    if os.path.exists(SRC_NESTED_PATH):
        for patient_folder in sorted(os.listdir(SRC_NESTED_PATH)):
            patient_path = os.path.join(SRC_NESTED_PATH, patient_folder)
            if not os.path.isdir(patient_path) or patient_folder.startswith('.'): continue
            surname = f"IR_Def2_{patient_folder}"
            for fname in os.listdir(patient_path):
                fname_lower = fname.lower()
                img_full_path = os.path.join(patient_path, fname)
                
                if 'preop' in fname_lower or '_0.' in fname_lower or 'baseline' in fname_lower: 
                    grouped_patients[surname]['pre'] = img_full_path
                elif '12 mo' in fname_lower or '_12.' in fname_lower: 
                    grouped_patients[surname]['12mo'] = img_full_path
                elif '1 mo' in fname_lower or '_1.' in fname_lower: 
                    grouped_patients[surname]['1mo'] = img_full_path
                elif '3 mo' in fname_lower or '_3.' in fname_lower: 
                    grouped_patients[surname]['3mo'] = img_full_path

    print(f"Estrazione completata. Trovati {len(grouped_patients)} pazienti totali. Inizio Allineamento...\n")

    # --- FASE 2: ALLINEAMENTO ROBUSTO E PLOT PAPER ---
    subject_counter = 1
    count_success = 0

    for surname in sorted(grouped_patients.keys()):
        images_dict = grouped_patients[surname]
        
        if images_dict['pre'] is None:
            print(f"[-] SKIPPED {surname}: Manca immagine Baseline (PRE).")
            continue
            
        print(f"Elaborazione in corso: {surname} ...")

        imgPre = cv.imread(images_dict['pre'], 0)
        if imgPre is None: continue
        h, w = imgPre.shape

        # Pre-processing Baseline
        wavelet_pre = wavelet_filtering(imgPre)
        median_pre = cv.medianBlur(wavelet_pre, 5)
        clahe_pre = clahe(median_pre, clipLimit=1.5, gridSize=16)

        # --- ROI ORIGINALE (Solo Destra per l'Optic Nerve) ---
        larghezza_roi = int(w * 0.35)
        altezza_roi = int(h * 0.5)
        y_start = (h // 2) - (altezza_roi // 2)
        y_end = y_start + altezza_roi

        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv.rectangle(final_mask, (w - larghezza_roi, y_start), (w, y_end), 255, -1)

        sift = cv.SIFT_create(nfeatures=3000)
        kp1, des1 = sift.detectAndCompute(clahe_pre, mask=final_mask)

        if des1 is None or len(des1) < 10:
            print(f"[-] SKIPPED {surname}: Baseline troppo buia o senza dettagli sufficienti.")
            continue

        patient_aligned = False
        temp_subject_id = f"Subject_{subject_counter:03d}"
        subj_out_dir = os.path.join(DST_ALIGNED_PATH, temp_subject_id)

        for timepoint in ['1mo', '3mo', '12mo']:
            post_path = images_dict[timepoint]
            if post_path is None: continue 
                
            imgPost = cv.imread(post_path, 0)
            if imgPost is None: continue

            # Controllo e normalizzazione risoluzione geometrica
            h_post, w_post = imgPost.shape
            if (h_post, w_post) != (h, w):
                print(f"[!] Attenzione: Risoluzione diversa trovata per {surname} ({timepoint}). Ridimensionamento forzato.")
                imgPost = cv.resize(imgPost, (w, h), interpolation=cv.INTER_AREA)
            
            wavelet_post = wavelet_filtering(imgPost)
            median_post = cv.medianBlur(wavelet_post, 5)
            clahe_post = clahe(median_post, clipLimit=1.5, gridSize=16)

            kp2, des2 = sift.detectAndCompute(clahe_post, mask=final_mask)
            
            if des2 is None or len(des2) < 2:
                 continue
                 
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                matrix, mask = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC, ransacReprojThreshold=3.0)

                if matrix is not None:
                    # Controlli di coerenza fisica sulla trasformazione stimata
                    s = np.sqrt(matrix[0,0]**2 + matrix[0,1]**2)
                    angle = np.arctan2(matrix[1,0], matrix[0,0]) * (180 / np.pi)

                    if 0.95 < s < 1.05 and abs(angle) < 10:
                        matrix[:, :2] /= s  # Forziamo scala rigida unitaria pura
                    else:
                        print(f"[-] SKIPPED {surname}: Trasformazione non plausibile (S:{s:.2f}, A:{angle:.2f})")
                        continue 
                else:
                    continue

                # Se è il primo allineamento riuscito, creo la directory e salvo il PRE raw
                if not patient_aligned:
                    os.makedirs(subj_out_dir, exist_ok=True)
                    cv.imwrite(os.path.join(subj_out_dir, f"{temp_subject_id}_PRE.png"), imgPre)
                    patient_aligned = True

                # Generazione dell'immagine post-allineata geometricamente
                img_post_aligned = cv.warpAffine(imgPost, matrix, (w, h))
                cv.imwrite(os.path.join(subj_out_dir, f"{temp_subject_id}_POST_{timepoint}.png"), img_post_aligned)
                
                # =========================================================================
                # GENERAZIONE OVERLAYS CIANO/MAGENTA PER IL PAPER (FORMATO .PDF)
                # =========================================================================
                
                # 1. Overlay PRE-ALLINEAMENTO (Stato iniziale disallineato)
                overlay_unaligned = create_alignment_overlay(imgPre, imgPost)
                pdf_unaligned_path = os.path.join(DST_FIGURES_PATH, f"{temp_subject_id}_{timepoint}_Unaligned.pdf")
                save_paper_pdf(overlay_unaligned, pdf_unaligned_path, w, h)
                
                # 2. Overlay POST-ALLINEAMENTO (Stato finale registrato rigidamente)
                overlay_aligned = create_alignment_overlay(imgPre, img_post_aligned)
                pdf_aligned_path = os.path.join(DST_FIGURES_PATH, f"{temp_subject_id}_{timepoint}_Aligned.pdf")
                save_paper_pdf(overlay_aligned, pdf_aligned_path, w, h)

        # Scrittura sicura dei log ad ogni iterazione di successo
        if patient_aligned:
            with open(mapping_file, 'a', encoding='utf-8') as f:
                f.write(f"{temp_subject_id} = {surname}\n")
            print(f" [+] {temp_subject_id} ASSEGNATO -> {surname}")
            subject_counter += 1
            count_success += 1

    print(f"\n--- REPORT FINALE ---")
    print(f"Pazienti processati con successo: {count_success}")
    print(f"I file PDF per il paper sono pronti in: {DST_FIGURES_PATH}")