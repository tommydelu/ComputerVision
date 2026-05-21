import os
import cv2 as cv
import numpy as np
import pandas as pd

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) 

# Cartella con i tuoi risultati anonimizzati (SIFT)
SIFT_ALIGNED_DIR = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results', 'figures', 'DATASET2_aligned')

# Cartella con le immagini originali/manuali (Assumiamo sia la raw IR_Def come hai detto)
MANUAL_ALIGNED_DIR = os.path.join(GLOBAL_PATH, 'DATA', 'DATASET2', 'raw', 'Immagini_IR')

# Il log del mapping
MAPPING_FILE = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results', 'logs', 'mapping_pazienti.txt')
RESULTS_CSV = os.path.join(GLOBAL_PATH, 'IMAGE_REGISTRATION', 'results', 'data','RMSE_vs_Manual.csv')


def load_mapping(filepath):
    """Legge il file txt e restituisce un dizionario {Subject_001: IR_Def_Cognome}"""
    mapping = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    parts = line.strip().split('=')
                    subj_id = parts[0].strip()
                    orig_name = parts[1].strip()
                    mapping[subj_id] = orig_name
    return mapping

if __name__ == '__main__':
    
    mapping = load_mapping(MAPPING_FILE)
    if not mapping:
        print("[ERRORE] File di mapping non trovato o vuoto.")
        exit()

    results = []

    # Iteriamo sui soggetti allineati dal SIFT
    for subject_id in sorted(os.listdir(SIFT_ALIGNED_DIR)):
        subj_path = os.path.join(SIFT_ALIGNED_DIR, subject_id)
        if not os.path.isdir(subj_path) or subject_id not in mapping:
            continue

        original_prefix = mapping[subject_id]

        # Come hai detto tu, abbiamo il ground truth SOLO per i pazienti di IR_Def
        if not original_prefix.startswith('IR_Def_'):
            continue

        # Estraiamo il vero cognome togliendo il prefisso 'IR_Def_'
        real_surname = original_prefix.replace('IR_Def_', '')

        # Mappiamo i timepoint del nostro codice con i suffissi originali dei file medici
        timepoint_map = {'1mo': '1', '3mo': '3', '12mo': '12'}

        for tp_sift, tp_manual in timepoint_map.items():
            
            path_sift = os.path.join(subj_path, f"{subject_id}_POST_{tp_sift}.png")
            
            if not os.path.exists(path_sift):
                continue # Se l'allineamento SIFT per questo mese è fallito, saltiamo

            # Costruiamo il nome del file manuale (Es. Cognome_1.png o .tif)
            # Cerchiamo il file esatto nella cartella MANUAL_ALIGNED_DIR
            manual_file_path = None
            for f in os.listdir(MANUAL_ALIGNED_DIR):
                # Cerchiamo un file che inizi col cognome e finisca con _1 (ignorando l'estensione)
                nome_senza_ext = os.path.splitext(f)[0]
                if nome_senza_ext == f"{real_surname}_{tp_manual}":
                    manual_file_path = os.path.join(MANUAL_ALIGNED_DIR, f)
                    break

            if manual_file_path and os.path.exists(manual_file_path):
                img_sift = cv.imread(path_sift, 0)
                img_man = cv.imread(manual_file_path, 0)

                # Se le dimensioni differiscono, facciamo un resize del manuale
                # (permettendo la comparazione pixel per pixel)
                if img_sift.shape != img_man.shape:
                    img_man = cv.resize(img_man, (img_sift.shape[1], img_sift.shape[0]))

                # --- CALCOLO RMSE ---
                # Differenza assoluta tra i pixel
                diff = cv.absdiff(img_sift, img_man)
                # Mean Square Error (Errore Quadratico Medio)
                mse = np.mean(diff**2)
                # Root Mean Square Error
                rmse = np.sqrt(mse)

                print(f"[{subject_id} - {tp_sift}] vs [{os.path.basename(manual_file_path)}] -> RMSE: {rmse:.2f}")

                results.append({
                    'Subject': subject_id,
                    'Original_Name': real_surname,
                    'Timepoint': tp_sift,
                    'RMSE_Pixel_Intensity': rmse
                })
            else:
                print(f"[ATTENZIONE] File manuale non trovato per {real_surname} a {tp_sift}")

    # Salvataggio finale per Excel/Paper
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\n[FINE] Metriche calcolate con successo! Risultati salvati in: {RESULTS_CSV}")
        print(f"Errore Globale Medio (RMSE): {df['RMSE_Pixel_Intensity'].mean():.2f}")
    else:
        print("Nessun confronto possibile effettuato.")