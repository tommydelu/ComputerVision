import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

GLOBAL_PATH = os.getcwd()
IMG_PATH1 = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation', 'results', 'figures', "bigVessels_1")
IMG_PATH2 = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation', 'results', 'figures', "thinVessels_1")

# Cartella dedicata per le maschere a colori del paper
DST_PATH_PAPER_COLOR = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation', 'results', 'figures', "paper_color_masks")

border_limit = 50 # pixels
os.makedirs(DST_PATH_PAPER_COLOR, exist_ok=True)

if __name__ == "__main__":
    
    print("Inizio generazione maschere a colori per il paper...")

    for fname in sorted(os.listdir(IMG_PATH1)):
        if fname.startswith('.'):
            continue

        path1 = os.path.join(IMG_PATH1, fname)
        path2 = os.path.join(IMG_PATH2, fname)

        if not os.path.exists(path2):
            print(f"[-] File {fname} non trovato in thinVessels, salto.")
            continue

        imgBig = cv.imread(path1, 0)
        imgThin = cv.imread(path2, 0)
        h, w = imgBig.shape

        # 1. Taglio dei bordi su entrambe le maschere prima del merge
        for img in [imgBig, imgThin]:
            img[0:border_limit, :] = 0
            img[:, 0:border_limit] = 0
            img[h-border_limit:, :] = 0
            img[:, w-border_limit:] = 0

        # 2. Creazione dell'immagine a colori (RGB) a sfondo nero
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Assegnazione dei colori:
        # Verde [0, 255, 0] per i vasi secondari/sottili
        color_mask[imgThin > 0] = [0, 255, 0]
        # Rosso [255, 0, 0] per i vasi primari/grandi (sovrascrive in caso di overlap)
        color_mask[imgBig > 0] = [255, 0, 0]

        # --- CREAZIONE FIGURA SINGOLA (PAPER FORMAT) ---
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        # Mostra la maschera a colori (non serve cmap='gray' perché è un'immagine RGB)
        ax.imshow(color_mask)
        ax.axis('off')

        # Salvataggio stretto senza margini bianchi
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_name = fname.split('.')[0] + "_color_mask_paper.png"
        out_path_paper = os.path.join(DST_PATH_PAPER_COLOR, fig_name)
        
        plt.savefig(out_path_paper, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" [+] Generata maschera a colori per: {fname}")

    print(f"\nFinito! Controlla la cartella: {DST_PATH_PAPER_COLOR}")