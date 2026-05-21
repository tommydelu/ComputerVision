import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- I TUOI PATH ---
GLOBAL_PATH = os.getcwd()
POST_SIFT_IMG_PATH = os.path.join(GLOBAL_PATH,'IMAGE_REGISTRATION','results','figures','aligned_imgs1')
PRE_SIFT_IMG_PATH = os.path.join(GLOBAL_PATH,'DATA','DATASET1','raw','IR')
DST_PATH = os.path.join(GLOBAL_PATH,'IMAGE_REGISTRATION','results','figures','aligned_superpos1')

os.makedirs(DST_PATH, exist_ok=True)

# --- COSTANTI PER LA SCALE BAR ---
PIXEL_LENGTH = 4.651162790697675     # in um
SCALE_BAR_UM = 500                   # Vogliamo una barra che rappresenti 500 um
SCALE_BAR_PX = SCALE_BAR_UM / PIXEL_LENGTH

def create_alignment_overlay(fixed_img, moving_img):
    h, w = fixed_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    overlay[:,:,0] = 0           # Canale Blu a zero
    overlay[:,:,1] = moving_img  # Canale Verde (Post-op)
    overlay[:,:,2] = fixed_img   # Canale Rosso (Pre-op)

    return overlay

if __name__ == '__main__':

    for fname in sorted(os.listdir(PRE_SIFT_IMG_PATH)):
        
        if 'POST' in fname:
            continue
        
        subject = fname.split('PRE')[0]
        fname_post = f"{subject}POST.jpg"

        FIXED_IMG_PATH = os.path.join(PRE_SIFT_IMG_PATH,fname)
        POST_IMG_PRE_SIFT_PATH = os.path.join(PRE_SIFT_IMG_PATH,fname_post)
        POST_IMG_POST_SIFT_PATH = os.path.join(POST_SIFT_IMG_PATH,fname_post)

        if not os.path.exists(POST_IMG_POST_SIFT_PATH):
            continue

        fixed_img = cv.imread(FIXED_IMG_PATH,0)
        post_img_pre_sift = cv.imread(POST_IMG_PRE_SIFT_PATH,0)
        post_img_post_sift = cv.imread(POST_IMG_POST_SIFT_PATH,0)

        overlay_pre = create_alignment_overlay(fixed_img, post_img_pre_sift)
        overlay_post = create_alignment_overlay(fixed_img, post_img_post_sift)

        # Creazione della figura stile paper
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.35))
        
        overlay_pre_rgb = cv.cvtColor(overlay_pre, cv.COLOR_BGR2RGB)
        overlay_post_rgb = cv.cvtColor(overlay_post, cv.COLOR_BGR2RGB)

        # Dimensioni immagine per calcolare le coordinate della barra
        h, w, _ = overlay_post_rgb.shape

        # Pannello A: Pre-Allineamento
        axes[0].imshow(overlay_pre_rgb)
        axes[0].set_title('A. Pre-Alignment', loc='left', fontweight='bold', fontsize=10)
        axes[0].axis('off')

        # Pannello B: Post-Allineamento
        axes[1].imshow(overlay_post_rgb)
        axes[1].set_title('B. Post-Alignment (SIFT)', loc='left', fontweight='bold', fontsize=10)
        axes[1].axis('off')

        # --- AGGIUNTA DELLA SCALE BAR (Più sottile ed elegante) ---
        # Margini leggermente ridotti per non invadere troppo l'immagine
        margin_x = 20
        margin_y = 20
        bar_height = 4 # Spessore dimezzato (era 8)
        
        start_x = w - SCALE_BAR_PX - margin_x
        start_y = h - margin_y - bar_height

        # Disegna il rettangolo della barra
        rect = patches.Rectangle((start_x, start_y), SCALE_BAR_PX, bar_height, 
                                 linewidth=0, edgecolor='none', facecolor='white')
        axes[1].add_patch(rect)

        # Aggiunge il testo "500 µm"
        text_x = start_x + (SCALE_BAR_PX / 2)
        text_y = start_y - 6 # Avvicinato alla barra (era 10)
        axes[1].text(text_x, text_y, '500 µm', color='white', 
                     ha='center', va='bottom', fontsize=6) # Font ridotto a 6 e tolto il grassetto

        plt.tight_layout()

        # Salvataggio in alta risoluzione
        out_path = os.path.join(DST_PATH, f"{subject}_alignment.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)