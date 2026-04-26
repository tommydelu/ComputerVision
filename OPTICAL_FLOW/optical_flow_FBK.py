
import os
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------------------------------------#


# I miei path
GLOBAL_PATH        = os.getcwd()
IR_ALIGNED_PATH    = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'aligned_imgs_3')
MASKS_PATH         = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation_results_1')
LABELS_PATH        = os.path.join(GLOBAL_PATH, 'DATASET', 'labels_reversed')
CSV_SEG_PATH       = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_1.csv')

RBF_DST_PATH1 = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','RBF_plot')

if not os.path.exists(RBF_DST_PATH1):
    os.makedirs(RBF_DST_PATH1)


best_subjects = ['L_06','L_15','L_26','L_30','L_42','L_48','L_63','L_78','S_08','S_46']

MAX_ALLOWED_MOVEMENT = 50


#--------------------------------------------------------------------------------------------------------------------------------#


def clahe(img,clipLimit=2,gridSize=2):

    clahe = cv.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridSize,gridSize))
    clahe_img = clahe.apply(img)
    return clahe_img


def create_subject_data(fname: str) -> dict:
    
    subject_dict = dict()

    if 'POST' in fname: # ad ogni iterazione processo la coppia PRE-POST
         return None
    
    subject = fname.split('PRE')[0] # esempio: da L_06PRE.JPG estraggo L_06
    phase = 'PRE'
    subject_phase = subject + phase

    # Se il soggetto non è tra i best non ci interessa
    if subject not in best_subjects:
        return None
    
    fname_post = f"{subject}POST.JPG" # estraggo il nome del file POST


    IMG_PATH1 = os.path.join(IR_ALIGNED_PATH,fname)
    IMG_PATH2 = os.path.join(IR_ALIGNED_PATH,fname_post)

    MASK_PATH1 = os.path.join(MASKS_PATH,fname)
    MASK_PATH2 = os.path.join(MASKS_PATH,fname_post)

    LABEL_PATH1 = os.path.join(LABELS_PATH,subject,'total_1.png')
    LABEL_PATH2 = os.path.join(LABELS_PATH,subject,'total_2.png')

    imgPre  = cv.imread(IMG_PATH1,0)
    imgPre = clahe(imgPre,2,4)

    imgPost = cv.imread(IMG_PATH2,0)
    imgPost = clahe(imgPost,2,4)

    maskPre  = cv.imread(MASK_PATH1,0)
    maskPost = cv.imread(MASK_PATH2,0)

    labelPre  = cv.imread(LABEL_PATH1,0)
    labelPost = cv.imread(LABEL_PATH2,0)

    # Controllo se la label è flippata in modo giusto o no
    flipped_cols_csv = pd.read_csv(CSV_SEG_PATH, index_col=False,usecols=['SUBJECT','Flipped'])    
    flipped_subject_state = flipped_cols_csv.loc[flipped_cols_csv['SUBJECT'] == subject_phase,'Flipped'].values[0]

    if flipped_subject_state: # se la label è specchiata rispetto all'immagine
        maskPre = cv.flip(maskPre,1)
        maskPost = cv.flip(maskPost,1)
        labelPre = cv.flip(labelPre,1)
        labelPost = cv.flip(labelPost,1)


    if DRAW_ON == 3:
        fname_colored_superposition = os.path.join(SRC_PATH,f'{subject}_confronto_post.png')
        img_colors = cv.imread(fname_colored_superposition)
    elif DRAW_ON == 1:
        img_colors = cv.cvtColor(imgPre,cv.COLOR_GRAY2BGR)
    else:
        img_colors = cv.cvtColor(labelPre,cv.COLOR_GRAY2BGR)

    subject_dict['ID'] = subject
    subject_dict['PHASE'] = phase
    subject_dict['ID and PHASE'] = subject_phase
    subject_dict['IMG PRE'] = imgPre
    subject_dict['IMG POST'] = imgPost
    subject_dict['MASK PRE'] = maskPre
    subject_dict['MASK POST'] = maskPost
    subject_dict['LABEL PRE'] = labelPre
    subject_dict['LABEL POST'] = labelPost
    subject_dict['DRAW IMG'] = img_colors

    return subject_dict


#--------------------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':

    all_patients_data = []

    for fname in sorted(os.listdir(IR_ALIGNED_PATH)):

        subject_data = create_subject_data(fname)

        if subject_data is None:
             continue

        h, w = subject_data['IMG PRE'].shape
        
        # CALCOLO OPTICAL FLOW
        optical_flow_field = cv.calcOpticalFlowFarneback(prev=subject_data['IMG PRE'], next=subject_data['IMG POST'], flow=None, pyr_scale=0.5, levels=3, winsize=21, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
        
        magnitude = np.sqrt(optical_flow_field[:,:,0]**2 + optical_flow_field[:,:,1]**2)
        mask_magnitude = (magnitude < MAX_ALLOWED_MOVEMENT)
        mask_labelPre = (subject_data['LABEL PRE'] > 0).astype(np.bool)
        
        # Adesso devo creare il vettore di coordinate da passare all'RBF interpolator       
        x = np.arange(w)
        y = np.arange(h)

        xx, yy = np.meshgrid(x, y) # due matrici 1908 x 1908

        xx_valid = xx[(mask_labelPre) & (mask_magnitude)]
        yy_valid = yy[(mask_labelPre) & (mask_magnitude)]

        valid_coordinates = np.column_stack([xx_valid,yy_valid])
        valid_values = optical_flow_field[mask_labelPre & mask_magnitude]

        step = 40
        rbf = RBFInterpolator(valid_coordinates[::step],valid_values[::step],kernel='thin_plate_spline')

        all_coordinates = np.column_stack([xx.flatten(),yy.flatten()]) # calcolo l'interpolazione su tutte le coordinate
        print("Inizio calcolo vector field interpolato")

        #-------------------------------------------------------------------------------------------------------------#

        # Calcolo il campo interpolato a blocchi (chunk) per non saturare la RAM

        chunk_size = 30000
        chunk_result = []
        for i in range(0,len(all_coordinates),chunk_size):
             field_at_chunk = rbf(all_coordinates[i:i+chunk_size])
             chunk_result.append(field_at_chunk)

        interpolated_field = np.vstack(chunk_result).reshape(h, w, 2)

        #-------------------------------------------------------------------------------------------------------------#

        print(f"Analisi per il soggetto {subject_data['ID']} terminata!\n")


        # PLOT DEI RISULTATI 

        magnitude_interpolated = np.sqrt(interpolated_field[:,:,0]**2 + interpolated_field[:,:,1]**2)

        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.pcolormesh(xx, yy, magnitude_interpolated, shading='gouraud', cmap='viridis')
        ax.scatter(valid_coordinates[::50, 0], valid_coordinates[::50, 1], 
           c='white', s=2, alpha=0.3, label='Punti sorgente')
        
        ax.set_title(f"Magnitudo Optical Flow Interpolata - {subject_data['ID']}")
        ax.invert_yaxis() # Le immagini hanno l'origine in alto a sinistra
        fig.colorbar(im, ax=ax, label='Spostamento (pixel)')

        plt.savefig(os.path.join(RBF_DST_PATH1, f"{subject_data['ID']}_RBF_plot.png"), dpi=300)
        plt.close(fig)

        

        

        
        
