

import os
import cv2 as cv
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as patches


#-----------------------------------------------------#


# I miei path
GLOBAL_PATH        = os.getcwd()
IR_ALIGNED_PATH    = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'aligned_imgs_3')
MASKS_PATH         = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'best_10_subjects_segmentation_1')
LABELS_PATH        = os.path.join(GLOBAL_PATH, 'DATASET', 'labels_reversed')
CSV_SEG_PATH       = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW','segmentation_1.csv')
CSV_RBF_FIELD_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW','LK_results','RBF_quadrants_1.csv')

DRAW_OPTIONS = {'IR': 1, 'labels': 2, 'coloured_superposition': 3}
DRAW_ON = DRAW_OPTIONS['IR']

# In base all'opzione definisco un src path da cui prendere le immagini, e un dst path in cui inserire le immagini modificate
if DRAW_ON == 1:
    SRC_PATH = IR_ALIGNED_PATH
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW','LK_results', 'RBF_on_ir_1')

elif DRAW_ON == 2:
    SRC_PATH = LABELS_PATH
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'LK_results', 'RBF_on_labels_1')

else:
    SRC_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','alignment_color_superposition_3')
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'LK_results', 'RBF_on_coloured_superposition_1')

RBF_MATRICES_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','LK_results', 'RBF_matrices_1')

if not os.path.exists(RBF_MATRICES_PATH):
    os.makedirs(RBF_MATRICES_PATH)

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)

best_subjects = ['L_06','L_15','L_26','L_30','L_42','L_48','L_63','L_78','S_08','S_46']

MAX_ALLOWED_MOVEMENT = 40
PIXEL_LENGTH   = 4.651162790697675     # in um
INNER_DIAMETER = 4000/PIXEL_LENGTH     # pixels --> 4 mm = 4000 um / PIXEL_LENGTH
OUTER_DIAMETER = 8000/PIXEL_LENGTH     # pixels --> 8 mm = 8000 um / PIXEL_LENGTH
INNER_RADIUS   = int(INNER_DIAMETER/2) # pixels
OUTER_RADIUS   = int(OUTER_DIAMETER/2) # pixels


# Per ottenere la stessa struttura dei risultati del csv di riferimento
results = {"patient_name":        None,
           "X total":             None,
           "Y total":             None,
           "magnitude total":     None,
           "X inner avg":         None,
           "Y inner avg":         None,
           "magnitude inner avg": None,
           "X outer avg":         None,
           "Y outer avg":         None,
           "magnitude outer avg": None,
           "X 45 inner":          None,
           "X 135 inner":         None,
           "X 225 inner":         None,
           "X 315 inner":         None,
           "X 45 outer":          None,
           "X 135 outer":         None,
           "X 225 outer":         None,
           "X 315 outer":         None,
           "Y 45 inner":          None,
           "Y 135 inner":         None,
           "Y 225 inner":         None,
           "Y 315 inner":         None,
           "Y 45 outer":          None,
           "Y 135 outer":         None,
           "Y 225 outer":         None,
           "Y 315 outer":         None,
           "magnitude 45 inner":  None,
           "magnitude 135 inner": None,
           "magnitude 225 inner": None,
           "magnitude 315 inner": None,
           "magnitude 45 outer":  None,
           "magnitude 135 outer": None,
           "magnitude 225 outer": None,
           "magnitude 315 outer": None}


fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

COMPUTE_RBF_INTERPOLATOR = False
COMPUTE_METRICS = True


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
        img_colors = cv.cvtColor(img_colors,cv.COLOR_BGR2RGB)
        
    elif DRAW_ON == 1:
        img_colors = cv.cvtColor(imgPre,cv.COLOR_GRAY2BGR)
    else:
        img_colors = cv.cvtColor(labelPre,cv.COLOR_GRAY2BGR)

    subject_dict['ID'] = subject
    subject_dict['PHASE'] = phase
    subject_dict['ID and PHASE'] = subject_phase
    subject_dict['IMG PRE'] = imgPre
    subject_dict['IMG POST'] = imgPost
    subject_dict['LABEL PRE'] = labelPre
    subject_dict['LABEL POST'] = labelPost
    subject_dict['DRAW IMG'] = img_colors

    return subject_dict


def safe_mean(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Se non c'è nessun elemento di mask = true su cui calcolare la media allora returna 0
    return np.mean(data[mask]) if np.any(mask) else 0


#-----------------------------------------------------#


if __name__ == '__main__':

    all_patients_data = []

    for fname in sorted(os.listdir(IR_ALIGNED_PATH)):
        
        subject_data = create_subject_data(fname)

        if subject_data is None:
            continue

        h, w = subject_data['IMG PRE'].shape
        cx,cy = fovea_center_dict[subject_data['ID']]

        # CALCOLO OPTICAL FLOW

        step = 5
        x = np.arange(0, w, step, dtype=np.float32)
        y = np.arange(0, h, step, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        pts_x = X.flatten()
        pts_y = Y.flatten()
        idxs_to_keep = subject_data['LABEL PRE'][pts_y.astype(int), pts_x.astype(int)] > 0 # tengo solo i punti che cadono su un vaso della maschera di segmentazione PRE
        prevPts = np.stack((pts_x[idxs_to_keep], pts_y[idxs_to_keep]), axis=-1) # axis = -1 è l'ultimo asse, sono i punti su cui calcolo l'optical flow

        nextPts, status, err =  cv.calcOpticalFlowPyrLK(subject_data['IMG PRE'],
                                subject_data['IMG POST'],
                                prevPts,
                                None,
                                winSize=(31,31),
                                maxLevel=3,
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
                                minEigThreshold=1e-4
                                )
        
        # Filtraggio di controllo a posteriori (doppio controllo)

        good_status = status.flatten() == 1
        next_x_int = np.round(nextPts[:, 0]).astype(int)
        next_y_int = np.round(nextPts[:, 1]).astype(int)
        in_bounds = (next_x_int >= 0) & (next_x_int < w) & (next_y_int >= 0) & (next_y_int < h) # guardo che nessun punto next sia finito fuori dall'immagine

        valid_points = good_status & in_bounds 
        
        lands_on_vessel = np.zeros_like(valid_points, dtype=bool)

        # Controlliamo che il punto di arrivo atterri su un vaso POST
        lands_on_vessel[valid_points] = subject_data['LABEL POST'][next_y_int[valid_points], next_x_int[valid_points]] > 0

        final_mask = valid_points & lands_on_vessel

        final_prevPts = prevPts[final_mask]
        final_nextPts = nextPts[final_mask]
        final_vectors = final_nextPts - final_prevPts # [u, v]

        # Filtraggio in base allo spostamento massimo consentito
        magnitude = np.sqrt(final_vectors[:, 0]**2 + final_vectors[:, 1]**2)
        mask_magnitude = (magnitude < MAX_ALLOWED_MOVEMENT)

        valid_coordinates = final_prevPts[mask_magnitude]
        valid_values = final_vectors[mask_magnitude]

        if COMPUTE_RBF_INTERPOLATOR:

            rbf = RBFInterpolator(valid_coordinates, valid_values, kernel='thin_plate_spline')

            x_dense = np.arange(w)
            y_dense = np.arange(h)
            XX, YY = np.meshgrid(x_dense, y_dense)

            all_coordinates = np.column_stack([XX.flatten(), YY.flatten()])

            print(f"Inizio calcolo vector field interpolato per {subject_data['ID']}")

            chunk_size = 30000
            chunk_result = []
            for i in tqdm(range(0, len(all_coordinates), chunk_size), desc=f"Interpolazione {subject_data['ID']}", leave=False):
                field_at_chunk = rbf(all_coordinates[i:i+chunk_size])
                chunk_result.append(field_at_chunk)

            interpolated_field = np.vstack(chunk_result).reshape(h, w, 2)

            file_name = f"{subject_data['ID']}_interpolated_flow.npy"
            file_path = os.path.join(RBF_MATRICES_PATH, file_name)
            np.save(file_path, interpolated_field)

            print(f"Analisi per il soggetto {subject_data['ID']} terminata!\n")
        

        if COMPUTE_METRICS:

            # 1. Carichiamo la matrice densa generata dall'RBF (dopo Lucas-Kanade)
            interpolated_field = np.load(os.path.join(RBF_MATRICES_PATH,f'{subject_data["ID"]}_interpolated_flow.npy'))
            u = interpolated_field[:,:,0]
            v = interpolated_field[:,:,1]
            magnitude_rbf = np.sqrt(u**2 + v**2)

            # 2. Taglio degli overshoot per evitare rumore estremo
            mask_overshoot = magnitude_rbf > MAX_ALLOWED_MOVEMENT
            scale_factor = MAX_ALLOWED_MOVEMENT / magnitude_rbf[mask_overshoot]
            interpolated_field[mask_overshoot, 0] = u[mask_overshoot] * scale_factor
            interpolated_field[mask_overshoot, 1] = v[mask_overshoot] * scale_factor

            u_final = interpolated_field[:,:,0]
            v_final = interpolated_field[:,:,1]
            magnitude_final = np.sqrt(u_final**2 + v_final**2)

            # 3. Creazione delle maschere per le metriche (Cerchi e Quadranti)
            y_coords, x_coords = np.indices((h,w)) # 1908 x 1908   
            dx = x_coords - cx
            dy = y_coords - cy
            distance = np.sqrt(dx**2 + dy**2)

            mask_inner = (distance <= INNER_RADIUS) 
            mask_outer = (distance > INNER_RADIUS) & (distance <= OUTER_RADIUS) 
            mask_global = mask_inner | mask_outer   

            angle = np.rad2deg(np.atan2(-dy,dx))

            mask_right  = (angle >= -45) & (angle < 45) 
            mask_top    = (angle >= 45) & (angle < 135) 
            mask_left   = ((angle >= 135) | (angle < -135)) 
            mask_bottom = (angle >= -135) & (angle < -45) 

            # 4. Calcolo vettoriale delle medie (niente più cicli for!)
            X_total = safe_mean(u_final, mask_global)        
            Y_total = safe_mean(v_final, mask_global)
            magnitude_total = safe_mean(magnitude_final, mask_global)
            
            X_inner_avg = safe_mean(u_final, mask_inner)
            Y_inner_avg = safe_mean(v_final, mask_inner)
            magnitude_inner_avg = safe_mean(magnitude_final, mask_inner) 

            X_outer_avg = safe_mean(u_final, mask_outer)
            Y_outer_avg = safe_mean(v_final, mask_outer)
            magnitude_outer_avg = safe_mean(magnitude_final, mask_outer)

            X_inner_45  = safe_mean(u_final, mask_inner & mask_bottom)
            X_inner_135 = safe_mean(u_final, mask_inner & mask_left)
            X_inner_225 = safe_mean(u_final, mask_inner & mask_top)
            X_inner_315 = safe_mean(u_final, mask_inner & mask_right)

            Y_inner_45  = safe_mean(v_final, mask_inner & mask_bottom)
            Y_inner_135 = safe_mean(v_final, mask_inner & mask_left)
            Y_inner_225 = safe_mean(v_final, mask_inner & mask_top)
            Y_inner_315 = safe_mean(v_final, mask_inner & mask_right)

            magnitude_inner_45  = safe_mean(magnitude_final, mask_inner & mask_bottom)
            magnitude_inner_135 = safe_mean(magnitude_final, mask_inner & mask_left)
            magnitude_inner_225 = safe_mean(magnitude_final, mask_inner & mask_top)
            magnitude_inner_315 = safe_mean(magnitude_final, mask_inner & mask_right)

            X_outer_45  = safe_mean(u_final, mask_outer & mask_bottom)
            X_outer_135 = safe_mean(u_final, mask_outer & mask_left)
            X_outer_225 = safe_mean(u_final, mask_outer & mask_top)
            X_outer_315 = safe_mean(u_final, mask_outer & mask_right)

            Y_outer_45  = safe_mean(v_final, mask_outer & mask_bottom)
            Y_outer_135 = safe_mean(v_final, mask_outer & mask_left)
            Y_outer_225 = safe_mean(v_final, mask_outer & mask_top)
            Y_outer_315 = safe_mean(v_final, mask_outer & mask_right)

            magnitude_outer_45  = safe_mean(magnitude_final, mask_outer & mask_bottom)
            magnitude_outer_135 = safe_mean(magnitude_final, mask_outer & mask_left)
            magnitude_outer_225 = safe_mean(magnitude_final, mask_outer & mask_top)
            magnitude_outer_315 = safe_mean(magnitude_final, mask_outer & mask_right)

            # 5. Salvataggio dei risultati nel dizionario
            current_results = results.copy()
            current_results['patient_name'] = subject_data['ID']

            current_results["X total"]         = X_total * PIXEL_LENGTH
            current_results["Y total"]         = Y_total * PIXEL_LENGTH
            current_results['magnitude total'] = magnitude_total * PIXEL_LENGTH

            current_results["X inner avg"]         = X_inner_avg * PIXEL_LENGTH
            current_results["Y inner avg"]         = Y_inner_avg * PIXEL_LENGTH
            current_results["magnitude inner avg"] = magnitude_inner_avg * PIXEL_LENGTH

            current_results["X outer avg"]         = X_outer_avg * PIXEL_LENGTH
            current_results["Y outer avg"]         = Y_outer_avg * PIXEL_LENGTH
            current_results["magnitude outer avg"] = magnitude_outer_avg * PIXEL_LENGTH

            current_results["X 45 inner"]  = X_inner_45 * PIXEL_LENGTH
            current_results["X 135 inner"] = X_inner_135 * PIXEL_LENGTH
            current_results["X 225 inner"] = X_inner_225 * PIXEL_LENGTH
            current_results["X 315 inner"] = X_inner_315 * PIXEL_LENGTH
            current_results["Y 45 inner"]  = Y_inner_45 * PIXEL_LENGTH
            current_results["Y 135 inner"] = Y_inner_135 * PIXEL_LENGTH
            current_results["Y 225 inner"] = Y_inner_225 * PIXEL_LENGTH
            current_results["Y 315 inner"] = Y_inner_315 * PIXEL_LENGTH

            current_results["X 45 outer"]  = X_outer_45 * PIXEL_LENGTH
            current_results["X 135 outer"] = X_outer_135 * PIXEL_LENGTH
            current_results["X 225 outer"] = X_outer_225 * PIXEL_LENGTH
            current_results["X 315 outer"] = X_outer_315 * PIXEL_LENGTH
            current_results["Y 45 outer"]  = Y_outer_45 * PIXEL_LENGTH
            current_results["Y 135 outer"] = Y_outer_135 * PIXEL_LENGTH
            current_results["Y 225 outer"] = Y_outer_225 * PIXEL_LENGTH
            current_results["Y 315 outer"] = Y_outer_315 * PIXEL_LENGTH

            current_results["magnitude 45 inner"]  = magnitude_inner_45 * PIXEL_LENGTH
            current_results["magnitude 135 inner"] = magnitude_inner_135 * PIXEL_LENGTH
            current_results["magnitude 225 inner"] = magnitude_inner_225 * PIXEL_LENGTH
            current_results["magnitude 315 inner"] = magnitude_inner_315 * PIXEL_LENGTH

            current_results["magnitude 45 outer"]  = magnitude_outer_45 * PIXEL_LENGTH
            current_results["magnitude 135 outer"] = magnitude_outer_135 * PIXEL_LENGTH
            current_results["magnitude 225 outer"] = magnitude_outer_225 * PIXEL_LENGTH
            current_results["magnitude 315 outer"] = magnitude_outer_315 * PIXEL_LENGTH

            all_patients_data.append(current_results)

            # ---------------------------------- PLOT DEI RISULTATI ---------------------------------- #

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(subject_data['DRAW IMG'])
            grid_color = 'white'
            thickness = 1.5
            linestyle = '--'
            
            # Disegno cerchi e diagonali
            inner_circle = patches.Circle((cx, cy), INNER_RADIUS, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle)
            outer_circle = patches.Circle((cx, cy), OUTER_RADIUS, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle)
            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)
            offset = OUTER_RADIUS * 0.707
            ax.plot([cx - offset, cx + offset], [cy - offset, cy + offset], color=grid_color, linewidth=thickness, linestyle=linestyle)
            ax.plot([cx - offset, cx + offset], [cy + offset, cy - offset], color=grid_color, linewidth=thickness, linestyle=linestyle)

            ax.plot(cx, cy, marker='+', color=grid_color, markersize=15, markeredgewidth=thickness)
            
            # Impostazione dinamica di scala e colore in base allo sfondo
            if DRAW_ON == 1:
                q_color, q_scale = 'lime', 500
            elif DRAW_ON == 2:
                q_color, q_scale = 'lime', 500
            else:
                q_color, q_scale = 'mediumblue', 500

            # Maschera per non stampare vettori sullo sfondo bianco
            retina_mask = subject_data['IMG PRE'] < 250

            u_plot = u_final.copy()
            v_plot = v_final.copy()

            u_plot[~retina_mask] = np.nan
            v_plot[~retina_mask] = np.nan

            # Plot del campo vettoriale principale usando Matplotlib Quiver
            step_q = 30
            ax.quiver(x_coords[::step_q,::step_q], y_coords[::step_q,::step_q], 
                      u_plot[::step_q,::step_q], v_plot[::step_q,::step_q], 
                      angles='xy', color=q_color, alpha=0.8, scale=q_scale, width=0.001, headwidth=3, headlength=4, headaxislength=3.5)

            # Creazione della freccia di riferimento (Scale Bar)
            ref_length_um = 50 
            ref_length_px = ref_length_um / PIXEL_LENGTH
            
            x_ref = w - 300
            y_ref = h - 200

            ax.quiver(x_ref, y_ref, ref_length_px, 0, 
                      angles='xy', color='white', scale=q_scale, width=0.003, headwidth=4)
            
            ax.text(x_ref + (ref_length_px/2), y_ref + 80, f'{ref_length_um} $\mu m$', 
                color='white', fontsize=18, fontweight='bold', family='serif', ha='center', va='top')

            ax.axis('off')
            plt.tight_layout()
            
            # Salvataggio delle immagini
            plt.savefig(os.path.join(DST_PATH, f"{subject_data['ID']}_RBF_plot.png"), dpi=600, bbox_inches='tight', pad_inches=0)
            plt.savefig(os.path.join(DST_PATH, f"{subject_data['ID']}_RBF_plot.pdf"), dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Aggiornamento CSV
            df = pd.DataFrame(all_patients_data)
            df.to_csv(CSV_RBF_FIELD_PATH, index=False)

