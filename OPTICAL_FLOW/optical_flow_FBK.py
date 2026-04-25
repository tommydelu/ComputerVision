
import os
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


#--------------------------------------------------------------------------------------------------------------------------------#


# I miei path
GLOBAL_PATH        = os.getcwd()
IR_ALIGNED_PATH    = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'aligned_imgs_3')
MASKS_PATH         = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation_results_1')
LABELS_PATH        = os.path.join(GLOBAL_PATH, 'DATASET', 'labels_reversed')
CSV_SEG_PATH       = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_1.csv')
CSV_FLOW_PATH      = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','FBK_optical_flow_results_1.csv')


# Il campo vettoriale si può disegnare su 3 tipi di immagine diverse
DRAW_OPTIONS = {'IR': 1, 'labels': 2, 'coloured_superposition': 3}
DRAW_ON = DRAW_OPTIONS['IR']

# In base all'opzione definisco un src path da cui prendere le immagini, e un dst path in cui inserire le immagini modificate
if DRAW_ON == 1:
    SRC_PATH = IR_ALIGNED_PATH
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'FBK_field_on_IR_1')

elif DRAW_ON == 2:
    SRC_PATH = LABELS_PATH
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'FBK_field_on_labels_1')

else:
    SRC_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','alignment_color_superposition_3')
    DST_PATH = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'FBK_field_on_coloured_superposition_1')


RBF_DST_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','RBF_matrices_prova1')

if not os.path.exists(RBF_DST_PATH):
    os.makedirs(RBF_DST_PATH)





best_subjects = ['L_06','L_15','L_26','L_30','L_42','L_48','L_63','L_78','S_08','S_46']

fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

PIXEL_LENGTH = 4.651162790697675 # in um

INNER_DIAMETER = 4000/PIXEL_LENGTH  # pixels --> 4 mm = 4000 um / PIXEL_LENGTH
OUTER_DIAMETER = 8000/PIXEL_LENGTH  # pixels --> 8 mm = 8000 um / PIXEL_LENGTH

INNER_RADIUS = int(INNER_DIAMETER/2) # pixels
OUTER_RADIUS = int(OUTER_DIAMETER/2) # pixels

MAX_ALLOWED_MOVEMENT = 50

# Per ottenere la stessa struttura dei risultati del csv di riferimento
# results = {"patient_name":        None,
#            "X total":             None,
#            "Y total":             None,
#            "magnitude total":     None,
#            "X inner avg":         None,
#            "Y inner avg":         None,
#            "magnitude inner avg": None,
#            "X outer avg":         None,
#            "Y outer avg":         None,
#            "magnitude outer avg": None,
#            "X 45 inner":          None,
#            "X 135 inner":         None,
#            "X 225 inner":         None,
#            "X 315 inner":         None,
#            "X 45 outer":          None,
#            "X 135 outer":         None,
#            "X 225 outer":         None,
#            "X 315 outer":         None,
#            "Y 45 inner":          None,
#            "Y 135 inner":         None,
#            "Y 225 inner":         None,
#            "Y 315 inner":         None,
#            "Y 45 outer":          None,
#            "Y 135 outer":         None,
#            "Y 225 outer":         None,
#            "Y 315 outer":         None,
#            "magnitude 45 inner":  None,
#            "magnitude 135 inner": None,
#            "magnitude 225 inner": None,
#            "magnitude 315 inner": None,
#            "magnitude 45 outer":  None,
#            "magnitude 135 outer": None,
#            "magnitude 225 outer": None,
#            "magnitude 315 outer": None}


#--------------------------------------------------------------------------------------------------------------------------------#


def safe_mean(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Se non c'è nessun elemento di mask = true su cui calcolare la media allora returna 0
    return np.mean(data[mask]) if np.any(mask) else 0


def draw_vector_field(u:np.ndarray, v:np.ndarray, img_to_draw:np.ndarray) -> None:

        step = 8
        u_ = u[::step, ::step]
        v_ = v[::step, ::step]
        y_coords, x_coords = np.indices((h,w)) # dimensions = shape of the grid (1908 x 1908)   
        x = x_coords[::step,::step]
        y = y_coords[::step,::step]

        u_vect = u_.flatten()
        v_vect = v_.flatten()
        u_vect = np.nan_to_num(u_vect)
        v_vect = np.nan_to_num(v_vect)
        x_vect = x.flatten()
        y_vect = y.flatten()

        scale_factor = 3

        for i in range(len(x_vect)):
            if labelPre[y_vect[i], x_vect[i]] > 0:
                    start_pt = (int(x_vect[i]), int(y_vect[i]))
                    end_pt = (int(x_vect[i] + u_vect[i] * scale_factor), 
                            int(y_vect[i] + v_vect[i] * scale_factor))
                    
                    # Disegno 
                    cv.arrowedLine(img_colors, start_pt, end_pt, (0, 255, 0), 1, tipLength=0.3)
                    cv.circle(img_colors, end_pt, 2, (0, 0, 255), -1)


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
        cx,cy = fovea_center_dict[subject_data['ID']]
        
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

        output_npy = os.path.join(RBF_DST_PATH, f"{subject_data['ID']}_rbf_array.npy")
        np.save(output_npy, interpolated_field)

        #-------------------------------------------------------------------------------------------------------------#

        print(f"Analisi per il soggetto {subject_data['ID']} terminata!\n")









        
    

       



       

       


       

        





        









        # # Adesso voglio capire per ogni coordinata se si trova nel cerchio inner o nel cerchio outer e in che quadrante
        # dx = x_coords - cx
        # dy = y_coords - cy
        # distance = np.sqrt(dx**2 + dy**2)

        # mask_inner = (distance <= INNER_RADIUS) & (labelPre > 0)
        # mask_outer = (distance > INNER_RADIUS) & (distance <= OUTER_RADIUS) & (labelPre > 0)
        # mask_global = mask_inner | mask_outer   

        # angle = np.rad2deg(np.atan2(-dy,dx))

        # mask_right  = (angle >= -45) & (angle < 45) & (labelPre > 0)
        # mask_top    = (angle >= 45) & (angle < 135) & (labelPre > 0)
        # mask_left   = ((angle >= 135) | (angle < -135)) & (labelPre > 0)
        # mask_bottom = (angle >= -135) & (angle < -45) & (labelPre > 0)


        # CALCOLO DI TUTTE LE METRICHE

        # X_total = safe_mean(u, mask_global)        
        # Y_total = safe_mean(v, mask_global)
        # magnitude_total = safe_mean(magnitude, mask_global)
        
        # X_inner_avg = safe_mean(u, mask_inner)
        # Y_inner_avg = safe_mean(v, mask_inner)
        # magnitude_inner_avg = safe_mean(magnitude, mask_inner) 

        # X_outer_avg = safe_mean(u, mask_outer)
        # Y_outer_avg = safe_mean(v, mask_outer)
        # magnitude_outer_avg = safe_mean(magnitude, mask_outer)

        # X_inner_45  = safe_mean(u, mask_inner & mask_bottom)
        # X_inner_135 = safe_mean(u, mask_inner & mask_left)
        # X_inner_225 = safe_mean(u, mask_inner & mask_top)
        # X_inner_315 = safe_mean(u, mask_inner & mask_right)

        # Y_inner_45  = safe_mean(v, mask_inner & mask_bottom)
        # Y_inner_135 = safe_mean(v, mask_inner & mask_left)
        # Y_inner_225 = safe_mean(v, mask_inner & mask_top)
        # Y_inner_315 = safe_mean(v, mask_inner & mask_right)

        # magnitude_inner_45  = safe_mean(magnitude, mask_inner & mask_bottom)
        # magnitude_inner_135 = safe_mean(magnitude, mask_inner & mask_left)
        # magnitude_inner_225 = safe_mean(magnitude, mask_inner & mask_top)
        # magnitude_inner_315 = safe_mean(magnitude, mask_inner & mask_right)

        # OUTER RING
        # X_outer_45  = safe_mean(u, mask_outer & mask_bottom)
        # X_outer_135 = safe_mean(u, mask_outer & mask_left)
        # X_outer_225 = safe_mean(u, mask_outer & mask_top)
        # X_outer_315 = safe_mean(u, mask_outer & mask_right)

        # Y_outer_45  = safe_mean(v, mask_outer & mask_bottom)
        # Y_outer_135 = safe_mean(v, mask_outer & mask_left)
        # Y_outer_225 = safe_mean(v, mask_outer & mask_top)
        # Y_outer_315 = safe_mean(v, mask_outer & mask_right)

        # magnitude_outer_45  = safe_mean(magnitude, mask_outer & mask_bottom)
        # magnitude_outer_135 = safe_mean(magnitude, mask_outer & mask_left)
        # magnitude_outer_225 = safe_mean(magnitude, mask_outer & mask_top)
        # magnitude_outer_315 = safe_mean(magnitude, mask_outer & mask_right)

        # STORING DEI RISULTATI

        # current_results = results.copy()
        # current_results['patient_name'] = subject

        # current_results["X total"]         = X_total * PIXEL_LENGTH
        # current_results["Y total"]         = Y_total * PIXEL_LENGTH
        # current_results['magnitude total'] = magnitude_total * PIXEL_LENGTH

        # current_results["X inner avg"]         = X_inner_avg * PIXEL_LENGTH
        # current_results["Y inner avg"]         = Y_inner_avg * PIXEL_LENGTH
        # current_results["magnitude inner avg"] = magnitude_inner_avg * PIXEL_LENGTH

        # current_results["X outer avg"]         = X_outer_avg * PIXEL_LENGTH
        # current_results["Y outer avg"]         = Y_outer_avg * PIXEL_LENGTH
        # current_results["magnitude outer avg"] = magnitude_outer_avg * PIXEL_LENGTH

        # current_results["X 45 inner"]  = X_inner_45 * PIXEL_LENGTH
        # current_results["X 135 inner"] = X_inner_135 * PIXEL_LENGTH
        # current_results["X 225 inner"] = X_inner_225 * PIXEL_LENGTH
        # current_results["X 315 inner"] = X_inner_315 * PIXEL_LENGTH
        # current_results["Y 45 inner"]  = Y_inner_45 * PIXEL_LENGTH
        # current_results["Y 135 inner"] = Y_inner_135 * PIXEL_LENGTH
        # current_results["Y 225 inner"] = Y_inner_225 * PIXEL_LENGTH
        # current_results["Y 315 inner"] = Y_inner_315 * PIXEL_LENGTH

        # current_results["X 45 outer"]  = X_outer_45 * PIXEL_LENGTH
        # current_results["X 135 outer"] = X_outer_135 * PIXEL_LENGTH
        # current_results["X 225 outer"] = X_outer_225 * PIXEL_LENGTH
        # current_results["X 315 outer"] = X_outer_315 * PIXEL_LENGTH
        # current_results["Y 45 outer"]  = Y_outer_45 * PIXEL_LENGTH
        # current_results["Y 135 outer"] = Y_outer_135 * PIXEL_LENGTH
        # current_results["Y 225 outer"] = Y_outer_225 * PIXEL_LENGTH
        # current_results["Y 315 outer"] = Y_outer_315 * PIXEL_LENGTH

        # current_results["magnitude 45 inner"]  = magnitude_inner_45 * PIXEL_LENGTH
        # current_results["magnitude 135 inner"] = magnitude_inner_135 * PIXEL_LENGTH
        # current_results["magnitude 225 inner"] = magnitude_inner_225 * PIXEL_LENGTH
        # current_results["magnitude 315 inner"] = magnitude_inner_315 * PIXEL_LENGTH

        # current_results["magnitude 45 outer"]  = magnitude_outer_45 * PIXEL_LENGTH
        # current_results["magnitude 135 outer"] = magnitude_outer_135 * PIXEL_LENGTH
        # current_results["magnitude 225 outer"] = magnitude_outer_225 * PIXEL_LENGTH
        # current_results["magnitude 315 outer"] = magnitude_outer_315 * PIXEL_LENGTH

        # all_patients_data.append(current_results)


        # grid_color = (255, 255, 255) # Bianco
        # thickness = 2
        # cv.circle(img_colors, (cx, cy), INNER_RADIUS, grid_color, thickness)
        # cv.circle(img_colors, (cx, cy), OUTER_RADIUS, grid_color, thickness)
        # offset = int(OUTER_RADIUS * 0.707)
        # cv.line(img_colors, (cx - offset, cy - offset), (cx + offset, cy + offset), grid_color, thickness)
        # cv.line(img_colors, (cx - offset, cy + offset), (cx + offset, cy - offset), grid_color, thickness)
        # cv.drawMarker(img_colors, (cx, cy), grid_color, cv.MARKER_CROSS, 20, 2)

        # cv.imwrite(os.path.join(DST_PATH,f'{fname}_flow.jpg'),img_colors)


    # df = pd.DataFrame(all_patients_data)
    # df.to_csv(CSV_PATH, index=False)  

        

        

        
        
