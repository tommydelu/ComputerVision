import os
import cv2 as cv
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as patches


from OPTICAL_FLOW.utils.classes import ConfigPaths
from OPTICAL_FLOW.segmentation.src.utils.extract_best_subjects import extract_best_subjects
from OPTICAL_FLOW.utils.config import *
from OPTICAL_FLOW.utils.functions import create_subject_data, safe_mean


config = ConfigPaths(mode="lukas_kannade",draw_option='IR')

best_subjects = sorted(extract_best_subjects(config.CSV_SEG_PATH, save_images=False))

COMPUTE_RBF_INTERPOLATOR = False


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




if __name__ == '__main__':

    all_patients_data = []

    for fname in sorted(os.listdir(config.SRC_PATH)):
        
        subject_data = create_subject_data(fname, best_subjects, config.SRC_PATH, config.MASKS_PATH, config.LABELS_PATH, config.CSV_SEG_PATH, config.DRAW_SRC_PATH, config.draw_option)

        if subject_data is None:
            continue

        h, w = subject_data['IMG PRE'].shape
        cx,cy = fovea_center_dict[subject_data['ID']]

        # CALCOLO OPTICAL FLOW


        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """        QUI SCELGO I PREVPTS COME LISTA DI COORDINATE SU CUI CALCOLARE L'OPTICAL FLOW.            """
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        # step = 5
        # x = np.arange(0, w, step, dtype=np.float32)
        # y = np.arange(0, h, step, dtype=np.float32)
        # X, Y = np.meshgrid(x, y)
        # pts_x = X.flatten()
        # pts_y = Y.flatten()
        # idxs_to_keep = subject_data['LABEL PRE'][pts_y.astype(int), pts_x.astype(int)] > 0 # tengo solo i punti che cadono su un vaso della maschera di segmentazione PRE
        # prevPts = np.stack((pts_x[idxs_to_keep], pts_y[idxs_to_keep]), axis=-1) # axis = -1 è l'ultimo asse, sono i punti su cui calcolo l'optical flow

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """             QUI SCELGO I PREVPTS COME FEATURE ESTRATTE CON GOODFEATURESTOTRACK.                  """
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        # prevPts = cv.goodFeaturesToTrack(
        #     image=subject_data['IMG PRE'],
        #     maxCorners=15000,         
        #     qualityLevel=0.01,        
        #     minDistance=5,            
        #     mask=subject_data['LABEL PRE'], 
        #     blockSize=3,              
        #     useHarrisDetector=True
        # )

        # if prevPts is None:
        #     print(f"Nessun punto trovato per {subject_data['ID']}")
        #     continue


        # Con questo blocco di codice salvo la posizione delle feature estratte da Shi-Tomasi per poterle visualizzare in un secondo momento (debug)

        # debug_img = cv.cvtColor(subject_data['IMG PRE'], cv.COLOR_GRAY2BGR)
        
        # for i in range(len(prevPts)):
        #     x, y = prevPts[i].ravel()
        #     cv.circle(debug_img, (int(x), int(y)), 2, (0, 0, 255), -1)
        # debug_path = os.path.join(config.GLOBAL_PATH, 'OPTICAL_FLOW','lukas_kannade','results','figures','featuresPosition','harris',f"{subject_data['ID']}.jpg")
        # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        # cv.imwrite(debug_path, debug_img)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """             QUI SCELGO I PREVPTS COME FEATURE ESTRATTE CON SIFT/ORB                              """
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        orb = cv.ORB_create()
        sift = cv.SIFT_create()
        keypoints_pre, descriptors_pre = sift.detectAndCompute(subject_data['IMG PRE'], mask=subject_data['LABEL PRE'])
        keypoints_post, descriptors_post = sift.detectAndCompute(subject_data['IMG POST'], mask=subject_data['LABEL POST'])
        # create BFMatcher object
        # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # # Match descriptors.
        # matches = bf.match(descriptors_pre, descriptors_post)
        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(descriptors_pre, descriptors_post, k=2)

        # # Sort them in the order of their distance.
        # # matches = sorted(matches, key = lambda x:x.distance)
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append([m])

        # # img3 = cv.drawMatches(subject_data['IMG PRE'], keypoints_pre, subject_data['IMG POST'], keypoints_post, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # img3 = cv.drawMatchesKnn(subject_data['IMG PRE'], keypoints_pre, subject_data['IMG POST'], keypoints_post, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(descriptors_pre,descriptors_post,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
        
        img3 = cv.drawMatchesKnn(subject_data['IMG PRE'], keypoints_pre, subject_data['IMG POST'], keypoints_post, matches, None, **draw_params)

        match_path = os.path.join(config.GLOBAL_PATH, 'OPTICAL_FLOW','lukas_kannade','results','figures','featuresPosition','sift+flann',f"{subject_data['ID']}.jpg")
        os.makedirs(os.path.dirname(match_path), exist_ok=True)
        cv.imwrite(match_path, img3)


        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        






        # nextPts, status, err =  cv.calcOpticalFlowPyrLK(subject_data['IMG PRE'],
        #                         subject_data['IMG POST'],
        #                         prevPts,
        #                         None,
        #                         winSize=(31,31),
        #                         maxLevel=3,
        #                         criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        #                         minEigThreshold=1e-4
        #                         )
        

        
        # prevPts = prevPts.reshape(-1, 2)
        # nextPts = nextPts.reshape(-1, 2)


        # good_status = status.flatten() == 1


        # matching_img = cv.cvtColor(subject_data['IMG PRE'], cv.COLOR_GRAY2BGR)

        # # Filtriamo solo i punti che Lucas-Kanade ha tracciato con successo
        # good_prev = prevPts[good_status]
        # good_next = nextPts[good_status]

        # # TRUCCO CLINICO: I movimenti retinici sono di pochi pixel (a volte 2 o 3). 
        # # Una freccia così piccola non si vede ad occhio nudo sull'immagine intera. 
        # # Moltiplichiamo il vettore visivo per 3 per renderlo evidente (modifica solo il disegno, non i dati veri)
        # visual_scale_factor = 0.2

        # for old, new in zip(good_prev, good_next):
        #     # Coordinate di partenza (PRE)
        #     x0, y0 = int(old[0]), int(old[1])
            
        #     # Calcolo del vettore reale e della destinazione "scalata" per il disegno
        #     u_vis = (new[0] - old[0]) * visual_scale_factor
        #     v_vis = (new[1] - old[1]) * visual_scale_factor
        #     x1, y1 = int(old[0] + u_vis), int(old[1] + v_vis)

        #     # 1. Disegna il punto di partenza (Pallino Rosso)
        #     cv.circle(matching_img, (x0, y0), 2, (0, 0, 255), -1)
            
        #     # 2. Disegna la freccia dello spostamento (Freccia Verde fluo)
        #     # Solo se c'è un minimo di movimento, altrimenti saltiamo la freccia
        #     if abs(u_vis) > 1 or abs(v_vis) > 1:
        #         cv.arrowedLine(matching_img, (x0, y0), (x1, y1), (0, 255, 0), 1, tipLength=0.3)

        # # Salvataggio
        # match_path = os.path.join(config.GLOBAL_PATH, 'OPTICAL_FLOW','lukas_kannade','results','figures','featureMatching',f"{subject_data['ID']}_Feature_Matching.jpg")
        # cv.imwrite(match_path, matching_img)
        # print(f"Salvataggio immagine feature matching in: {match_path}")





        
        # # Filtraggio di controllo a posteriori (doppio controllo)

        # next_x_int = np.round(nextPts[:, 0]).astype(int)
        # next_y_int = np.round(nextPts[:, 1]).astype(int)
        # in_bounds = (next_x_int >= 0) & (next_x_int < w) & (next_y_int >= 0) & (next_y_int < h) # guardo che nessun punto next sia finito fuori dall'immagine

        # valid_points = good_status & in_bounds 
        # kernel = np.ones((5,5), np.uint8)
        # dilated_label_post = cv.dilate(subject_data['LABEL POST'], kernel, iterations=1)
        
        # lands_on_vessel = np.zeros_like(valid_points, dtype=bool)

        # # Controlliamo che il punto di arrivo atterri su un vaso POST
        # lands_on_vessel[valid_points] = dilated_label_post[next_y_int[valid_points], next_x_int[valid_points]] > 0

        # final_mask = valid_points & lands_on_vessel

        # final_prevPts = prevPts[final_mask]
        # final_nextPts = nextPts[final_mask]
        # final_vectors = final_nextPts - final_prevPts # [u, v]

        # # Filtraggio in base allo spostamento massimo consentito
        # magnitude = np.sqrt(final_vectors[:, 0]**2 + final_vectors[:, 1]**2)
        # mask_magnitude = (magnitude < MAX_ALLOWED_MOVEMENT)

        # valid_coordinates = final_prevPts[mask_magnitude]
        # valid_values = final_vectors[mask_magnitude]

        # if valid_coordinates.shape[0] == 0:
        #     continue


        # if COMPUTE_RBF_INTERPOLATOR:

        #     rbf = RBFInterpolator(valid_coordinates, valid_values, kernel='thin_plate_spline')

        #     x_dense = np.arange(w)
        #     y_dense = np.arange(h)
        #     XX, YY = np.meshgrid(x_dense, y_dense)

        #     all_coordinates = np.column_stack([XX.flatten(), YY.flatten()])

        #     print(f"Inizio calcolo vector field interpolato per {subject_data['ID']}")

        #     chunk_size = 30000
        #     chunk_result = []
        #     for i in tqdm(range(0, len(all_coordinates), chunk_size), desc=f"Interpolazione {subject_data['ID']}", leave=False):
        #         field_at_chunk = rbf(all_coordinates[i:i+chunk_size])
        #         chunk_result.append(field_at_chunk)

        #     interpolated_field = np.vstack(chunk_result).reshape(h, w, 2)

        #     file_name = f"{subject_data['ID']}_interpolated_flow.npy"
        #     file_path = os.path.join(config.RBF_MATRICES_PATH, file_name)
        #     np.save(file_path, interpolated_field)

        #     print(f"Analisi per il soggetto {subject_data['ID']} terminata!\n")
        

        # if not COMPUTE_RBF_INTERPOLATOR:

        #     # 1. Carichiamo la matrice densa generata dall'RBF (dopo Lucas-Kanade)
        #     interpolated_field = np.load(os.path.join(config.RBF_MATRICES_PATH,f'{subject_data["ID"]}_interpolated_flow.npy'))
        #     u = interpolated_field[:,:,0]
        #     v = interpolated_field[:,:,1]
        #     magnitude_rbf = np.sqrt(u**2 + v**2)

        #     # 2. Taglio degli overshoot per evitare rumore estremo
        #     mask_overshoot = magnitude_rbf > MAX_ALLOWED_MOVEMENT
        #     scale_factor = MAX_ALLOWED_MOVEMENT / magnitude_rbf[mask_overshoot]
        #     interpolated_field[mask_overshoot, 0] = u[mask_overshoot] * scale_factor
        #     interpolated_field[mask_overshoot, 1] = v[mask_overshoot] * scale_factor

        #     u_final = interpolated_field[:,:,0]
        #     v_final = interpolated_field[:,:,1]
        #     magnitude_final = np.sqrt(u_final**2 + v_final**2)

        #     # 3. Creazione delle maschere per le metriche (Cerchi e Quadranti)
        #     y_coords, x_coords = np.indices((h,w)) # 1908 x 1908   
        #     dx = x_coords - cx
        #     dy = y_coords - cy
        #     distance = np.sqrt(dx**2 + dy**2)

        #     mask_inner = (distance <= INNER_RADIUS) 
        #     mask_outer = (distance > INNER_RADIUS) & (distance <= OUTER_RADIUS) 
        #     mask_global = mask_inner | mask_outer   

        #     angle = np.rad2deg(np.atan2(-dy,dx))

        #     mask_right  = (angle >= -45) & (angle < 45) 
        #     mask_top    = (angle >= 45) & (angle < 135) 
        #     mask_left   = ((angle >= 135) | (angle < -135)) 
        #     mask_bottom = (angle >= -135) & (angle < -45) 

        #     # 4. Calcolo vettoriale delle medie (niente più cicli for!)
        #     X_total = safe_mean(u_final, mask_global)        
        #     Y_total = safe_mean(v_final, mask_global)
        #     magnitude_total = safe_mean(magnitude_final, mask_global)
            
        #     X_inner_avg = safe_mean(u_final, mask_inner)
        #     Y_inner_avg = safe_mean(v_final, mask_inner)
        #     magnitude_inner_avg = safe_mean(magnitude_final, mask_inner) 

        #     X_outer_avg = safe_mean(u_final, mask_outer)
        #     Y_outer_avg = safe_mean(v_final, mask_outer)
        #     magnitude_outer_avg = safe_mean(magnitude_final, mask_outer)

        #     X_inner_45  = safe_mean(u_final, mask_inner & mask_bottom)
        #     X_inner_135 = safe_mean(u_final, mask_inner & mask_left)
        #     X_inner_225 = safe_mean(u_final, mask_inner & mask_top)
        #     X_inner_315 = safe_mean(u_final, mask_inner & mask_right)

        #     Y_inner_45  = safe_mean(v_final, mask_inner & mask_bottom)
        #     Y_inner_135 = safe_mean(v_final, mask_inner & mask_left)
        #     Y_inner_225 = safe_mean(v_final, mask_inner & mask_top)
        #     Y_inner_315 = safe_mean(v_final, mask_inner & mask_right)

        #     magnitude_inner_45  = safe_mean(magnitude_final, mask_inner & mask_bottom)
        #     magnitude_inner_135 = safe_mean(magnitude_final, mask_inner & mask_left)
        #     magnitude_inner_225 = safe_mean(magnitude_final, mask_inner & mask_top)
        #     magnitude_inner_315 = safe_mean(magnitude_final, mask_inner & mask_right)

        #     X_outer_45  = safe_mean(u_final, mask_outer & mask_bottom)
        #     X_outer_135 = safe_mean(u_final, mask_outer & mask_left)
        #     X_outer_225 = safe_mean(u_final, mask_outer & mask_top)
        #     X_outer_315 = safe_mean(u_final, mask_outer & mask_right)

        #     Y_outer_45  = safe_mean(v_final, mask_outer & mask_bottom)
        #     Y_outer_135 = safe_mean(v_final, mask_outer & mask_left)
        #     Y_outer_225 = safe_mean(v_final, mask_outer & mask_top)
        #     Y_outer_315 = safe_mean(v_final, mask_outer & mask_right)

        #     magnitude_outer_45  = safe_mean(magnitude_final, mask_outer & mask_bottom)
        #     magnitude_outer_135 = safe_mean(magnitude_final, mask_outer & mask_left)
        #     magnitude_outer_225 = safe_mean(magnitude_final, mask_outer & mask_top)
        #     magnitude_outer_315 = safe_mean(magnitude_final, mask_outer & mask_right)

        #     # 5. Salvataggio dei risultati nel dizionario
        #     current_results = results.copy()
        #     current_results['patient_name'] = subject_data['ID']

        #     current_results["X total"]         = X_total * PIXEL_LENGTH
        #     current_results["Y total"]         = Y_total * PIXEL_LENGTH
        #     current_results['magnitude total'] = magnitude_total * PIXEL_LENGTH

        #     current_results["X inner avg"]         = X_inner_avg * PIXEL_LENGTH
        #     current_results["Y inner avg"]         = Y_inner_avg * PIXEL_LENGTH
        #     current_results["magnitude inner avg"] = magnitude_inner_avg * PIXEL_LENGTH

        #     current_results["X outer avg"]         = X_outer_avg * PIXEL_LENGTH
        #     current_results["Y outer avg"]         = Y_outer_avg * PIXEL_LENGTH
        #     current_results["magnitude outer avg"] = magnitude_outer_avg * PIXEL_LENGTH

        #     current_results["X 45 inner"]  = X_inner_45 * PIXEL_LENGTH
        #     current_results["X 135 inner"] = X_inner_135 * PIXEL_LENGTH
        #     current_results["X 225 inner"] = X_inner_225 * PIXEL_LENGTH
        #     current_results["X 315 inner"] = X_inner_315 * PIXEL_LENGTH
        #     current_results["Y 45 inner"]  = Y_inner_45 * PIXEL_LENGTH
        #     current_results["Y 135 inner"] = Y_inner_135 * PIXEL_LENGTH
        #     current_results["Y 225 inner"] = Y_inner_225 * PIXEL_LENGTH
        #     current_results["Y 315 inner"] = Y_inner_315 * PIXEL_LENGTH

        #     current_results["X 45 outer"]  = X_outer_45 * PIXEL_LENGTH
        #     current_results["X 135 outer"] = X_outer_135 * PIXEL_LENGTH
        #     current_results["X 225 outer"] = X_outer_225 * PIXEL_LENGTH
        #     current_results["X 315 outer"] = X_outer_315 * PIXEL_LENGTH
        #     current_results["Y 45 outer"]  = Y_outer_45 * PIXEL_LENGTH
        #     current_results["Y 135 outer"] = Y_outer_135 * PIXEL_LENGTH
        #     current_results["Y 225 outer"] = Y_outer_225 * PIXEL_LENGTH
        #     current_results["Y 315 outer"] = Y_outer_315 * PIXEL_LENGTH

        #     current_results["magnitude 45 inner"]  = magnitude_inner_45 * PIXEL_LENGTH
        #     current_results["magnitude 135 inner"] = magnitude_inner_135 * PIXEL_LENGTH
        #     current_results["magnitude 225 inner"] = magnitude_inner_225 * PIXEL_LENGTH
        #     current_results["magnitude 315 inner"] = magnitude_inner_315 * PIXEL_LENGTH

        #     current_results["magnitude 45 outer"]  = magnitude_outer_45 * PIXEL_LENGTH
        #     current_results["magnitude 135 outer"] = magnitude_outer_135 * PIXEL_LENGTH
        #     current_results["magnitude 225 outer"] = magnitude_outer_225 * PIXEL_LENGTH
        #     current_results["magnitude 315 outer"] = magnitude_outer_315 * PIXEL_LENGTH

        #     all_patients_data.append(current_results)

        #     # ---------------------------------- PLOT DEI RISULTATI ---------------------------------- #

        #     fig, ax = plt.subplots(figsize=(8, 8))
        #     ax.imshow(subject_data['DRAW IMG'])
        #     grid_color = 'white'
        #     thickness = 1.5
        #     linestyle = '--'
            
        #     # Disegno cerchi e diagonali
        #     inner_circle = patches.Circle((cx, cy), INNER_RADIUS, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle)
        #     outer_circle = patches.Circle((cx, cy), OUTER_RADIUS, fill=False, edgecolor=grid_color, linewidth=thickness, linestyle=linestyle)
        #     ax.add_patch(inner_circle)
        #     ax.add_patch(outer_circle)
        #     offset = OUTER_RADIUS * 0.707
        #     ax.plot([cx - offset, cx + offset], [cy - offset, cy + offset], color=grid_color, linewidth=thickness, linestyle=linestyle)
        #     ax.plot([cx - offset, cx + offset], [cy + offset, cy - offset], color=grid_color, linewidth=thickness, linestyle=linestyle)

        #     ax.plot(cx, cy, marker='+', color=grid_color, markersize=15, markeredgewidth=thickness)
            
        #     # Impostazione dinamica di scala e colore in base allo sfondo
        #     if config.draw_option == 'IR':
        #         q_color, q_scale = 'lime', 1000
        #     elif config.draw_option == 'labels':
        #         q_color, q_scale = 'lime', 1000
        #     else:
        #         q_color, q_scale = 'mediumblue', 1000

        #     # Maschera per non stampare vettori sullo sfondo bianco
        #     retina_mask = subject_data['IMG PRE'] < 250

        #     u_plot = u_final.copy()
        #     v_plot = v_final.copy()

        #     u_plot[~retina_mask] = np.nan
        #     v_plot[~retina_mask] = np.nan

        #     # Plot del campo vettoriale principale usando Matplotlib Quiver
        #     step_q = 30
        #     ax.quiver(x_coords[::step_q,::step_q], y_coords[::step_q,::step_q], 
        #               u_plot[::step_q,::step_q], v_plot[::step_q,::step_q], 
        #               angles='xy', color=q_color, alpha=0.8, scale=q_scale, width=0.001, headwidth=3, headlength=4, headaxislength=3.5)

        #     # Creazione della freccia di riferimento (Scale Bar)
        #     ref_length_um = 50 
        #     ref_length_px = ref_length_um / PIXEL_LENGTH
            
        #     x_ref = w - 300
        #     y_ref = h - 200

        #     ax.quiver(x_ref, y_ref, ref_length_px, 0, 
        #               angles='xy', color='white', scale=q_scale, width=0.003, headwidth=4)
            
        #     ax.text(x_ref + (ref_length_px/2), y_ref + 80, f'{ref_length_um} $\mu m$', 
        #         color='white', fontsize=18, fontweight='bold', family='serif', ha='center', va='top')

        #     ax.axis('off')
        #     plt.tight_layout()
            
        #     # Salvataggio delle immagini
        #     plt.savefig(os.path.join(config.DRAW_DST_PATH, f"{subject_data['ID']}_RBF_plot.png"), dpi=600, bbox_inches='tight', pad_inches=0)
        #     plt.savefig(os.path.join(config.DRAW_DST_PATH, f"{subject_data['ID']}_RBF_plot.pdf"), dpi=600, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig)

        #     # Aggiornamento CSV
        #     df = pd.DataFrame(all_patients_data)
        #     df.to_csv(config.CSV_RBF_FIELD_PATH, index=False)

