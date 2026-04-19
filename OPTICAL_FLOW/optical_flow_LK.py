#--------------------- LIBRARIES ---------------------#


import os
import cv2 as cv
import numpy as np
from functions import *
import pandas as pd


#-----------------------------------------------------#

fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

GLOBAL_PATH = os.getcwd()
SRC_PATH    = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'aligned_images_optic_nerve')
SRC_PATH1    = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'alignment_colors_results_optic_nerve')
MASKS_PATH  = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation_results_1')
LABELS_PATH = os.path.join(GLOBAL_PATH, 'DATASET', 'labels_reversed')
DST_PATH    = os.path.join(GLOBAL_PATH, 'OPTICAL_FLOW', 'LK_aligned_colored_flow_vectors_results_1')
CSV_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','LK_optical_flow_results_1.csv')

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)

best_subjects = ['L_06','L_15','L_26','L_30','L_42','L_48','L_63','L_78','S_08','S_46']

PIXEL_LENGTH = 4.651162790697675 # in um

INNER_DIAMETER = 4000/PIXEL_LENGTH  # pixels --> 4 mm = 4000 um / PIXEL_LENGTH
OUTER_DIAMETER = 8000/PIXEL_LENGTH # pixels --> 8 mm = 8000 um / PIXEL_LENGTH

# NB: AL MOMENTO LE DUE CIRCONFERENZE SONO DISEGNATE AL CENTRO DELL'IMMAGINE E NON ESATTAMENTE AL CENTRO DELLA FOVEA

INNER_RADIUS = int(INNER_DIAMETER/2) # pixels
OUTER_RADIUS = int(OUTER_DIAMETER/2) # pixels

MAX_ALLOWED_MOVEMENT = 50


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



def get_quadrant(px,py,cx,cy,inner_r,outer_r):

    """
    Funzione per capire a che quadrante appartiene il punto su cui stiamo calcolando il vettore spostamento.
    Input:
    px --> coordinata x del punto
    py --> coordinata y del punto
    cx, cy --> coordinate del centro dei cerchi
    inner_r, outer_r --> raggio della circonferenza interna ed esterna

    output --> ring: il punto appartiene al ring interno o esterno, quad: a che quadrante appartiene il punto
    """

    dx = px - cx
    dy = py - cy
    dist = np.sqrt(dx**2 + dy**2)
    if dist > outer_r:
        # Fuori dal reticolo: punto inutile ai fini dei risultati
        return None 
    
    angle = np.degrees(np.arctan2(-dy, dx))
    
    ring = "inner" if dist <= inner_r else "outer"
    
    if -45 <= angle < 45:
        quad = "315" 
    elif 45 <= angle < 135:
        quad = "225"         
    elif angle >= 135 or angle < -135:
        quad = "135" 
    else:
        quad = "45"         
        
    return ring,quad



#-----------------------------------------------------#


if __name__ == '__main__':

    all_patients_data = []

    for fname in sorted(os.listdir(SRC_PATH)):
        
        current_results = results.copy()

        if 'POST' in fname:
            continue
    
        subject = fname.split('PRE')[0] # estraggo il soggetto che sto processando
        
        if subject not in best_subjects:
            continue

        current_results['patient_name'] = subject

        fname_post = f"{subject}POST.JPG"

        fname_aligned_colored_imgs = os.path.join(SRC_PATH1,f'{subject}_confronto_post.png')
        aligned_colored_imgs = cv.imread(fname_aligned_colored_imgs)


        IMG_PATH1 = os.path.join(SRC_PATH,fname)
        IMG_PATH2 = os.path.join(SRC_PATH,fname_post)

        MASK_PATH1 = os.path.join(MASKS_PATH,fname)
        MASK_PATH2 = os.path.join(MASKS_PATH,fname_post)

        LABEL_PATH1 = os.path.join(LABELS_PATH,subject,'total_1.png')
        LABEL_PATH2 = os.path.join(LABELS_PATH,subject,'total_2.png')

        imgPre  = cv.imread(IMG_PATH1,0)
        imgPre = clahe(imgPre,2,4)

        imgPost = cv.imread(IMG_PATH2,0)
        imgPost = clahe(imgPost,2,4)

        h, w = imgPre.shape
        cx,cy = fovea_center_dict[subject]

        maskPre  = cv.imread(MASK_PATH1,0)
        maskPost = cv.imread(MASK_PATH2,0)

        labelPre  = cv.imread(LABEL_PATH1,0)
        labelPost = cv.imread(LABEL_PATH2,0)

        step = 5
        x = np.arange(0, w, step, dtype=np.float32)
        y = np.arange(0, h, step, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        pts_x = X.flatten()
        pts_y = Y.flatten()

        idxs_to_keep = labelPre[pts_y.astype(int), pts_x.astype(int)] > 0
        prevPts = np.stack((pts_x[idxs_to_keep], pts_y[idxs_to_keep]), axis=-1) # axis = -1 è l'ultimo asse

        # Now we can compute the optical flow
        nextPts, status, err =  cv.calcOpticalFlowPyrLK(imgPre,
                                imgPost,
                                prevPts,
                                None,
                                winSize=(31,31),
                                maxLevel=3,
                                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
                                minEigThreshold=1e-4
                                )
        
        flow_vectors = nextPts - prevPts

        X_total = []
        Y_total = []
        magnitude_total = []

        X_inner_avg = []
        Y_inner_avg = []
        magnitude_inner_avg = []

        X_outer_avg = []
        Y_outer_avg = []
        magnitude_outer_avg = []

        X_inner_45 = []
        X_inner_135 = []
        X_inner_225 = []
        X_inner_315 = []
        X_outer_45 = []
        X_outer_135 = []
        X_outer_225 = []
        X_outer_315 = []

        Y_inner_45 = []
        Y_inner_135 = []
        Y_inner_225 = []
        Y_inner_315 = []
        Y_outer_45 = []
        Y_outer_135 = []
        Y_outer_225 = []
        Y_outer_315 = []

        magnitude_inner_45 = []
        magnitude_inner_135 = []
        magnitude_inner_225 = []
        magnitude_inner_315 = []
        magnitude_outer_45 = []
        magnitude_outer_135 = []
        magnitude_outer_225 = []
        magnitude_outer_315 = []

        flow_vectors_img = cv.cvtColor(imgPre, cv.COLOR_GRAY2BGR)
        draw_on_label_pre = cv.cvtColor(labelPre, cv.COLOR_GRAY2BGR)

        scale_factor = 2  # Rendo lo spostamento nel disegno più visibile
        draw_step = 10 
        counter = 0

        for i in range(len(flow_vectors)):
            
            if status[i] == 1:

                u = flow_vectors[i,0]
                v = flow_vectors[i,1]
                magnitude = np.sqrt(u**2 + v**2)

                if magnitude < MAX_ALLOWED_MOVEMENT:

                    if counter % draw_step == 0:
                        pt1 = (int(prevPts[i,0]), int(prevPts[i,1]))
                        # pt2 = (int(nextPts[i,0]), int(nextPts[i,1]))            

                        pt2_x = int(prevPts[i,0] + u * scale_factor)
                        pt2_y = int(prevPts[i,1] + v * scale_factor)
                        pt2 = (pt2_x, pt2_y)

                        cv.arrowedLine(aligned_colored_imgs, pt1, pt2, (255, 0, 0), 1, tipLength=0.3)
                        #cv.line(aligned_colored_imgs, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)
                        cv.circle(aligned_colored_imgs, pt2, 2, (0, 0, 255), -1)
                    counter += 1

                    res = get_quadrant(prevPts[i,0], prevPts[i,1], cx, cy, INNER_RADIUS, OUTER_RADIUS)

                    if res is not None:
                        ring, quad = res

                        X_total.append(u)
                        Y_total.append(v)
                        magnitude_total.append(magnitude)

                        if ring == 'inner':
                            X_inner_avg.append(u)
                            Y_inner_avg.append(v)
                            magnitude_inner_avg.append(np.sqrt(u**2 + v**2))

                            if quad == '45':
                                X_inner_45.append(u)
                                Y_inner_45.append(v)
                                magnitude_inner_45.append(np.sqrt(u**2 + v**2))
                            if quad == '135':
                                X_inner_135.append(u)
                                Y_inner_135.append(v)
                                magnitude_inner_135.append(np.sqrt(u**2 + v**2))
                            if quad == '225':
                                X_inner_225.append(u)
                                Y_inner_225.append(v)
                                magnitude_inner_225.append(np.sqrt(u**2 + v**2))
                            if quad == '315':
                                X_inner_315.append(u)
                                Y_inner_315.append(v)
                                magnitude_inner_315.append(np.sqrt(u**2 + v**2))

                        if ring == 'outer':
                            X_outer_avg.append(u)
                            Y_outer_avg.append(v)
                            magnitude_outer_avg.append(np.sqrt(u**2 + v**2))

                            if quad == '45':
                                X_outer_45.append(u)
                                Y_outer_45.append(v)
                                magnitude_outer_45.append(np.sqrt(u**2 + v**2))
                            if quad == '135':
                                X_outer_135.append(u)
                                Y_outer_135.append(v)
                                magnitude_outer_135.append(np.sqrt(u**2 + v**2))
                            if quad == '225':
                                X_outer_225.append(u)
                                Y_outer_225.append(v)
                                magnitude_outer_225.append(np.sqrt(u**2 + v**2))
                            if quad == '315':
                                X_outer_315.append(u)
                                Y_outer_315.append(v)
                                magnitude_outer_315.append(np.sqrt(u**2 + v**2))


        grid_color = (0,0, 0) # Bianco
        thickness = 2
        cv.circle(flow_vectors_img, (cx, cy), INNER_RADIUS, grid_color, thickness)
        cv.circle(flow_vectors_img, (cx, cy), OUTER_RADIUS, grid_color, thickness)
        offset = int(OUTER_RADIUS * 0.707)
        cv.line(flow_vectors_img, (cx - offset, cy - offset), (cx + offset, cy + offset), grid_color, thickness)
        # Diagonale 2: Alto-Sx a Basso-Dx
        cv.line(flow_vectors_img, (cx - offset, cy + offset), (cx + offset, cy - offset), grid_color, thickness)

        # 3. Opzionale: Un piccolo crocino sul centro della fovea
        cv.drawMarker(flow_vectors_img, (cx, cy), grid_color, cv.MARKER_CROSS, 20, 2)

        img_name = f"{subject}_flow.jpg"
        cv.imwrite(os.path.join(DST_PATH, img_name),aligned_colored_imgs)


    #     current_results["X total"]         = np.mean(X_total) * PIXEL_LENGTH
    #     current_results["Y total"]         = np.mean(Y_total) * PIXEL_LENGTH
    #     current_results['magnitude total'] = np.mean(magnitude_total) * PIXEL_LENGTH

    #     current_results["X inner avg"]         = np.mean(X_inner_avg) * PIXEL_LENGTH
    #     current_results["Y inner avg"]         = np.mean(Y_inner_avg) * PIXEL_LENGTH
    #     current_results["magnitude inner avg"] = np.mean(magnitude_inner_avg) * PIXEL_LENGTH

    #     current_results["X outer avg"]         = np.mean(X_outer_avg) * PIXEL_LENGTH
    #     current_results["Y outer avg"]         = np.mean(Y_outer_avg) * PIXEL_LENGTH
    #     current_results["magnitude outer avg"] = np.mean(magnitude_outer_avg) * PIXEL_LENGTH

    #     current_results["X 45 inner"]  = np.mean(X_inner_45) * PIXEL_LENGTH
    #     current_results["X 135 inner"] = np.mean(X_inner_135) * PIXEL_LENGTH
    #     current_results["X 225 inner"] = np.mean(X_inner_225) * PIXEL_LENGTH
    #     current_results["X 315 inner"] = np.mean(X_inner_315) * PIXEL_LENGTH
    #     current_results["Y 45 inner"]  = np.mean(Y_inner_45) * PIXEL_LENGTH
    #     current_results["Y 135 inner"] = np.mean(Y_inner_135) * PIXEL_LENGTH
    #     current_results["Y 225 inner"] = np.mean(Y_inner_225) * PIXEL_LENGTH
    #     current_results["Y 315 inner"] = np.mean(Y_inner_315) * PIXEL_LENGTH

    #     current_results["X 45 outer"]  = np.mean(X_outer_45) * PIXEL_LENGTH
    #     current_results["X 135 outer"] = np.mean(X_outer_135) * PIXEL_LENGTH
    #     current_results["X 225 outer"] = np.mean(X_outer_225) * PIXEL_LENGTH
    #     current_results["X 315 outer"] = np.mean(X_outer_315) * PIXEL_LENGTH
    #     current_results["Y 45 outer"]  = np.mean(Y_outer_45) * PIXEL_LENGTH
    #     current_results["Y 135 outer"] = np.mean(Y_outer_135) * PIXEL_LENGTH
    #     current_results["Y 225 outer"] = np.mean(Y_outer_225) * PIXEL_LENGTH
    #     current_results["Y 315 outer"] = np.mean(Y_outer_315) * PIXEL_LENGTH

    #     current_results["magnitude 45 inner"]  = np.mean(magnitude_inner_45) * PIXEL_LENGTH
    #     current_results["magnitude 135 inner"] = np.mean(magnitude_inner_135) * PIXEL_LENGTH
    #     current_results["magnitude 225 inner"] = np.mean(magnitude_inner_225) * PIXEL_LENGTH
    #     current_results["magnitude 315 inner"] = np.mean(magnitude_inner_315) * PIXEL_LENGTH

    #     current_results["magnitude 45 outer"]  = np.mean(magnitude_outer_45) * PIXEL_LENGTH
    #     current_results["magnitude 135 outer"] = np.mean(magnitude_outer_135) * PIXEL_LENGTH
    #     current_results["magnitude 225 outer"] = np.mean(magnitude_outer_225) * PIXEL_LENGTH
    #     current_results["magnitude 315 outer"] = np.mean(magnitude_outer_315) * PIXEL_LENGTH

    #     all_patients_data.append(current_results)

    # df = pd.DataFrame(all_patients_data)
    # df.to_csv(CSV_PATH, index=False)  





        




