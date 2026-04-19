import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# fai feature matching su immagini filtrate

GLOBAL_PATH = os.getcwd()
IR_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR')
DST_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','aligned_images_optic_nerve')
DST_LABEL_PATH = os.path.join(GLOBAL_PATH, 'ALIGNMENT_PREPOST', 'aligned_labels')
LABEL_PATH = os.path.join(GLOBAL_PATH, 'DATASET', 'labels_reversed')
txt_path = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','lato_nervo.txt')


if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)

if not os.path.exists(DST_LABEL_PATH):
    os.makedirs(DST_LABEL_PATH)

info_lati = {}
with open(txt_path, 'r') as f:
    for riga in f:
        parti = riga.strip().split()
        if len(parti) == 2:
            info_lati[parti[0]] = parti[1]


if __name__ == '__main__':

    for fname in sorted(os.listdir(IR_PATH)):


        if 'POST' in fname:
                    continue

        if 'PRE' in fname:
            id = fname.split('PRE')[0]
       
        lato_nervo = info_lati.get(id)

        if lato_nervo is None:
            print(f"\n[ATTENZIONE] ID {id} non trovato nel file di testo per il file {fname}")
            continue
        
        
        IMG_PRE_PATH = os.path.join(IR_PATH,fname)

        subject = fname.split('PRE')[0]
        fname_post = f"{subject}POST.jpg"

        IMG_POST_PATH = os.path.join(IR_PATH,fname_post)

        LABEL_DIR_PATH = os.path.join(LABEL_PATH,subject)
        LABEL_PRE_PATH = os.path.join(LABEL_DIR_PATH,'total_1.png')
        LABEL_POST_PATH = os.path.join(LABEL_DIR_PATH,'total_2.png')

        if not os.path.exists(LABEL_PRE_PATH) or not os.path.exists(LABEL_POST_PATH):
            continue

        labelPre = cv.imread(LABEL_PRE_PATH,0)
        labelPost = cv.imread(LABEL_POST_PATH,0)

        imgPre = cv.imread(IMG_PRE_PATH,0)
        imgPost = cv.imread(IMG_POST_PATH,0)

        h,w = imgPre.shape                


        wavelet_filtered_imgPre = wavelet_filtering(imgPre)
        wavelet_filtered_imgPost = wavelet_filtering(imgPost)

        median_blurPre = cv.medianBlur(wavelet_filtered_imgPre,5)
        median_blurPost = cv.medianBlur(wavelet_filtered_imgPost,5)

        clahe_imgPre = clahe(median_blurPre,clipLimit=1.5,gridSize=16)
        clahe_imgPost = clahe(median_blurPost,clipLimit=1.5,gridSize=16)

        mask = np.zeros_like(clahe_imgPre)

        larghezza_roi = int(w * 0.25)
        altezza_roi = int(h * 0.5)
        y_start = (h // 2) - (altezza_roi // 2)
        y_end = y_start + altezza_roi

        if lato_nervo == 'sx':
            cv.rectangle(mask, (0, y_start), (larghezza_roi, y_end), 255, -1)

        elif lato_nervo == 'dx':     
            cv.rectangle(mask, (w - larghezza_roi, y_start), (w, y_end), 255, -1)
      

        sift = cv.SIFT_create(nfeatures = 3000)
        kp1, des1 = sift.detectAndCompute(clahe_imgPre, mask=mask) # plot kps
        kp2, des2 = sift.detectAndCompute(clahe_imgPost, mask=mask)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2,k=2) # per ogni punto nella PRE mi trova i due punti + simili nella POST

        good_matches = []

        # Lowe's ratio test: m primo match, n secondo match
        for m, n in matches:
            if m.distance < 0.7 * n.distance: # il primo match deve essere abbastanza diverso dal secondo, non ambiguo
                good_matches.append(m)
                
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

           # The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to combinations of translation, rotation, and uniform scaling 
            matrix, inliers = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC)

            """
            Tra PRE e POST l'immagine può essere solo traslata, ruotata o leggerment zoomata. Non può essere inclinata in prospettiva o
            piegata --> uso cv.estimateAffinePartial2D perchè limita i gradi di libertà a 4 parametri, impedisco distorsioni
            """

            height, width = imgPre.shape

            # The function cv.warpAffine transforms the source image using the specified matrix:
            img_post_aligned = cv.warpAffine(imgPost, matrix, (width, height))
            label_post_aligned = cv.warpAffine(labelPost, matrix, (width, height), 
                                                   flags=cv.INTER_NEAREST)
        

            OUTPUT_SUBJECT_PATH = os.path.join(DST_LABEL_PATH, subject)
            if not os.path.exists(OUTPUT_SUBJECT_PATH):
                os.makedirs(OUTPUT_SUBJECT_PATH, exist_ok=True)

            label_output_path1 = os.path.join(OUTPUT_SUBJECT_PATH,'total_1.png')
            label_output_path2 = os.path.join(OUTPUT_SUBJECT_PATH,'total_2.png')

            output_path1 = os.path.join(DST_PATH,fname)
            output_path2 = os.path.join(DST_PATH,fname_post)

            cv.imwrite(label_output_path1,labelPre)
            cv.imwrite(label_output_path2,label_post_aligned)
            cv.imwrite(output_path1,imgPre)
            cv.imwrite(output_path2,img_post_aligned)

        else:
            subject_discarded = f"{subject} non aveva abbastaza matches"
            with open("ALIGNMENT_PREPOST/subjects_discarded.txt",'a',encoding='utf-8') as file:
                file.write(subject_discarded+'\n')


