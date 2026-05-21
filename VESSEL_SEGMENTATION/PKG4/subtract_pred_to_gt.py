import os
import cv2 as cv
import pandas as pd
import numpy as np

GLOBAL_PATH = os.getcwd()
PRED_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','segmentation_results_2')
GT_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels_reversed')
DST_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','gt_label_subtraction_segmentation_2')

metrics_path = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','segmentation_2_results.csv')

if not os.path.exists(DST_PATH):
    os.mkdir(DST_PATH)

if __name__ == '__main__':

    list_perc_missed_pixels = []
    for fname in sorted(os.listdir(PRED_PATH)):

        if 'PRE' in fname:
            subject = fname.split('PRE')[0]
            phase = 'PRE'
            target = 'total_1.png'
        elif 'POST' in fname:
            subject = fname.split('POST')[0]
            phase = 'POST'
            target = 'total_2.png'

        row_id = f"{subject}{phase}"

        csv_flipped_param = pd.read_csv(metrics_path).set_index('SUBJECT')
        is_flipped = csv_flipped_param.loc[row_id,'Flipped']

        IMG_PATH = os.path.join(PRED_PATH, fname)
        LABEL_PATH = os.path.join(GT_PATH,subject,target)
        if not os.path.exists(LABEL_PATH):
            continue

        img = cv.imread(IMG_PATH,0)
        label = cv.imread(LABEL_PATH,0)
        _, label = cv.threshold(label,127,255,cv.THRESH_BINARY)
        
        if is_flipped:
            label = cv.flip(label, 1)


        total_num_pixels = img.size

        result = cv.subtract(label,img)

        mask = (result > 127)
        missed_true_pixels = np.sum(mask)

        percentage_missed_pixels = missed_true_pixels/total_num_pixels*100
        df_info = {"ID": row_id,
                   "% missed pixels":percentage_missed_pixels}

        list_perc_missed_pixels.append(percentage_missed_pixels)

        OUTPUT_PATH = os.path.join(DST_PATH,fname)
        cv.imwrite(OUTPUT_PATH,result)

    df = pd.DataFrame(list_perc_missed_pixels)

    csv_filename = f"PACKAGE_codice4/missed_pixels_segmentation_2.csv"

    df.to_csv(csv_filename,index=False)
        

