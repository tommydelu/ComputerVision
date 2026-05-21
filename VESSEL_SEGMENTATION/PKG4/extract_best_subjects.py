import os
import cv2 as cv
import numpy as np
import pandas as pd
import shutil

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','segmentation_results_2')
DST_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','best_5_subjects_segmentation_2')
METRICS_PATH = os.path.join(GLOBAL_PATH,'PACKAGE_codice4','segmentation_2_results.csv')

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)


if __name__ == '__main__':

    df = pd.read_csv(METRICS_PATH)
    df = df[df['SUBJECT'] != 'METRIC MEAN']

    top_5 = df.sort_values(by='Dice Score', ascending=False).head()
    
    subjects_processed = set()
    for idx,row in top_5.iterrows():

        full_id = row['SUBJECT']
        
        base_subject = full_id.replace('PRE', '').replace('POST', '')
        
        if base_subject in subjects_processed:
            continue
        
        filename1 = f"{base_subject}PRE.jpg"
        filename2 = f"{base_subject}POST.jpg"

        path1  = os.path.join(SRC_PATH,filename1)
        path2  = os.path.join(SRC_PATH,filename2)

        dst_path1 = os.path.join(DST_PATH,filename1)
        dst_path2 = os.path.join(DST_PATH,filename2)

        if os.path.exists(path1) and os.path.exists(path2):
            shutil.copy(path1, dst_path1)
            shutil.copy(path2, dst_path2)
            subjects_processed.add(base_subject)

        else:
            if os.path.exists(path1): shutil.copy(path1, dst_path1)
            if os.path.exists(path2): shutil.copy(path2, dst_path2)
        
