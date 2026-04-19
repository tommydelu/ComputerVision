import os
import cv2 as cv
import numpy as np
import pandas as pd
import shutil

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_results_1')
DST_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','best_10_subjects_segmentation_1')
METRICS_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_1.csv')

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)


if __name__ == '__main__':

    df = pd.read_csv(METRICS_PATH)
    df = df[df['SUBJECT'] != 'METRIC MEAN']

    top_10 = df.sort_values(by='Dice Score', ascending=False).head(10)
    
    subjects_processed = set()
    for idx,row in top_10.iterrows():

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
        
