import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics_functions import *



GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'OPTICAL_FLOW','segmentation_results_1')
LABEL_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','aligned_labels')


if __name__ == '__main__':


    results = []
    for file in tqdm(sorted(os.listdir(SRC_PATH))):

        if 'PRE' in file:

            subject = file.split('PRE')[0]
            phase = 'PRE'
            target = 'total_1.png'

        elif 'POST' in file:

            subject = file.split('POST')[0]
            phase = 'POST'
            target = 'total_2.png'

        src_path = os.path.join(SRC_PATH,file)
        label_path = os.path.join(LABEL_PATH,subject,target)

        if not os.path.exists(label_path):
            continue

        pred = cv.imread(src_path,0)
        label = cv.imread(label_path,0)
        
        row_id = f"{subject}{phase}"

        # CALCOLO DELLE METRICHE CON LA LABEL ORIGINALE
        sens, spec, acc, fpr, mcc, dice, IoU = get_metrics(pred,label)

        # CALCOLO DELLE METRICHE CON LA LABEL FLIPPATA
        label_flipped = cv.flip(label, 1)
        sens1, spec1, acc1, fpr1, mcc1, dice1, IoU1 = get_metrics(pred,label_flipped)

        # PRENDO LE METRICHE CORRISPONDENTI A UN MCC PIU ALTO
        if dice > dice1:
            metrics = {"SUBJECT":row_id,
                       "Sensitivity": sens,
                       "Specificity": spec,
                       "Accuracy": acc,
                       "False Positive Rate": fpr,
                       "MCC": mcc,
                       "Dice Score":dice,
                       "Jaccard Index": IoU,
                       "Flipped": False}
        else:
            metrics = {"SUBJECT":row_id,
                       "Sensitivity": sens1,
                       "Specificity": spec1,
                       "Accuracy": acc1,
                       "False Positive Rate": fpr1,
                       "MCC": mcc1,
                       "Dice Score":dice1,
                       "Jaccard Index": IoU1,
                       "Flipped": True}
                
        results.append(metrics)

    df = pd.DataFrame(results)
    means = df.mean(numeric_only=True)

    results_row = {"SUBJECT": "METRIC MEAN",
                   "Sensitivity": means["Sensitivity"],
                   "Specificity": means["Specificity"],
                   "Accuracy": means["Accuracy"],
                   "False Positive Rate":means["False Positive Rate"],
                   "MCC": means["MCC"],
                   "Dice Score":means["Dice Score"],
                   "Jaccard Index":means["Jaccard Index"],
                   "Flipped": None}
    
    results_row_df = pd.DataFrame([results_row])
    
    df = pd.concat([df,results_row_df],ignore_index=True)

    csv_name = os.path.basename(SRC_PATH)
    csv_split = csv_name.split('_')

    csv_filename = f"OPTICAL_FLOW/{csv_split[0]}_{csv_split[2]}.csv"

    df.to_csv(csv_filename,index=False)





