import os
import cv2 as cv
import numpy as np


class ConfigPaths:


    def __init__(self,mode,draw_option):

        self.GLOBAL_PATH = os.getcwd()
        self.mode = mode
        self.draw_option = draw_option

        self.SRC_PATH = os.path.join(self.GLOBAL_PATH,'IMAGE_REGISTRATION','results','figures','aligned_imgs3')
        self.MASKS_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW', 'segmentation', 'results','figures','best_10_subjects_segmentation_1')
        self.LABELS_PATH = os.path.join(self.GLOBAL_PATH,'IMAGE_REGISTRATION','results','figures','aligned_labels')

        self.CSV_SEG_PATH  = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW','segmentation','results','data','segmentation_1.csv')


        if mode == "farneback":
            self.CSV_RBF_FIELD_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW','farneback','results','data','RBF_quadrants_2.csv')
            self.RBF_MATRICES_PATH = os.path.join(self.GLOBAL_PATH,'OPTICAL_FLOW','farneback','results','data','RBF_matrices_2')
        elif mode == "lukas_kannade":
            self.CSV_RBF_FIELD_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW','lukas_kannade','results','data','RBF_quadrants_2.csv')
            self.RBF_MATRICES_PATH = os.path.join(self.GLOBAL_PATH,'OPTICAL_FLOW','lukas_kannade','results','data','RBF_matrices_2')

    
        if draw_option == 'IR':
            self.DRAW_SRC_PATH = self.SRC_PATH
            self.DRAW_DST_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW',self.mode,'results','figures','RBF_on_IR_2')
        elif draw_option == 'labels':
            self.DRAW_SRC_PATH = self.LABELS_PATH
            self.DRAW_DST_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW',self.mode, 'RBF_on_labels_2')
        elif draw_option == 'coloured_superposition':
            self.DRAW_SRC_PATH = os.path.join(self.GLOBAL_PATH, 'IMAGE_REGISTRATION','results','figures','alignment_color_superposition_3')
            self.DRAW_DST_PATH = os.path.join(self.GLOBAL_PATH, 'OPTICAL_FLOW', self.mode, 'results','figures','RBF_on_coloured_superposition_2')

        os.makedirs(self.DRAW_DST_PATH, exist_ok=True)
        os.makedirs(self.RBF_MATRICES_PATH, exist_ok=True)







class Subject:

    def __init__(self,fname,src_path,labels_path):

        if 'PRE' in fname:
            self.subject = fname.split('PRE')[0]
            self.phase = 'PRE'
            self.target = 'total_1.png'

        elif 'POST' in fname:
            self.subject = fname.split('POST')[0]
            self.phase = 'POST'
            self.target = 'total_2.png'

        self.img_path = os.path.join(src_path, fname)
        self.img = cv.imread(self.img_path,0)
        self.label_path = os.path.join(labels_path, self.subject, self.target)

        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found for subject {self.subject} and phase {self.phase}. Expected at: {self.label_path}")

        self.label = cv.imread(self.label_path,0)


    def correct_artifacts(self,img_to_correct):

        _, mask_artifacts = cv.threshold(self.img, 253, 255, cv.THRESH_BINARY)
        _, mask_black_zones = cv.threshold(self.img,2,255,cv.THRESH_BINARY_INV)
        kernel_mask = np.ones((15, 15), np.uint8) 
        forbidden_zone = cv.dilate(mask_artifacts, kernel_mask, iterations=3)

        img_to_correct[forbidden_zone == 255] = 0
        img_to_correct[mask_black_zones == 255] = 0 # Aggiunta a segmentation_results_2

        if img_to_correct.shape[:2] != self.label.shape[:2]:
            img_to_correct = cv.resize(img_to_correct,(self.label.shape[1],self.label.shape[0]),interpolation=cv.INTER_NEAREST)

        return img_to_correct


       