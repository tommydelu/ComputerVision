"""
This code is performed to extract from peeling_results all the labels 
"""

import os
import shutil

GLOBAL_PATH = os.getcwd()
PEELING_DIR_PATH = os.path.join(GLOBAL_PATH,'DATASET',"peeling_results")
DST_PATH = os.path.join(GLOBAL_PATH,'DATASET', "labels")

if not os.path.exists(DST_PATH):
        os.makedirs(DST_PATH)

if __name__ == "__main__":


    for dir in sorted(os.listdir(PEELING_DIR_PATH)):

        SUBJECT_PATH = os.path.join(PEELING_DIR_PATH, dir)
        
        if not os.path.isdir(SUBJECT_PATH):
            continue

        OUTPUT_PATH = os.path.join(DST_PATH, dir)

        if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)


        target_files = ["total_1.png", "total_2.png"]

        found_files = [f for f in os.listdir(SUBJECT_PATH) if f in target_files]

        if found_files:
                for fname in found_files:
                    src_file = os.path.join(SUBJECT_PATH, fname)
                    dst_file = os.path.join(OUTPUT_PATH, fname)
                    shutil.copy2(src_file, dst_file)



   
            

    

    
        
        
        

                
            
            

