import os
import cv2 as cv
import numpy as np

GLOBAL_PATH = os.getcwd()
POST_SIFT_IMG_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','aligned_images_optic_nerve')
PRE_SIFT_IMG_PATH = os.path.join(GLOBAL_PATH,'DATASET','IR')
DST_PATH = os.path.join(GLOBAL_PATH,'ALIGNMENT_PREPOST','alignment_colors_results_optic_nerve')

if not os.path.exists(DST_PATH):
    os.makedirs(DST_PATH)


def create_alignment_overlay(fixed_img, moving_img):
   
    h, w = fixed_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:,:,0] = cv.addWeighted(fixed_img, 0.5, moving_img, 0.5, 0)
    overlay[:,:,1] = moving_img
    overlay[:,:,2] = fixed_img

    return overlay


if __name__ == '__main__':

    for fname in sorted(os.listdir(PRE_SIFT_IMG_PATH)):
        
        if 'POST' in fname:
            continue
        
        subject = fname.split('PRE')[0]
        fname_post = f"{subject}POST.jpg"

        FIXED_IMG_PATH = os.path.join(PRE_SIFT_IMG_PATH,fname)
        POST_IMG_PRE_SIFT_PATH = os.path.join(PRE_SIFT_IMG_PATH,fname_post)
        POST_IMG_POST_SIFT_PATH = os.path.join(POST_SIFT_IMG_PATH,fname_post)

        if not os.path.exists(POST_IMG_POST_SIFT_PATH):
            continue

        fixed_img = cv.imread(FIXED_IMG_PATH,0)
        post_img_pre_sift = cv.imread(POST_IMG_PRE_SIFT_PATH,0)
        post_img_post_sift = cv.imread(POST_IMG_POST_SIFT_PATH,0)

        overlay_pre = create_alignment_overlay(fixed_img, post_img_pre_sift)
        overlay_post = create_alignment_overlay(fixed_img, post_img_post_sift)

        out_path1 = os.path.join(DST_PATH,f"{subject}_confronto_pre.png")
        out_path2 = os.path.join(DST_PATH,f"{subject}_confronto_post.png")

        cv.imwrite(out_path1, overlay_pre)
        cv.imwrite(out_path2, overlay_post)








