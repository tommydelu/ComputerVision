"""

When I launch the script in the terminal this would be the ideal command: python3 codice1.py IR/L_01PRE.jpg
But the files in the directory contains a white space --> this function remove all the white spaces from the names
of the files

"""

import os

GLOBAL_PATH = os.getcwd()
SRC_PATH    = os.path.join(GLOBAL_PATH,'DATASET','IR')

if __name__ == '__main__':

    for fname in os.listdir(SRC_PATH):

        new_fname = fname.replace(" ", "") # replace a white space with the None character

        old_fpath = os.path.join(SRC_PATH, fname)
        new_fpath = os.path.join(SRC_PATH, new_fname)

        os.rename(old_fpath, new_fpath)