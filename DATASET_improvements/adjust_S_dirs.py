import os

GLOBAL_PATH = os.getcwd()
SRC_PATH = os.path.join(GLOBAL_PATH,'DATASET','labels')
SRC_PATH2 = os.path.join(GLOBAL_PATH,'DATASET','labels_reversed')

if __name__ == '__main__':

    for dir in sorted(os.listdir(SRC_PATH),reverse=True):
        
        old_path1 = os.path.join(SRC_PATH,dir)
        old_path2 = os.path.join(SRC_PATH2,dir)

        if 'L' in dir:
            continue

        sub = dir.split('_')[0]
        num = int(dir.split('_')[1])

        if num < 18:
            continue
        
        if num >= 19 and num < 24:
            num +=1
        elif num >= 24 and num < 28:
            num+=2
        elif num >= 28:
            num+=3
        
        new_dir_name = f"{sub}_{num}"
        new_path1 = os.path.join(SRC_PATH,new_dir_name)
        new_path2 = os.path.join(SRC_PATH2,new_dir_name)

        os.rename(old_path1,new_path1)
        os.rename(old_path2,new_path2)