import os
import cv2 as cv
import numpy as np
import random



# ## Crop Images
# CROP_SIZE = 512
# RGB_DIR = './Potsdam/2_Ortho_RGB'
# IRRG_DIR = './Potsdam/3_Ortho_IRRG'
# LABEL_DIR = './Potsdam/5_Labels_all'
# OUT_RGB_DIR = './Potsdam/rgb'
# OUT_IRRG_DIR = './Potsdam/irrg'
# OUT_LABEL_DIR = './Potsdam/label'

# if not os.path.exists(OUT_RGB_DIR):
#     os.mkdir(OUT_RGB_DIR)
# if not os.path.exists(OUT_IRRG_DIR):
#     os.mkdir(OUT_IRRG_DIR)
# if not os.path.exists(OUT_LABEL_DIR):
#     os.mkdir(OUT_LABEL_DIR)

# files = os.listdir(RGB_DIR)
# files = [[i, i.replace('RGB', 'IRRG'), i.replace('RGB', 'label')] 
#             for i in files if '.tif' in i]
# files = [[os.path.join(RGB_DIR, i[0]), os.path.join(IRRG_DIR, i[1]), os.path.join(LABEL_DIR, i[2])] 
#             for i in files]

# mode = 1
# for path in files:
#     rgb_path, irrg_path, label_path = path
#     rgb = cv.imread(rgb_path, mode)
#     irrg = cv.imread(irrg_path, mode)
#     label = cv.imread(label_path, mode)

#     cnt = 0
#     height, width = rgb.shape[:2]
#     height_num, width_num = np.int(np.ceil(height/CROP_SIZE)), np.int(np.ceil(width/CROP_SIZE))
#     print (rgb_path, width, height, width_num, height_num)
#     for h in range(height_num):
#         for w in range(width_num):
#             h_str, h_end = h*CROP_SIZE, min((h+1)*CROP_SIZE, height)
#             w_str, w_end = w*CROP_SIZE, min((w+1)*CROP_SIZE, width)
#             rgb_cur = rgb[h_str:h_end, w_str:w_end]
#             irrg_cur = irrg[h_str:h_end, w_str:w_end]
#             label_cur = label[h_str:h_end, w_str:w_end]
#             cnt += 1

#             # pad
#             if h_end-h_str < CROP_SIZE:
#                 h_diff = CROP_SIZE - (h_end-h_str)
#                 rgb_pad_h = np.zeros((h_diff, rgb_cur.shape[1], rgb_cur.shape[2]), dtype=np.uint8)
#                 label_pad_h = np.zeros((h_diff, rgb_cur.shape[1], rgb_cur.shape[2]), dtype=np.uint8)
#                 label_pad_h[:,:,2] = 255
#                 rgb_cur = np.concatenate([rgb_cur, rgb_pad_h], axis=0)                
#                 irrg_cur = np.concatenate([irrg_cur, rgb_pad_h], axis=0)                
#                 label_cur = np.concatenate([label_cur, label_pad_h], axis=0)                
#             if w_end-w_str < CROP_SIZE:
#                 w_diff = CROP_SIZE - (w_end-w_str)
#                 rgb_pad_w = np.zeros((rgb_cur.shape[0], w_diff, rgb_cur.shape[2]), dtype=np.uint8)
#                 label_pad_w = np.zeros((rgb_cur.shape[0], w_diff, rgb_cur.shape[2]), dtype=np.uint8)
#                 label_pad_w[:,:,2] = 255
#                 rgb_cur = np.concatenate([rgb_cur, rgb_pad_w], axis=1)                
#                 irrg_cur = np.concatenate([irrg_cur, rgb_pad_w], axis=1)                
#                 label_cur = np.concatenate([label_cur, label_pad_w], axis=1)                

#             rgb_path_cur = rgb_path.replace(RGB_DIR, OUT_RGB_DIR)
#             irrg_path_cur = irrg_path.replace(IRRG_DIR, OUT_IRRG_DIR)
#             label_path_cur = label_path.replace(LABEL_DIR, OUT_LABEL_DIR)
#             rgb_path_cur = rgb_path_cur.replace('_RGB.', '_'+str(cnt)+'_RGB.')
#             irrg_path_cur = irrg_path_cur.replace('_IRRG.', '_'+str(cnt)+'_IRRG.')
#             label_path_cur = label_path_cur.replace('_label.', '_'+str(cnt)+'_label.')
#             cv.imwrite(rgb_path_cur, rgb_cur)
#             cv.imwrite(irrg_path_cur, irrg_cur)
#             cv.imwrite(label_path_cur, label_cur)
#             print (irrg_path_cur, irrg_cur.shape)



## Split lists
RGB_DIR = './Vaihingen_IRRG/image'
LABEL_DIR = './Vaihingen_IRRG/label'
LIST_RGB_DIR = './Potsdam_RGB/lists'
LIST_IRRG_DIR = './Potsdam_IRRG/lists'
if not os.path.exists(LIST_RGB_DIR):
    os.mkdir(LIST_RGB_DIR)

files_rgb = os.listdir(RGB_DIR)
random.shuffle(files_rgb)
files_rgb_train = files_rgb[:int(0.6*len(files_rgb))]
files_rgb_val   = files_rgb[int(0.6*len(files_rgb)):int(0.8*len(files_rgb))]
files_rgb_test  = files_rgb[int(0.8*len(files_rgb)):]

train_txt = os.path.join(LIST_RGB_DIR, 'train.txt')
with open(train_txt, 'w') as f:
    for i in files_rgb_train:
        f.write('image/' + i + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
val_txt = os.path.join(LIST_RGB_DIR, 'val.txt')
with open(val_txt, 'w') as f:
    for i in files_rgb_val:
        f.write('image/' + i + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
test_txt = os.path.join(LIST_RGB_DIR, 'test.txt')
with open(test_txt, 'w') as f:
    for i in files_rgb_test:
        f.write('image/' + i + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')

if not os.path.exists(LIST_IRRG_DIR):
    os.mkdir(LIST_IRRG_DIR)
train_txt = os.path.join(LIST_IRRG_DIR, 'train.txt')
with open(train_txt, 'w') as f:
    for i in files_rgb_train:
        f.write('image/' + i.replace('RGB', 'IRRG') + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
val_txt = os.path.join(LIST_IRRG_DIR, 'val.txt')
with open(val_txt, 'w') as f:
    for i in files_rgb_val:
        f.write('image/' + i.replace('RGB', 'IRRG') + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
test_txt = os.path.join(LIST_IRRG_DIR, 'test.txt')
with open(test_txt, 'w') as f:
    for i in files_rgb_test:
        f.write('image/' + i.replace('RGB', 'IRRG') + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')


train_labeled_ratios = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.2, 0.5, 1]
for train_labeled_ratio in train_labeled_ratios:
    untrain_labeled_ratio = 1 - train_labeled_ratio

    len_train_labeled = int(train_labeled_ratio*len(files_rgb_train))
    len_train_unlabeled = int(untrain_labeled_ratio*len(files_rgb_train))
    files_rgb_train_labeled = files_rgb_train[:len_train_labeled]
    files_rgb_train_unlabeled = files_rgb_train[len_train_labeled:]

    labeled_train_txt = os.path.join(LIST_RGB_DIR, 'train_{:.1f}%_labeled.txt'.format(train_labeled_ratio*100))
    train_unlabeled_txt = os.path.join(LIST_RGB_DIR, 'train_{:.1f}%_unlabeled.txt'.format(train_labeled_ratio*100))
    with open(labeled_train_txt, 'w') as f:
        for i in files_rgb_train_labeled:
            f.write('image/' + i + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
    with open(train_unlabeled_txt, 'w') as f:
        for i in files_rgb_train_unlabeled:
            f.write('image/' + i + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')

    labeled_train_txt = os.path.join(LIST_IRRG_DIR, 'train_{:.1f}%_labeled.txt'.format(train_labeled_ratio*100))
    train_unlabeled_txt = os.path.join(LIST_IRRG_DIR, 'train_{:.1f}%_unlabeled.txt'.format(train_labeled_ratio*100))
    with open(labeled_train_txt, 'w') as f:
        for i in files_rgb_train_labeled:
            f.write('image/' + i.replace('RGB', 'IRRG') + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')
    with open(train_unlabeled_txt, 'w') as f:
        for i in files_rgb_train_unlabeled:
            f.write('image/' + i.replace('RGB', 'IRRG') + ' ' + 'label/' + i.replace('RGB', 'label') + '\n')


