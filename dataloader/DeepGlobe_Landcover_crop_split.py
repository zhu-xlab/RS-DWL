import os
import tifffile
import cv2 as cv
import numpy as np
import random



# Crop Images
CROP_SIZE = 512
INPUT_DIR = './data'
OUTPUT_DIR = './data_cropped'

# if not os.path.exists(OUTPUT_DIR):
#     os.mkdir(OUTPUT_DIR)

# files = os.listdir(INPUT_DIR)
# files = [i for i in files if 'sat' in i]

# files = [[os.path.join(INPUT_DIR, i), 
# 		  os.path.join(INPUT_DIR, i.replace('sat.jpg', 'mask.png'))] 
#            for i in files]

# mode = 1
# for path in files:
#     image_path, label_path = path
#     image = cv.imread(image_path, mode)
#     label = cv.imread(label_path, mode)

#     cnt = 0
#     height, width = image.shape[:2]
#     height_num, width_num = np.int(np.ceil(height/CROP_SIZE)), np.int(np.ceil(width/CROP_SIZE))
#     print (image_path, width, height, width_num, height_num)
#     for h in range(height_num):
#         for w in range(width_num):
#             h_str, h_end = h*CROP_SIZE, min((h+1)*CROP_SIZE, height)
#             w_str, w_end = w*CROP_SIZE, min((w+1)*CROP_SIZE, width)
#             image_cur = image[h_str:h_end, w_str:w_end]
#             label_cur = label[h_str:h_end, w_str:w_end]
#             cnt += 1

#             # pad
#             if h_end-h_str < CROP_SIZE:
#                 h_diff = CROP_SIZE - (h_end-h_str)
#                 image_pad_h = np.zeros((h_diff, image_cur.shape[1], image_cur.shape[2]), dtype=np.uint8)
#                 label_pad_h = np.zeros((h_diff, image_cur.shape[1], image_cur.shape[2]), dtype=np.uint8)
#                 image_cur = np.concatenate([image_cur, image_pad_h], axis=0)                
#                 label_cur = np.concatenate([label_cur, label_pad_h], axis=0)                
#             if w_end-w_str < CROP_SIZE:
#                 w_diff = CROP_SIZE - (w_end-w_str)
#                 image_pad_w = np.zeros((image_cur.shape[0], w_diff, image_cur.shape[2]), dtype=np.uint8)
#                 label_pad_w = np.zeros((image_cur.shape[0], w_diff, image_cur.shape[2]), dtype=np.uint8)
#                 image_cur = np.concatenate([image_cur, image_pad_w], axis=1)                
#                 label_cur = np.concatenate([label_cur, label_pad_w], axis=1)                

#             image_path_cur = image_path.replace(INPUT_DIR, OUTPUT_DIR)
#             label_path_cur = label_path.replace(INPUT_DIR, OUTPUT_DIR)
#             image_path_cur = image_path_cur.replace('.jpg', '_'+str(cnt)+'.jpg')
#             label_path_cur = label_path_cur.replace('.png', '_'+str(cnt)+'.png')
#             cv.imwrite(image_path_cur, image_cur)
#             cv.imwrite(label_path_cur, label_cur)
#             print (image_path_cur, image_cur.shape)


# Split Lists
LIST_DIR = './lists'
if not os.path.exists(LIST_DIR):
    os.mkdir(LIST_DIR)

files_image = os.listdir(OUTPUT_DIR)
files_image = [i for i in files_image if 'sat' in i]
random.shuffle(files_image)
files_image_train = files_image[:int(0.6*len(files_image))]
files_image_val   = files_image[int(0.6*len(files_image)):int(0.8*len(files_image))]
files_image_test  = files_image[int(0.8*len(files_image)):]

train_txt = os.path.join(LIST_DIR, 'train.txt')
with open(train_txt, 'w') as f:
    for i in files_image_train:
        f.write('data_cropped/' + i + ' ' + \
                'data_cropped/' + i.replace('sat', 'mask').replace('jpg', 'png') + '\n')
val_txt = os.path.join(LIST_DIR, 'val.txt')
with open(val_txt, 'w') as f:
    for i in files_image_val:
        f.write('data_cropped/' + i + ' ' + \
                'data_cropped/' + i.replace('sat', 'mask').replace('jpg', 'png') + '\n')
test_txt = os.path.join(LIST_DIR, 'test.txt')
with open(test_txt, 'w') as f:
    for i in files_image_test:
        f.write('data_cropped/' + i + ' ' + \
                'data_cropped/' + i.replace('sat', 'mask').replace('jpg', 'png') + '\n')

train_labeled_ratios = [0.01, 0.02, 0.05, 0.10, 0.2, 0.5, 1]
for train_labeled_ratio in train_labeled_ratios:
    len_labeled_train = int(train_labeled_ratio*len(files_image_train))
    files_image_train_labeled   = files_image_train[:len_labeled_train]
    files_image_train_unlabeled = files_image_train[len_labeled_train:]

    train_labeled_txt = os.path.join(LIST_DIR, 'train_{:.1f}%_labeled.txt'.format(train_labeled_ratio*100))
    train_unlabeled_txt = os.path.join(LIST_DIR, 'train_{:.1f}%_unlabeled.txt'.format(train_labeled_ratio*100))
    with open(train_labeled_txt, 'w') as f:
        for i in files_image_train_labeled:
            f.write('data_cropped/' + i + ' ' + \
                    'data_cropped/' + i.replace('sat', 'mask').replace('jpg', 'png') + '\n')
    with open(train_unlabeled_txt, 'w') as f:
        for i in files_image_train_unlabeled:
            f.write('data_cropped/' + i + ' ' + \
                    'data_cropped/' + i.replace('sat', 'mask').replace('jpg', 'png') + '\n')




