import os
import tifffile
import cv2 as cv
import numpy as np
import random


labeled_train_ratios = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.2, 0.5, 1]


def crop(FILE_DIR):
    files = os.listdir(FILE_DIR)
    files = [os.path.join(FILE_DIR, i) for i in files]
    for path in files:
        if '_1' not in path and '_2' not in path and '_3' not in path and '_4' not in path:
            print (path)
            image =  cv.imread(path)
            image_ul = image[:512,:512,:]
            image_ur = image[:512,512:,:]
            image_dl = image[512:,:512,:]
            image_dr = image[512:,512:,:]
            image_path_ul = path.replace('.png', '_1.png')
            image_path_ur = path.replace('.png', '_2.png')
            image_path_dl = path.replace('.png', '_3.png')
            image_path_dr = path.replace('.png', '_4.png')
            cv.imwrite(image_path_ul, image_ul)
            cv.imwrite(image_path_ur, image_ur)
            cv.imwrite(image_path_dl, image_dl)
            cv.imwrite(image_path_dr, image_dr)
            os.remove(path)


def split(TRAIN_IMAGE_DIR, VAL_IMAGE_DIR, OUT_DIR, DOMAIN):
    images_train = os.listdir(TRAIN_IMAGE_DIR)
    images_train = [os.path.join(TRAIN_IMAGE_DIR, i) for i in images_train]
    random.shuffle(images_train)

    for image in images_train:
        print (image)

    # labeled_train_ratios = [0.02]
    for labeled_train_ratio in labeled_train_ratios:
        len_labeled_train = int(labeled_train_ratio*len(images_train))

        images_labeled_train   = images_train[:len_labeled_train]
        images_unlabeled_train = images_train[len_labeled_train:]
        masks_labeled_train    = [i.replace('image', 'mask') for i in images_labeled_train]
        masks_unlabeled_train  = [i.replace('image', 'mask') for i in images_unlabeled_train]

        labeled_train_txt = os.path.join(OUT_DIR, DOMAIN+'_train_{:.1f}%_labeled.txt'.format(labeled_train_ratio*100))
        with open(labeled_train_txt, 'w') as f:
            for i in images_labeled_train:
                f.write(i + ' ' + i.replace('image', 'mask') + '\n')

        unlabeled_train_txt = os.path.join(OUT_DIR, DOMAIN+'_train_{:.1f}%_unlabeled.txt'.format(labeled_train_ratio*100))
        with open(unlabeled_train_txt, 'w') as f:
            for i in images_unlabeled_train:
                f.write(i + ' ' + i.replace('image', 'mask') + '\n')

    images_val = os.listdir(VAL_IMAGE_DIR)
    images_val = [os.path.join(VAL_IMAGE_DIR, i) for i in images_val]
    random.shuffle(images_val)
    val_txt = os.path.join(OUT_DIR, DOMAIN+'_val.txt')
    with open(val_txt, 'w') as f:
        for i in images_val:
            f.write(i + ' ' + i.replace('image', 'mask') + '\n')


def merge_rural_urban(rural_txt, urban_txt, merged_txt):
    with open(rural_txt, 'r') as rural_file:
        data_rural = rural_file.read()
    with open(urban_txt, 'r') as urban_file:
        data_urban = urban_file.read()

    merged_data = data_rural + data_urban
    with open(merged_txt, 'w') as merged_file:
        merged_file.write(merged_data)



OUT_DIR = './lists'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

# Crop Images
CROP_SIZE = 512
TRAIN_RURAL_IMAGE_DIR = './Train/Rural/images_png'
TRAIN_RURAL_MASK_DIR  = './Train/Rural/masks_png'
TRAIN_URBAN_IMAGE_DIR = './Train/Urban/images_png'
TRAIN_URBAN_MASK_DIR  = './Train/Urban/masks_png'
VAL_RURAL_IMAGE_DIR = './Val/Rural/images_png'
VAL_RURAL_MASK_DIR  = './Val/Rural/masks_png'
VAL_URBAN_IMAGE_DIR = './Val/Urban/images_png'
VAL_URBAN_MASK_DIR  = './Val/Urban/masks_png'
RURAL_DOMAIN = 'rural'
URBAN_DOMAIN = 'urban'

# crop(TRAIN_RURAL_IMAGE_DIR)
# crop(TRAIN_RURAL_MASK_DIR)
# crop(TRAIN_URBAN_IMAGE_DIR)
# crop(TRAIN_URBAN_MASK_DIR)
# crop(VAL_RURAL_IMAGE_DIR)
# crop(VAL_RURAL_MASK_DIR)
# crop(VAL_URBAN_IMAGE_DIR)
# crop(VAL_URBAN_MASK_DIR)


split(TRAIN_RURAL_IMAGE_DIR, VAL_RURAL_IMAGE_DIR, OUT_DIR, RURAL_DOMAIN)
split(TRAIN_URBAN_IMAGE_DIR, VAL_URBAN_IMAGE_DIR, OUT_DIR, URBAN_DOMAIN)


for labeled_train_ratio in labeled_train_ratios:
    labeled_urban_train_txt = os.path.join(OUT_DIR, 'urban_train_{:.1f}%_labeled.txt'.format(labeled_train_ratio*100))
    labeled_rural_train_txt = os.path.join(OUT_DIR, 'rural_train_{:.1f}%_labeled.txt'.format(labeled_train_ratio*100))
    labeled_merged_train_txt = os.path.join(OUT_DIR, 'train_{:.1f}%_labeled.txt'.format(labeled_train_ratio*100))
    merge_rural_urban(labeled_rural_train_txt, labeled_urban_train_txt, labeled_merged_train_txt)

    unlabeled_urban_train_txt = os.path.join(OUT_DIR, 'urban_train_{:.1f}%_unlabeled.txt'.format(labeled_train_ratio*100))
    unlabeled_rural_train_txt = os.path.join(OUT_DIR, 'rural_train_{:.1f}%_unlabeled.txt'.format(labeled_train_ratio*100))
    unlabeled_merged_train_txt = os.path.join(OUT_DIR, 'train_{:.1f}%_unlabeled.txt'.format(labeled_train_ratio*100))
    merge_rural_urban(unlabeled_rural_train_txt, unlabeled_urban_train_txt, unlabeled_merged_train_txt)


urban_val_txt = os.path.join(OUT_DIR, 'urban_val.txt')
rural_val_txt = os.path.join(OUT_DIR, 'rural_val.txt')
merged_val_txt = os.path.join(OUT_DIR, 'val.txt')
merge_rural_urban(urban_val_txt, rural_val_txt, merged_val_txt)


data_val = []
with open(merged_val_txt, 'r') as file:
    for line in file:
        data_val.append(line)  # 去除换行符并添加到列表

random.shuffle(data_val)
data_split_val  = data_val[int(0.5*len(data_val)):]
data_split_test = data_val[:int(0.5*len(data_val))]

with open(os.path.join(OUT_DIR, 'val.txt'), 'w') as file:
    for line in data_split_val:
        file.write(line)
with open(os.path.join(OUT_DIR, 'test.txt'), 'w') as file:
    for line in data_split_test:
        file.write(line)
