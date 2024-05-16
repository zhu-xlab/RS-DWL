import os
import cv2
import torch
import random
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from .weak_augment import *
from .strong_augment import color_swap, hsv_shift, RandAugmentMC
from collections import namedtuple
from copy import deepcopy


class DataSet_RSseg(data.Dataset):
    def __init__(self, root, list_path, base_size=512, mode='val', ignore_index=-1):
        self.mode = mode
        self.root = root  # folder for GTA5 which contains subfolder images, labels
        self.list_path = osp.join(root, list_path)   # list of image names
        self.ignore_index = ignore_index
        self.strong = RandAugmentMC(n=2, m=10)
        self.paths = []
        self.base_size = base_size

        f = open(self.list_path)            
        self.paths = f.readlines()    
        self.paths = [path.strip('\n') for path in self.paths]  
        print (len(self.paths))

        f.close()  

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, label_path = self.paths[index].split(' ')
        image = Image.open(osp.join(self.root, image_path)).convert('RGB')
        label = Image.open(osp.join(self.root, label_path.strip('\n'))).convert('L')

        if self.mode == 'val':
            image, label = normalize(image, label)
            return image, label, [image_path, label_path]

        elif self.mode == 'train_l':
            ignore_value = self.ignore_index
            image, label = resize(image, label, (0.5, 2.0))
            image, label = crop(image, label, self.base_size, ignore_value)
            image, label = hflip(image, label, p=0.5)
            image, label = vflip(image, label, p=0.5)
            image, label = rotate(image, label)
            image, label = normalize(image, label)
            return image, label, [image_path, label_path]

        elif self.mode == 'train_u':
            ignore_value = self.ignore_index
            image, label = resize(image, label, (0.5, 2.0))
            image, label = crop(image, label, self.base_size, ignore_value)
            image, label = hflip(image, label, p=0.5)
            image, label = vflip(image, label, p=0.5)
            image, label = rotate(image, label)

            image_w = deepcopy(image)
            image_w = normalize(image_w)
            
            image_s = deepcopy(image)
            # image_s = self.strong(image_s)
            if random.random() < 0.8:
                image_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s)
            image_s = transforms.RandomGrayscale(p=0.2)(image_s)
            image_s = blur(image_s, p=0.5)
            image_s, label = normalize(image_s, label)
            return image_w, image_s, label, [image_path, label_path]



class DataSet_Cityscapes(data.Dataset):
    def __init__(self, root, list_path, base_size=700, mode='val', ignore_index=255):
        self.root = root  # folder for GTA5 which contains subfolder images, labels
        self.list_path = osp.join(root, list_path)   # list of image names
        self.ignore_index = ignore_index
        self.mode = mode
        self.paths = []
        self.base_size = base_size

        f = open(self.list_path)            
        self.paths = f.readlines()    
        self.paths = [path.strip('\n') for path in self.paths]  
        f.close()  
        print (len(self.paths))

        self.get_classes()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, label_path = self.paths[index].split(' ')
        image = Image.open(osp.join(self.root, image_path)).convert('RGB')
        label = Image.open(osp.join(self.root, label_path.strip('\n'))).convert('RGB')
        label = Image.fromarray(self.color_to_label(np.asarray(label, np.uint8)))

        if self.mode == 'val':
            return normalize(image, label), [image_path, label_path]

        elif self.mode == 'train_l':
            ignore_value = self.ignore_index
            # image, label = resize(image, label, (0.5, 2.0))
            image, label = crop(image, label, self.base_size, ignore_value)
            image, label = hflip(image, label, p=0.5)
            return normalize(image, label), [image_path, label_path]

        elif self.mode == 'train_u':
            ignore_value = self.ignore_index
            # image, label = resize(image, label, (0.5, 2.0))
            image, label = crop(image, label, self.base_size, ignore_value)
            image, label = hflip(image, label, p=0.5)

            image_w = deepcopy(image)
            image_w = normalize(image_w)

            image_s = deepcopy(image)
            if random.random() < 0.8:
                image_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s)
            image_s = transforms.RandomGrayscale(p=0.2)(image_s)
            image_s = blur(image_s, p=0.5)
            image_s, label = normalize(image_s, label)

            return image_w, image_s, label, [image_path, label_path]

    def color_to_label(self, label):
        label = label[:,:,0]*65536 + label[:,:,1]*256 + label[:,:,2]
        return self.color_to_label_matrix[label]

    def get_classes(self):
        CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
        classes = [
            CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
            CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
            CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
            CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
            CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
            CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
            CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
            CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
            CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
            CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
            CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
            CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
            CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
            CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
            CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
            CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
            CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
            CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
            CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
            CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
            CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
            CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
            CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
            CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
            CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
            CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
            CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
        ]
        
        self.color_to_label_list = {c.color:c.train_id for c in classes}
        self.color_to_label_matrix = np.zeros(256*256*256, dtype=np.int32)
        for color, indice in self.color_to_label_list.items():
            self.color_to_label_matrix[color[0]*65536 + color[1]*256 + color[2]] = indice

