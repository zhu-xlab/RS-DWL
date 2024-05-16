import os
import json
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F
from PIL import Image
from dataloader import CreateTestDataLoader
from torch.autograd import Variable
from options.test_options import TestOptions
from utils.metric import Evaluator
import cv2
torch.set_num_threads(4)


def evaluate(num_classes, val_loader, model):
    model.eval()

    metric = Evaluator(num_class=num_classes)
    with torch.no_grad():
        for i, (val_img, val_gt, _) in enumerate(val_loader):
            val_img  = val_img.cuda()      # to gpu
            val_pred = model(val_img)      
            val_pred = torch.max(val_pred, dim=1)[1].unsqueeze(dim=1)
            val_pred = val_pred.cpu().detach().numpy()
            val_gt = val_gt.unsqueeze(dim=1).cpu().detach().numpy()
            metric.add_batch(val_gt, val_pred)

    val_acc = metric.Pixel_Accuracy_Class()
    val_IoU = metric.Intersection_over_Union()

    model.train()

    return val_acc, val_IoU



