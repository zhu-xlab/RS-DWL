import numpy as np
from sklearn.metrics import f1_score



class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def F1_Score(self):
        Precison = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        Recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        F1 = 2*Precison*Recall / (Precison + Recall)
        return F1

    def Precision_Recall(self):
        Precison = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        Recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        return Precison, Recall

    def Mean_F1_Score(self):
        F1 = self.F1_Score()
        Mean_F1 = np.nanmean(F1)
        return Mean_F1

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
        
    def Kappa(self):
        # 计算Po
        Po = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        # 计算每类的预测和实际总数
        sum_pred = np.sum(self.confusion_matrix, axis=0)
        sum_true = np.sum(self.confusion_matrix, axis=1)
        
        # 计算Pe
        Pe = np.sum(sum_pred * sum_true) / (self.confusion_matrix.sum() ** 2)
        
        # 计算Kappa
        kappa = (Po - Pe) / (1 - Pe)
        return kappa

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def measure_pa_miou(num_class, gt_image, pre_image):
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    acc = metric.Pixel_Accuracy()
    mIoU = metric.Mean_Intersection_over_Union()
    print("Pixel_Accuracy:", acc, "mean_IOU:", mIoU)