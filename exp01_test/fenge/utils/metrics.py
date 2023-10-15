import torch
import torch.nn as nn
import numpy as np
import math
import scipy.spatial
import scipy.ndimage.morphology
"""
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本 
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""

def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    smooth = 1e-5#防止分母为0
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)#正确的肿块的像素个数,tp+fn
    tp = np.sum((im_pred == im_lab) * (im_lab > 0))#预测正确的肿块的像素个数
    fn = pixel_labeled-tp#应该是肿块，预测成背景
    background = np.sum(im_lab == 0)#tn+fp
    tn = np.sum((im_pred == im_lab) * (im_lab == 0))#预测正确的背景的像素个数
    fp = background-tn#应该是背景，预测成肿块

    precision = 1.0 * (tp+smooth) / (tp+fp+smooth)
    acc = np.sum(im_pred == im_lab)/(pixel_labeled + background)
    pixel_back = 1.0 * (tn+smooth) / (background+smooth)
    recall = 1.0 * (tp+smooth) / (pixel_labeled+smooth)
    dice = 2*(tp+smooth)/(2*tp+fp+fn+smooth)

    return recall, pixel_back, acc, precision,dice,pixel_labeled

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)#21*21的矩阵,行代表ground truth类别,列代表preds的类别,值代表

    def evaluation(self,mode):
        tp = self.confusion_matrix[1][1]
        tn = self.confusion_matrix[0][0]
        fp = self.confusion_matrix[0][1]
        fn = self.confusion_matrix[1][0]
        if mode == 'PA':#acc
            if tp+tn+fp+fn ==0:
                return 0.0
            else:
                return (tp+tn)/(tp+tn+fp+fn)
        if mode =='PA_mass':#recall,Sensitivity(敏感性))
            if (tp+fn) == 0:
                return 0.0,(tp+fn)
            else:
                return tp/(tp+fn),(tp+fn)
        if mode=='PA_beijing':#specificity（特异性)
            if (tn+fp)==0:
                return 0.0
            else:
                return tn/(tn+fp)
        if mode =='miou':
            if (tp+fp+fn)==0:
                return 0.0
            elif (tn+fn+fp)==0:
                return 0.0
            else:
                return (tp/(tp+fp+fn)+tn/(tn+fn+fp))/2
        if mode == 'iou':
            if (tp+fp+fn)==0:
                return 0.0
            else:
                return tp/(tp+fp+fn)
        if mode == 'precision':#精确率
            if (tp+fp) ==0:
                return 0.0
            else:
                return tp/(tp+fp)
        if mode == 'f1':#dice
            if (tp+fp) == 0:
                P = 0
            else:
                P=tp/(tp+fp)
            if (tp+fn) == 0:
                R = 0
            else:
                R=tp/(tp+fn)
            if (P+R)==0:
                return 0.0
            else:
                return 2*P*R/(P+R)
        if mode == 'FPR':
            if (fp+tn) == 0:
                return 0.0,fp
            else:
                return fp/(fp+tn),fp
        if mode == 'dice':
            if (2*tp + fn + fp) == 0:
                return 0.0
            else:
                return 2*tp/(2*tp + fn + fp)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    '''
    parameters:
        mask: ground truth中所有正确(值在[0, classe_num])的像素label的mask---为了保证ground truth中的标签值都在合理的范围[0, 20]
        label: 为了计算混淆矩阵, 混淆矩阵中一共有num_class*num_class个数, 所以label中的数值也是在0与num_class**2之间. [batch_size, 512, 512]
        cout(reshape): 记录了每个类别对应的像素个数,行代表真实类别,列代表预测的类别,count矩阵中(x, y)位置的元素代表该张图片中真实类别为x,被预测为y的像素个数
        np.bincount: https://blog.csdn.net/xlinsist/article/details/51346523
        confusion_matrix: 对角线上的值的和代表分类正确的像素点个数(preb与target一致),对角线之外的其他值的和代表所有分类错误的像素的个数
    '''
    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        label = label.astype('int')
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)#21 * 21(for pascal)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # tmp = self._generate_matrix(gt_image, pre_image)
        #矩阵相加是各个元素对应相加,即21*21的矩阵进行pixel-wise加和
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

