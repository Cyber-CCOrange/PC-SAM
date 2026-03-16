import numpy as np
from sklearn.metrics import confusion_matrix

class SegmentationMetric():
    def __init__(self,numclass):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass,) * 2)

    def CM(self, mask, pred):
        mask_num = mask.numpy().astype('int')
        pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask_num.flatten()
        cm = confusion_matrix(mask,pred)
        self.confusionMatrix += cm
        # return cm

    def PA(self):
        # PA = acc
        acc = np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        return acc

    def CPA(self):
        # CPA = (tp)/tp+fp
        # precision
        cpa = np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0)
        return cpa

    def meanPA(self):
        cpa = self.CPA()
        meanpa = np.nanmean(cpa)
        return meanpa
    def CPR(self):
        # CPR:class recall
        cpr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return cpr

    def IOU(self):
        intersection = np.diag(self.confusionMatrix)  
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iou = intersection/union
        return iou
