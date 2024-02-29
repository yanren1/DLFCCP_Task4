import torch
import numpy as np
from sklearn.metrics import average_precision_score

def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def MAPE(y_true, y_pred):
    epsilon = 1e-8
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true+epsilon))) * 100

    return mape


def cal_map(preds, labels):
    average_precisions = []
    for class_idx in range(preds.shape[1]):
        # 计算 Precision-Recall 曲线下的面积，即 Average Precision (AP)
        ap = average_precision_score(labels[:, class_idx], preds[:, class_idx])

        # 将 AP 添加到列表中
        average_precisions.append(ap)

    # 计算 mAP，即所有类别的 AP 的平均值
    map_value = np.mean(average_precisions)
    return map_value
