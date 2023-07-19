
import torch.nn as nn
import torch.nn.functional as F
import torch


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 分割结果掩码
        P = y_pred
        # 真实标签掩码
        G = y_true
        
        # 计算Dice损失
        intersection = torch.sum(P * G)
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (torch.sum(P) + torch.sum(G) + 1.0)
        
        # # 计算IOU损失
        # union = torch.sum(P) + torch.sum(G) - intersection
        # iou_loss = 1.0 - intersection / (union + 1.0)
        
        # # 计算Hausdorff距离
        # # 找到分割结果中每个像素的坐标
        # pred_indices = torch.nonzero(P)
        # # 找到真实标签中每个像素的坐标
        # target_indices = torch.nonzero(G)

        # # 计算分割结果与真实标签之间的欧氏距离
        # distance_matrix = torch.cdist(pred_indices.float(), target_indices.float())

        # # 计算最大距离
        # max_distance = torch.max(
        #     torch.max(distance_matrix, dim=1).values, dim=0
        # ).values.item()
        
        # # 组合损失
        # alpha = 0.4
        # beta = 0.3
        # gamma = 0.3
        # loss = alpha * dice_loss + beta * iou_loss + gamma * (1.0 - max_distance)
        
        return dice_loss
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        n, c, w, h = outputs.shape
        label_sums = torch.sum(torch.sum(label, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        total = w * h
        weight = label_sums / total
        loss = label * outputs
        
        loss *= weight
        
        entire_loss = torch.sum(loss)/(n*total)
        return entire_loss
    
def dice_loss(pred_mask, gt_mask):
    smooth = 1e-5

    # 将预测掩码和真实标签转换为二进制形式
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    # 计算相交和并集
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask)

    # 计算Dice系数
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)

    # 计算Dice损失
    loss = 1.0 - dice_coefficient

    return loss
