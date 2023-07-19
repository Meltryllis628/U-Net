import torch
import torch.nn as nn


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()

    def forward(self, y_pred, y_true):
        # 分割结果掩码
        P = (y_pred > 0.5).float()
        # 真实标签掩码
        G = (y_true > 0.5).float()

        # 计算Dice损失
        intersection = torch.sum(P * G)
        dice = (2.0 * intersection + 1.0) / (torch.sum(P) + torch.sum(G) + 1.0)

        # 计算IOU损失
        union = torch.sum(P) + torch.sum(G) - intersection
        iou = intersection / (union + 1.0)

        # # 计算Hausdorff距离
        # # 找到分割结果中每个像素的坐标
        # pred_indices = torch.nonzero(P)
        # # 找到真实标签中每个像素的坐标
        # target_indices = torch.nonzero(G)

        # 计算分割结果与真实标签之间的欧氏距离
        # distance_matrix = torch.cdist(pred_indices.float(), target_indices.float())

        # # 计算最大距离
        # max_distance = torch.max(
        #     torch.max(distance_matrix, dim=1).values, dim=0
        # ).values.item()

        # normalized_hausdorff_dist = max_distance / torch.sqrt(
        #     torch.tensor(G.size(-1) ** 2 + G.size(-2) ** 2)
        # )
        # normalized_hausdorff_dist = 1

        # 组合损失
        alpha = 0.4
        beta = 0.3
        gamma = 0.3
        score = alpha * dice + beta * iou
        # + gamma * (1.0 - normalized_hausdorff_dist)

        return score, dice, iou
