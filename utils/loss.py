import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, y_pred, label, gt_mask=None):
        if len(label.shape) != 4:
            label = label.unsqueeze(1)

        if gt_mask is not None:
            y_pred = -y_pred
            label = (1 - label) * gt_mask  
        y_pred = torch.sigmoid(y_pred)
        if gt_mask is not None:
            y_pred = y_pred * gt_mask

        input_unflat = y_pred
        target = label
        N = target.size(0)
        smooth = 1
        input_flat = input_unflat.view(N, -1)
        target_flat = target.contiguous().view(N, -1)
        intersection = torch.sum(input_flat * target_flat, dim=1)
        loss = (2. * intersection + smooth) / (torch.sum(input_flat, dim=1) + torch.sum(target_flat, dim=1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_fcn_withsigmoid = nn.BCELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, label, gt_mask=None):
        if len(label.shape) != 4:
            label = label.unsqueeze(1)

        if gt_mask is not None:
            pred = torch.sigmoid(-pred) * gt_mask
            label = (1 - label) * gt_mask
            loss = self.loss_fcn_withsigmoid(pred, label)
        else:
            loss = self.loss_fcn(pred, label)
            pred = torch.sigmoid(pred)
            
        p_t = label * pred + (1 - label) * (1 - pred)
        alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
class SSLoss(nn.Module):
    '''Connectivity'''
    def __init__(self, kernel_size=5, only_one=False):
        super(SSLoss, self).__init__()
        if only_one is False:
            if kernel_size == 5:
                self.kernel = torch.tensor(
                       [[1, 0, 1, 0, 1],
                        [0, 1, 1, 1, 0],
                        [1, 1, 0, 1, 1],
                        [0, 1, 1, 1 ,0],
                        [1, 0, 1, 0, 1],
                        ],dtype=torch.float32
                        ).view(1, 1, 5, 5)  # Shape: (1, 1, 5, 5)
                self.padding = 2

            elif kernel_size == 3:
                self.kernel = torch.tensor(
                   [[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    ],dtype=torch.float32
                    ).view(1, 1, 3, 3)  # Shape: (1, 1, 3, 3)
                self.padding = 1
            
            elif kernel_size == 7:
                self.kernel = torch.tensor(
                       [[1, 0, 0, 1, 0, 0, 1],
                        [0, 1, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1 ,1, 1],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 0, 1, 0 ,1, 0],
                        [1, 0, 0, 1, 0, 0, 1]
                        ],dtype=torch.float32
                        ).view(1, 1, 7, 7)  # Shape: (1, 1, 7, 7)
                self.padding = 3

            elif kernel_size == 9:
                self.kernel = torch.tensor(
                       [[1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0 ,1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1 ,1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1]
                        ],dtype=torch.float32
                        ).view(1, 1, 9, 9)  # Shape: (1, 1, 9, 9)
                self.padding = 4

            else:
                raise ValueError(f"kernel_size must be 3,5,7,9 , got {kernel_size}")
        
        else:
            self.kernel = torch.ones(kernel_size, kernel_size).view(1, 1, kernel_size, kernel_size)
            self.kernel[0, 0, kernel_size//2, kernel_size//2] = 0
            self.padding = kernel_size // 2
            
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        # output: 模型的输出，经过sigmoid处理后的概率值
        # target: 真实标签，0或1
        if len(target.shape) != 4:
            target = target.unsqueeze(1)

        self.kernel = self.kernel.to(output.device)
        # connectivity_output = output
        output = torch.sigmoid(output)
        connectivity_output = nn.functional.conv2d(output, self.kernel, padding=self.padding) * target
        connectivity_label = nn.functional.conv2d(target, self.kernel, padding=self.padding) * target
        loss = self.loss(connectivity_output, connectivity_label)

        return loss
    
class RepulsionLoss(nn.Module):
    def __init__(self, alpha=5.0):
        super(RepulsionLoss, self).__init__()        
        self.alpha = alpha

    def forward(self, pred_points):
        """
        pred_points: Tensor [batch_size, num_queries, 2], 坐标范围 [0,1]
        """
        bs, num_queries, _ = pred_points.shape

        # 两两点差: [bs, num_queries, num_queries, 2]
        diff = pred_points[:, :, None, :] - pred_points[:, None, :, :]
        dist = torch.norm(diff, dim=-1)  # [bs, num_queries, num_queries]

        # 避免自己和自己 (对角线置为 inf)
        mask = torch.eye(num_queries, device=pred_points.device).bool()
        dist = torch.where(mask[None, :, :], torch.tensor(float('inf'), device=dist.device), dist)

        # repulsion: exp(-alpha * d)，d越小越大惩罚
        repulsion = torch.exp(-self.alpha * dist)

        # print(repulsion)

        # 平均化
        loss = repulsion.mean()
        return loss 
