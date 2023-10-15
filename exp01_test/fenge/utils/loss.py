import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils.metrics import *


class CEdice(_Loss):
    def __init__(self,num_classes):
        super(CEdice, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true, loss_weight=None, dice: bool = True):
        loss = nn.functional.cross_entropy(y_pred, y_true, weight=loss_weight)
        if dice is True:
            dice_target = nn.functional.one_hot(y_true, self.num_classes).float()
            dice_target = dice_target.permute(0, 3, 1, 2)
            loss += dice_loss(y_pred, dice_target, multiclass=True)
        losses = loss
        return losses

def dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target)

class BinarySoftDiceLoss(_Loss):

    def __init__(self):
        super(BinarySoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_dice = diceCoeffv2(y_pred, y_true)
        return 1 - mean_dice


class SoftDiceLoss(_Loss):

    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        # 从1开始排除背景，前提是颜色表palette中背景放在第一个位置 [[0], ..., ...]
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))###之前是diceCoeffv2
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


class SoftDiceLossV2(_Loss):
    def __init__(self, num_classes, weight=[0.73, 0.73, 0.69, 0.93, 0.92], reduction="sum"):
        super(SoftDiceLossV2, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred, y_true):
        class_loss = []
        for i in range(1, self.num_classes):
            dice = diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :])
            class_loss.append((1-dice) * self.weight[i-1])
        if self.reduction == 'mean':
            return sum(class_loss) / len(class_loss)
        elif self.reduction == 'sum':
            return sum(class_loss)
        else:
            raise NotImplementedError("no such reduction.")


class BinaryTverskyLoss(_Loss):
    def __init__(self, alpha=0.7):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        mean_tl = tversky(y_pred, y_true, alpha=self.alpha)
        return 1 - mean_tl


class WBCELoss(_Loss):
    def __init__(self, num_classes,  smooth=0, size=None, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), reduction='mean', ignore_index=0):
        super(WBCELoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weights = (1,1)
        # if weight:
        #     weights = []
        #     w = torch.ones([1, size, size])
        #     for v in weight:
        #         weights.append(w * v)
        #     self.weights = torch.cat(weights, dim=0)
        self.bce_loss = nn.BCELoss(self.weights, reduction, ignore_index)

    def forward(self, inputs, targets):

        return self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes)


class BCE_Dice_Loss(_Loss):
    def __init__(self, num_classes, smooth=0, weight=(1.0, 1.0)):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.weight = weight
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        return self.weight[0] * self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes) + self.weight[1] * self.dice_loss(inputs, targets)















