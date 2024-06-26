import torch
from torch.nn import functional as F

def compute_class_weight(target):
    n, H, W = target.size()
    class_weights = torch.zeros(n, H, W).to(target.device)
    max_value = int(target.max().item())
    for i in range(max_value + 1):
        mask = (target == i).float()
        class_weight = 1.0 / (mask.sum() + 1e-6)
        class_weights += mask * class_weight
    return class_weights

def Dice_Loss_Coefficient(pred, target, smooth=1.):
    n, c, H, W = pred.shape  # pred의 차원 정보를 가져옵니다.

    # target을 pred의 크기인 H와 W로 리사이징
    #target_resized = F.interpolate(target.float().unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
    #target_resized = target_resized.unsqueeze(1).expand(n, c, H, W)  # [n, c, H, W]로 확장

    # 클래스 가중치 계산을 위해 리사이징된 target 사용
    target = target.long()
    if target.dim() == 4:
          target = target.squeeze(1)
    target = target.expand(n, H, W)
    weights = compute_class_weight(target)
    #weights = compute_class_weight(target_resized[:, 0, :, :])  # 원래 target 대신 리사이징된 target 사용
    weights = weights.unsqueeze(1).expand(n, c, H, W)  # [n, c, H, W]로 확장
    target = target.unsqueeze(1).expand(n, c, H, W)
    intersection = (pred * target * weights).sum(dim=2).sum(dim=2)
    union = (weights * pred).sum(dim=2).sum(dim=2) + (weights * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (union + smooth)))

    return loss.mean()

class Weighted_Cross_Entropy_Loss(torch.nn.Module):

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target):
      n, c, H, W = pred.shape

    # 클래스 레이블 검증 (0부터 c-1까지)
      if (target.max() >= c) or (target.min() < 0):
          print("target tensor contains out-of-range values")
          target.clamp_(0, c-1)  # 잘못된 값 조정
      if target.dim() == 4:
          target = target.squeeze(1)

      #target = F.interpolate(target.float().unsqueeze(1), size=(H, W), mode='nearest').unsqueeze(1)
      target = target.long()

      target = target.expand(n, H, W)
      weights = compute_class_weight(target)
      target = target.unsqueeze(1)
      logp = F.log_softmax(pred, dim=1)
      logp = torch.gather(logp, 1, target)
      weighted_logp = (logp * weights.unsqueeze(1)).view(n, -1)
      weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)
      weighted_loss = -weighted_loss.mean()

      return weighted_loss


class BU_Net_Loss(torch.nn.Module):
    def __init__(self):
        super(BU_Net_Loss, self).__init__()
        self.cross_entropy_loss = Weighted_Cross_Entropy_Loss()

    def forward(self, pred, target):
        wce_loss = self.cross_entropy_loss(pred, target)
        dice_loss = Dice_Loss_Coefficient(pred, target)
        total_loss = wce_loss + dice_loss
        return total_loss
