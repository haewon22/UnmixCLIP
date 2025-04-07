import torch
import torch.nn as nn
import torch.nn.functional as F

class MFILoss(nn.Module):
   def __init__(self, lambda_: float):
       """
       :param lambda_: Coefficient for the off-diagonal (mutual information) term.
       """
       super().__init__()
       self.lambda_ = lambda_

   def forward(self, t_prime: torch.Tensor) -> torch.Tensor:
       """
       :param t_prime: (B, N, D) shaped text embedding tensor
       :return: Scalar tensor representing the MFI loss
       """
       # (1) Normalize with L2 norm -> (B, N, D)
       t_norm = t_prime / t_prime.norm(dim=2, keepdim=True)
      
       # (2) Batch matrix multiply -> (B, N, N) similarity matrix S
       S = torch.bmm(t_norm, t_norm.transpose(1, 2))

       # (3) Diagonal term (Sᵢᵢ - 1)²
       diag_S = torch.diagonal(S, dim1=1, dim2=2)  # (B, N)
       collapse_term = ((diag_S - 1)**2).sum(dim=1)  # (B,)

       # (4) Off-diagonal (i ≠ j) term -> total S^2 minus diagonal squares
       S_sqr_sum = (S**2).sum(dim=(1, 2))           # (B,)
       diag_sqr_sum = (diag_S**2).sum(dim=1)       # (B,)
       off_diag_sqr = S_sqr_sum - diag_sqr_sum

       # (5) Batch-wise loss -> then mean
       batch_loss = collapse_term + self.lambda_ * off_diag_sqr  # (B,)
       loss = batch_loss.mean()  # usually batch mean
      
       return loss

class AsymmetricLoss(nn.Module):
   def __init__(self, gamma_neg=2, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
       super(AsymmetricLoss, self).__init__()

       self.gamma_neg = gamma_neg
       self.gamma_pos = gamma_pos
       self.clip = clip
       self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
       self.eps = eps

   def forward(self, x, y):
       """"
       Parameters
       ----------
       x: input logits
       y: targets (multi-label binarized vector)
       """

       # Calculating Probabilities
       x_sigmoid = torch.sigmoid(x)
       xs_pos = x_sigmoid
       xs_neg = 1 - x_sigmoid

       # Asymmetric Clipping
       if self.clip is not None and self.clip > 0:
           xs_neg = (xs_neg + self.clip).clamp(max=1)

       # Basic CE calculation
       los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
       los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
       loss = los_pos + los_neg

       # Asymmetric Focusing
       if self.gamma_neg > 0 or self.gamma_pos > 0:
           if self.disable_torch_grad_focal_loss:
               torch.set_grad_enabled(False)
           pt0 = xs_pos * y
           pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
           pt = pt0 + pt1
           one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
           one_sided_w = torch.pow(1 - pt, one_sided_gamma)
           if self.disable_torch_grad_focal_loss:
               torch.set_grad_enabled(True)
           loss *= one_sided_w

       return -loss.sum()
