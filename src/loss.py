import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models

class PerceptualLoss(nn.Module): 
    def __init__(self):
        super().__init__()      
        
        weights = models.VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=weights)

        for param in vgg16.parameters():
            param.requires_grad = False

        slices = [3, 8, 15]
        self.slice_1 = nn.Sequential(*list(vgg16.features.children())[: slices[0]+1])
        self.slice_2 = nn.Sequential(*list(vgg16.features.children())[slices[0]+1:slices[1]+1])
        self.slice_3 = nn.Sequential(*list(vgg16.features.children())[slices[1]+1:slices[2]+1])

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x_out1, xhat_out1 = self.slice_1(x), self.slice_1(x_hat)
        x_out2, xhat_out2 = self.slice_2(x_out1), self.slice_2(xhat_out1)
        x_out3, xhat_out3 = self.slice_3(x_out2), self.slice_3(xhat_out2)

        loss1 = F.mse_loss(x_out1, xhat_out1)
        loss2 = F.mse_loss(x_out2, xhat_out2)
        loss3 = F.mse_loss(x_out3, xhat_out3)

        return loss1 + loss2 + loss3



def kl_loss(mu, log_var, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()

def elbo(ImgTrue, ImgPred, mu, log_var, loss_fn:str, perceptual_fn: callable , verbose:bool=False,
         kl_weight:float=1.0, pecerptual_weight:float=0.001):
    N = mu.shape[0]
    ImgTrue_scaled = ImgTrue * 2 - 1
    ImgPred_scaled = ImgPred * 2 - 1
    if loss_fn == "bce":recons = F.binary_cross_entropy(ImgPred, ImgTrue)
    elif loss_fn == "mse": recons = F.mse_loss(ImgPred, ImgTrue) 
    else: raise ValueError(f"Unknown loss_fn: {loss_fn}")
    kl = kl_loss(mu, log_var, free_bits=0.5)# -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var)) /  N
    ploss = perceptual_fn(ImgTrue_scaled, ImgPred_scaled)
    if verbose: print(f"Recons: {recons.item():.4f} | KL: {kl.item():.4f} , PLOSS : {(pecerptual_weight*ploss).item():.4f}")
    return recons + pecerptual_weight * ploss +  kl * kl_weight