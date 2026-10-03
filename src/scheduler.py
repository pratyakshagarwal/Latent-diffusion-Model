import torch

class NoiseScheduler:
    def __init__(self, T=1000, schedule="linear", beta_start=0.0001, beta_end=0.02):
        self.T = T
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, T)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        elif schedule == "cosine":
            s = 0.008
            t = torch.arange(0, T+1)
            alpha_bars = torch.cos(((t/T + s) / (1 + s)) * torch.pi / 2) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            self.betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(0, 0.999)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x0, t, epsilon):
        # t is an integer or batch of integers
        device = x0.device
        alpha_bar_t = self.alpha_bars.to(t.device)[t]
        
        # reshape for broadcasting with image dimensions (B, C, H, W)
        while alpha_bar_t.dim() < x0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return xt

if __name__ == '__main__':pass