import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    # PE(t, 2i) = sin(t / 10000^(2i/d))
    # PE(t, 2i+1) = cos(t / 10000^(2i/d))
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim*2),
            nn.SiLU(),
            nn.Linear(in_features=dim*2, out_features=dim)
        )
    def forward(self, t):
        # t shape: (batch,)
        device = t.device
        half_dim = self.dim // 2
        i = torch.arange(0, half_dim, device=device) 
        div_term = 10000 ** (2 * i / self.dim)
        
        # t needs to be (batch, 1) to broadcast against div_term (half_dim,)
        t = t.unsqueeze(1).float()
        
        sins = torch.sin(t/div_term)
        coss = torch.cos(t/div_term)
        emb = torch.cat((sins, coss), dim=-1)
        return self.mlp(emb)  # shape: (batch, dim)

if __name__ == '__main__':pass