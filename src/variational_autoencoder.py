import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
    device = mu.device
    noise = torch.randn_like(mu).to(device)
    sigma = torch.exp(logvar) ** 0.5
    z = mu + (sigma*noise)
    return z

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x): return x + self.block(x)  # skip connection


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_layer = nn.Linear(dim, dim)
        self.k_layer = nn.Linear(dim, dim)
        self.v_layer = nn.Linear(dim, dim)

    def forward(self, x):
        B, HW, C = x.shape
        Q, K, V = self.q_layer(x), self.k_layer(x), self.v_layer(x)
        head_dim = (B, HW, self.n_heads, C//self.n_heads)
        Q_head, K_head, V_head = Q.view(head_dim), K.view(head_dim), V.view(head_dim) # (B, HW, n_heads, dim/nheads)
        permuted_seq = (0, 2, 1, 3)
        Q_head, K_head, V_head = Q_head.permute(permuted_seq), K_head.permute(permuted_seq), V_head.permute(permuted_seq) # (B, n_heads, HW, dim/nheads)
        attn_scores = (Q_head @ K_head.transpose(-2, -1)) / (C // self.n_heads) ** 0.5   # (B, n_heads, HW, HW)
        attn_weights = F.softmax(attn_scores, dim=-1)                                   # (B, n_heads, HW, softmax(HW))
        attn_out = attn_weights @ V_head                                            # ((B, n_heads, HW, 32))
        output = attn_out.permute(0, 2, 1, 3).contiguous().view(B, HW, C)
        assert output.shape == (B, HW, C)
        return output, attn_weights

class SelfAttention2D(nn.Module):
    def __init__(self, dim:int=256, n_heads:int=8, num_groups:int=16):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.multihead_attn = MultiHeadAttention(dim=dim, n_heads=n_heads)

    def forward(self, x:torch.Tensor): 
        _x = x
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        attn_out, _ = self.multihead_attn(x)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        assert attn_out.shape == (B, C, H, W)
        return _x + attn_out

    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, input_height:int=32, input_width:int=32, n_heads:int=8, num_groups:int=16) -> None:
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=2, padding=1), nn.ReLU(),
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3,
                      stride=1, padding=1), nn.ReLU(), 

            ResBlock(out_channels*2),

            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=3,
                                  stride=2, padding=1), nn.ReLU(), 

            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*8, kernel_size=3,
                                  stride=1, padding=1), nn.ReLU(), 

            SelfAttention2D(dim=out_channels*8, n_heads=n_heads, num_groups=num_groups),
            ResBlock(out_channels*8)
        )


        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, in_channels, input_height, input_width)
        #     dummy_output = self.conv_layer(dummy_input)
        #     flatten_dim = dummy_output.numel() 

        # # self.mu_head = nn.Linear(flatten_dim, latent_dim)
        # # self.logvar_head = nn.Linear(flatten_dim, latent_dim)

        self.mu_head = nn.Conv2d(in_channels=out_channels*8, out_channels=8, kernel_size=3,
                                 stride=1, padding=1)

        self.logvar_head = nn.Conv2d(in_channels=out_channels*8, out_channels=8, kernel_size=3,
                                         stride=1, padding=1)
        
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x -> (Batch, in_chanels, height, width)
        b_dim = x.shape[0]
        conv_out = self.conv_layer(x)
        # flatten_out = conv_out.view(b_dim, -1)
        mu, logvar = self.mu_head(conv_out), self.logvar_head(conv_out)
        # mu, logvar = torch.clamp(mu, -10, 10), torch.clamp(logvar, -4, 4)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_channel:int=256, out_channels:int=3, latent_dim:int=16, spatial_size:tuple=(8, 8), n_heads:int=8, num_groups:int=16) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.spatial_size = spatial_size
        self.project = nn.Conv2d(in_channels=8, out_channels=in_channel, kernel_size=1)

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=3,
                            stride=1, padding=1), nn.ReLU(), ResBlock(in_channel//2),
            SelfAttention2D(dim=in_channel//2, n_heads=n_heads, num_groups=num_groups),
            nn.ConvTranspose2d(in_channel//2, in_channel//4, kernel_size=3,
                            stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channel//4, in_channel//8, kernel_size=3,
                                        stride=1, padding=1), nn.ReLU(), ResBlock(in_channel//8),
            nn.ConvTranspose2d(in_channel//8, out_channels, kernel_size=3,
                                        stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z = self.project(z)
        return self.conv_layer(z)

class VAE(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, latent_dim:int, input_size:tuple) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, latent_dim,
                                input_height=input_size[0], input_width=input_size[0])
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            enc_out = self.encoder.conv_layer(dummy)
            self.encoded_shape = enc_out.shape[1:]  # (C, H, W)
        self.decoder = Decoder(out_channels*2, in_channels, latent_dim, spatial_size=(self.encoded_shape[1], self.encoded_shape[2]))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # mu: (batch_size, latent_dim), logvar: sigma^2 : (batch_size, latent_dim)
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recons_x = self.decoder(z)
        return recons_x, mu, logvar

if __name__ == '__main__':
    vae = VAE(3, 32, 16, (32, 32))
    x = torch.randn(size=(4, 3, 32, 32))
    recons_x, mu, logvar = vae(x)
    print(f"reconsx shape: {recons_x.shape}, mu shape: {mu.shape}, logvar shape: {logvar.shape}")