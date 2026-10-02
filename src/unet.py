import torch
import torch.nn as nn
from src.embeddings import TimeEmbedding

def get_groups(channels):
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, time_dim, num_groups=2):
        super().__init__()
        # block1: conv → groupnorm → silu
        self.block_1 = nn.Sequential(
            nn.GroupNorm(get_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=kernel_size, padding=1),
        )
        # block2: conv → groupnorm → silu
        self.block_2 = nn.Sequential(
            nn.GroupNorm(get_groups(out_channels), out_channels ),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                  kernel_size=kernel_size, padding=1),
        )
        # time_mlp: linear to project time embedding to out_channels
        self.mlp = nn.Linear(time_dim, out_channels)
        # residual_conv: 1x1 conv if in_channels != out_channels, else identity
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        
    def forward(self, x, t):
        org_x = x
        # 1. pass x through block1
        x = self.block_1(x)
        # 2. project t embedding and add to output
        mapped_embeddings = self.mlp(t).unsqueeze(-1).unsqueeze(-1)    
        x = x+mapped_embeddings
        # 3. pass through block2
        x = self.block_2(x)
        # 4. add residual connection
        return self.residual_conv(org_x) + x

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # normalize
        x_norm = self.norm(x)
        # reshape to sequence: (B, H*W, C)
        x_flat = x_norm.reshape(B, C, H*W).transpose(1, 2)
        # self attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        # reshape back: (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        # residual connection
        return x + attn_out

class Unet(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, time_dim, num_groups:int=2, n_class:int=11):
        super().__init__()
        # Encoder Part ResBlocks
        self.encoder_resblock_1 = ResBlock(in_channels, out_channel, kernel_size, time_dim, num_groups)
        self.encoder_resblock_2 = ResBlock(out_channel, out_channel*2, kernel_size, time_dim, num_groups)
        # Encoder Downsamplers
        self.downsampler1 = nn.MaxPool2d(2)
        self.downsampler2 = nn.MaxPool2d(2)

        # BottleNeck:
        self.bottleneck = ResBlock(out_channel*2, out_channel*2, kernel_size, time_dim, num_groups)
        self.attn_ = SelfAttention(channels=out_channel*2)
        # Decoder Resblocks
        self.decoder_resblock_1 = ResBlock(out_channel*4, out_channel, kernel_size, time_dim, num_groups)
        self.decoder_resblock_2 = ResBlock(out_channel*2, out_channel, kernel_size, time_dim, num_groups)
        # Decoder Upsamplers
        self.upsampler1 = nn.Upsample(scale_factor=2)
        self.upsampler2 = nn.Upsample(scale_factor=2)

        # Embedding layer
        self.embeddings = TimeEmbedding(dim=time_dim)
        self.cfg_embeddings = nn.Embedding(n_class,time_dim)
        # output layer
        self.output_layer = nn.Conv2d(out_channel, in_channels, kernel_size=1)

    def forward(self, x, t, c):
        # 1. time embedding
        t_emb = self.embeddings(t)
        c_emb = self.cfg_embeddings(c)
        t_emb = t_emb + c_emb
        # 2. encoder block 1 + save skip1
        ex1 = self.encoder_resblock_1(x, t_emb)
        # 3. downsample
        down_ex1 = self.downsampler1(ex1)
        # 4. encoder block 2 + save skip2
        ex2 = self.encoder_resblock_2(down_ex1, t_emb)
        # 5. downsample
        down_ex2 = self.downsampler2(ex2)

        # 6. bottleneck
        btnx = self.bottleneck(down_ex2, t_emb)
        btnx = self.attn_(btnx)

        # 7. upsample
        up_dx1 = self.upsampler1(btnx)
        # 8. concat skip2 + decoder block 1
        cat1 = torch.cat([up_dx1, ex2], dim=1)
        dx1 = self.decoder_resblock_1(cat1, t_emb) 
        # 9. upsample
        up_dx2 = self.upsampler2(dx1)
        # 10. concat skip1 + decoder block 2
        cat2 = torch.cat([up_dx2, ex1], dim=1)
        dx2 = self.decoder_resblock_2(cat2, t_emb)

        # 11. output conv
        out = self.output_layer(dx2)
        return out

if __name__ == '__main__':pass
