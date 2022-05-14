import torch
from torch import nn
import numpy as np
import math 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                # Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.depth = depth
        self.dim = dim
        self.rho_sfmx = math.sqrt(dim - 1) / (self.dim * math.sqrt(dim_head))
        self.q_sn = [None] * depth
        self.k_sn = [None] * depth
        self.v_sn = [None] * depth
        self.w_sn = [None] * depth
        self.wp_sn = [None] * depth
        self.q_fn = [None] * depth
        self.k_fn = [None] * depth
        self.v_fn = [None] * depth
        self.w_fn = [None] * depth
        self.wp_fn = [None] * depth
        self.spectral_norm_precompute = None
        self.spectral_precompute = None

    # TODO: Is it ok to do the spectral of the concatted qkv matrix? Or have to seperate?
    def spectral_clipping(self, clip_caps): # clip_percent
        with torch.no_grad():
            for cap, module in zip(clip_caps, self.transformer.layers):
                # np.linalg.norm(x, ord=2)
                module[0].fn.to_qkv.weight[:] = cap * module[0].fn.to_qkv.weight / np.linalg.norm(module[0].fn.to_qkv.weight.cpu(), ord=2)
                module[1].fn.net[0].weight[:] = cap * module[1].fn.net[0].weight / np.linalg.norm(module[1].fn.net[0].weight.cpu(), ord=2)
                module[1].fn.net[3].weight[:] = cap * module[1].fn.net[3].weight / np.linalg.norm(module[1].fn.net[3].weight.cpu(), ord=2)

    def update_spectral_terms(self):
        with torch.no_grad():
            self.spectral_precompute = 1
            self.spectral_norm_precompute = 0
            for i, module in enumerate(self.transformer.layers):
                qkv = module[0].fn.to_qkv.weight.chunk(3, dim = -1)
                # Spectral norms
                self.q_sn[i] = np.linalg.norm(qkv[0].cpu(), ord=2)
                self.k_sn[i] = np.linalg.norm(qkv[1].cpu(), ord=2)
                self.v_sn[i] = np.linalg.norm(qkv[2].cpu(), ord=2)
                self.w_sn[i] = np.linalg.norm(module[1].fn.net[0].weight.cpu(), ord=2)
                self.wp_sn[i] = np.linalg.norm(module[1].fn.net[3].weight.cpu(), ord=2)
                # Frobenius norms
                self.q_fn[i] = np.linalg.norm(qkv[0].cpu(), ord=None)
                self.k_fn[i] = np.linalg.norm(qkv[1].cpu(), ord=None)
                self.v_fn[i] = np.linalg.norm(qkv[2].cpu(), ord=None)
                self.w_fn[i] = np.linalg.norm(module[1].fn.net[0].weight.cpu(), ord=None)
                self.wp_fn[i] = np.linalg.norm(module[1].fn.net[3].weight.cpu(), ord=None)
                self.spectral_norm_precompute += ((self.q_sn[i]/self.q_fn[i])**(2.0/3) + (self.k_sn[i]/self.k_fn[i])**(2.0/3) + (self.v_sn[i]/self.v_fn[i])**(2.0/3) + 
                                                (self.w_sn[i]/self.w_fn[i])**(2.0/3) + (self.wp_sn[i]/self.wp_fn[i])**(2.0/3))
                self.spectral_precompute *= (self.q_sn[i] * self.k_sn[i] * self.v_sn[i] * self.w_sn[i] * self.wp_sn[i] * self.rho_sfmx) ** (2 ** (self.depth - i))

    # Must be batch size 1
    def spectral_complexity(self, x):
        with torch.no_grad():
            x = torch.squeeze(self.to_patch_embedding(x))
            xsn = np.linalg.norm(x.cpu(), ord=2)
            xfn = np.linalg.norm(x.cpu(), ord=None)
            spectral_term = self.spectral_precompute * xfn * (xsn ** (2 ** self.depth))
            spectral_norm_term = self.spectral_norm_precompute + self.depth * 2 * (xfn/xsn)**(2.0/3)
            spectral_norm_term = spectral_norm_term ** (3.0/2)
            spectral_complex = spectral_term * spectral_norm_term
            return spectral_complex

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)