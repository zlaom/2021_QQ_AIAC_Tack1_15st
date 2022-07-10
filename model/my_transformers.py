from logging import setLoggerClass
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # Resudual connect: fn(x) + x
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # using Layer Normalization before input to fn layer
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # Feed Forward Neural Network
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Two linear network with GELU and Dropout
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# Multi-Self-Attention Layer
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        # get q,k,v from a single weight matrix
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):

        # x:[batch_size, patch_num, pathch_embedding_dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([batch, patch_num, head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split q,k,v from [batch, patch_num, head_num*head_dim] -> [batch, head_num, patch_num, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [batch, head_num, patch_num, patch_num]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # mask value: -inf
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, None, :] * mask[:, None, :, None]
            mask = mask.bool()
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)

        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # cat all output -> [batch, patch_num, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Linear + Dropout
        out = self.to_out(out)

        # out: [batch, patch_num, embedding_dim]
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # using multi-self-attention and feed forward neural network repeatly
            self.layers.append(
                nn.ModuleList([
                    Residual(
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                               dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class FrameFeatureTrs2(nn.Module):
    def __init__(self,
                 max_frame=32,
                 frame_dim=1536,
                 dim=1536,
                 depth=6,
                 heads=12,
                 mlp_dim=512,
                 pool='cls',
                 dim_head=128,
                 reduce=False,
                 num_classes=10000,
                 pos_emb='global',
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        assert pool in {'cls',
                        'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.max_frame = max_frame
        self.patch_to_embedding = nn.Linear(frame_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.pos_emb = pos_emb
        self.reduce = reduce
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        if self.pos_emb == 'global':
            self.pos_embedding = nn.Parameter(torch.randn(1, max_frame + 1, dim))
        elif self.pos_emb == 'conv':
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6, 1), padding='same')

    def forward(self, frame_feature, mask=None):
        # embedding every patch vector to embedding size: [batch, frame_num, embedding_size]
        x = frame_feature[:, :self.max_frame, :]
        # if self.reduce:
        #     x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # repeat class token to batch_size and cat to x
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        # NOTES: position embedding is random initialized and learnable
        if self.pos_emb == 'global':
            x += self.pos_embedding[:, :(n + 1)]
        elif self.pos_emb == 'conv':
            x = x.unsqueeze(1)
            x = x + self.conv(x)
            x = x.squeeze()

        x = self.dropout(x)

        # transformer: x[batch, patch_num + 1, embedding_size] -> x[batch, patch_num + 1, embedding_size]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        # mean: using all tokens' mean value
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # Identity layer
        x = self.to_latent(x)

        # MLP classification layer
        # return self.mlp_head(x), x
        return x


class FrameFeatureTrs(nn.Module):
    def __init__(self,
                 max_frame=32,
                 frame_dim=1536,
                 dim=1536,
                 depth=6,
                 heads=12,
                 mlp_dim=512,
                 pool='cls',
                 dim_head=128,
                 reduce=False,
                 num_classes=10000,
                 pos_emb='conv',
                 dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        assert pool in {'cls',
                        'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.max_frame = max_frame
        self.patch_to_embedding = nn.Linear(frame_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.pos_emb = pos_emb
        self.reduce = reduce
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        if self.pos_emb == 'global':
            self.pos_embedding = nn.Parameter(torch.randn(1, max_frame + 1, dim))
        elif self.pos_emb == 'conv':
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6, 1), padding='same')
        elif self.pos_emb == 'all':
            self.pos_embedding = nn.Parameter(torch.randn(1, max_frame + 1, dim))
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6, 1), padding='same')

    def forward(self, frame_feature, mask=None):
        # embedding every patch vector to embedding size: [batch, frame_num, embedding_size]
        x = frame_feature[:, :self.max_frame, :]
        if self.reduce:
            x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # repeat class token to batch_size and cat to x
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        # NOTES: position embedding is random initialized and learnable
        if self.pos_emb == 'global':
            x += self.pos_embedding[:, :(n + 1)]
        elif self.pos_emb == 'conv':
            x = x.unsqueeze(1)
            x = x + self.conv(x)
            x = x.squeeze()
        elif self.pos_emb == 'all':
            x = x.unsqueeze(1)
            x = x + self.conv(x)
            x = x.squeeze()
            x += self.pos_embedding[:, :(n + 1)]

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)

        x = self.dropout(x)

        # transformer: x[batch, patch_num + 1, embedding_size] -> x[batch, patch_num + 1, embedding_size]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        # mean: using all tokens' mean value
        _x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # Identity layer
        # x = self.to_latent(x)

        # MLP classification layer
        # return self.mlp_head(x), x
        return x, _x


if __name__ == "__main__":
    model = FrameFeatureTrs(num_classes=10000,
                            max_frame=32,
                            frame_dim=1536,
                            dim=1536,
                            heads=16,
                            dim_head=64)
    frame_feature_batch = torch.randn((10, 32, 1536))
    mask = torch.randn((10, 32)) > 0.5
    embedding = model(frame_feature_batch, mask)
    print(embedding.shape)