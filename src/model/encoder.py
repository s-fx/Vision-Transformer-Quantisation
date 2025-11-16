import torch
import torch.nn as nn
import math
from src.model.encoder_input_layer import EncoderInputLayer
from src.model.patch_embedding import PatchEmbedding


class LayerNormalization(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 eps: float = 1e-6
                ):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """
        normalize across the feature dimension for each individual example
        x: (B, num_tokens, embed_dim)
        out: (B, num_tokens, embed_dim)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0
                ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
                'Embedding dimension must be dividable by number of heads'
        # nn.Linear applies affine linear transformation to the incoming data
        # y = xA^T + b
        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(p=dropout_rate)
        self.proj_dropout = nn.Dropout(p=dropout_rate)

    @staticmethod
    def scaled_dot_attention(q, k, v, dropout: nn.Dropout = None):
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        attn = scores.softmax(dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, q, k, v):
        batch_size, num_tokens, embed_dim = q.shape

        # Linear projections
        Q = self.w_q(q).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Attention
        x, attn = self.scaled_dot_attention(Q, K, V, self.attention_dropout)

        # Combine heads
        x = x.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)

        # Final projection
        x = self.proj_dropout(self.w_o(x))
        return x, attn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff_scale: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff_scale * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff_scale * d_model, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout_rate, ff_dropout_rate, dff_scale: int = 4):
        super().__init__()
        self.normalisation_stage1 = LayerNormalization(embed_dim)
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, attn_dropout_rate)
        self.normalisation_stage2 = LayerNormalization(embed_dim)
        self.feed_forward_layer = FeedForwardLayer(embed_dim, dff_scale, ff_dropout_rate)

    def forward(self, x):
        residual1 = x
        x = self.normalisation_stage1(x)
        x, _ = self.mhsa(x, x, x)
        x = x + residual1
        residual2 = x
        ffn_out = self.feed_forward_layer(self.normalisation_stage2(x))
        x = ffn_out + residual2
        return x


class Encoder(nn.Module):
    def __init__(self, num_of_encoders, embeddings, dff_scale, num_heads, attn_dropout_rate, ff_dropout_rate):
        super().__init__()
        self.encoder_stack = nn.ModuleList(
            EncoderBlock(
                embed_dim=embeddings,
                num_heads=num_heads,
                dff_scale=dff_scale,
                attn_dropout_rate=attn_dropout_rate,
                ff_dropout_rate=ff_dropout_rate
            ) for _ in range(num_of_encoders)
        )

    def forward(self, x):
        for module in self.encoder_stack:
            x = module(x)
        return x

