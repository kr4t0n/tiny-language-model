import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from collections import OrderedDict


def get_rotary_matrix(context_length: int, d_model: int) -> torch.Tensor:
    R = torch.zeros(
        (context_length, d_model, d_model),
        requires_grad=False,
    )

    for m in range(context_length):
        for i in range(d_model // 2):
            theta = 10000.0 ** (-2.0 * (i - 1) / d_model)
            m_theta = torch.tensor(m * theta)

            R[m, 2 * i, 2 * i] = torch.cos(m_theta)
            R[m, 2 * i, 2 * i + 1] = -torch.sin(m_theta)
            R[m, 2 * i + 1, 2 * i] = torch.sin(m_theta)
            R[m, 2 * i + 1, 2 * i + 1] = torch.cos(m_theta)

    return R


class RMSNorm(nn.Module):
    def __init__(
        self,
        features: Tuple[int, int],
        bias: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter("bias", None)

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms

        x = self.weight * x
        if self.bias is not None:
            x = x + self.bias

        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()

        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = self.w(x) * torch.sigmoid(self.beta * self.w(x))
        act = swish * self.v(x)

        return act


class RoPECausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, context_length, d_model
        B, C, D = x.size()

        q, k, v = self.qkv(x).split(self.d_model, dim=-1)

        # make q, k, v into batch_size, n_heads, context_length, d_heads
        q = q.view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, C, self.n_heads, self.d_head).transpose(1, 2)

        # get rotary matrix
        R = get_rotary_matrix(C, self.d_head).to(q.device)

        # rotate q, k
        q_rotated = torch.einsum("bhci,cij->bhcj", q, R)
        k_rotated = torch.einsum("bhci,cij->bhcj", k, R)

        # calculate attention
        attn_numerator = torch.exp((q_rotated @ k_rotated.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head)))
        attn_denominator = torch.exp((q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head)))
        attn_denominator = torch.sum(attn_denominator, dim=-1, keepdim=True)

        attn = attn_numerator / attn_denominator

        # mask attention to make it causal
        attn_mask = torch.tril(torch.ones(C, C)).view(1, 1, C, C).to(attn.device)
        attn = attn.masked_fill(attn_mask[:, :, :C, :C] == 0, 0.0)

        # batch_size, n_heads, context_length, d_heads
        y = attn @ v

        # re-assemble all heads
        y = y.transpose(1, 2).contiguous().view(B, C, D)

        # out projection
        y = self.out(y)

        return y


class TLMBlock(nn.Module):
    def __init__(
        self,
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.rms = RMSNorm((context_length, d_model))
        self.attn = RoPECausalAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            SwiGLU(d_model),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-normalizatoin
        x = self.rms(x)

        # attention and skip connection
        x = x + self.dropout(self.attn(x))

        # pre-normalization
        x = self.rms(x)

        # ffn, dropout and skip connection
        x = x + self.dropout(self.ffn(x))

        return x


class TLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"TLMBlock_{i}",
                        TLMBlock(
                            context_length,
                            d_model,
                            n_heads,
                            dropout=dropout,
                        ),
                    )
                    for i in range(n_layers)
                ]
            )
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            SwiGLU(d_model),
            nn.Linear(d_model, d_model),
            SwiGLU(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.out(x)

        return x

    def loss_fn(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yhat = yhat.view(-1, yhat.size(-1))
        y = y.view(-1)

        loss = F.cross_entropy(yhat, y, ignore_index=-100)

        return loss
