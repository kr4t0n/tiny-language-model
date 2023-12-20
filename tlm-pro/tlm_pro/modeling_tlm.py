import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # register rotary matrix and attention mask
        self.register_buffer(
            "R",
            get_rotary_matrix(context_length, self.d_head),
        )
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, context_length, d_model
        B, C, D = x.size()

        q, k, v = self.qkv(x).split(self.d_model, dim=-1)

        # make q, k, v into batch_size, n_heads, context_length, d_heads
        q = q.view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, C, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, C, self.n_heads, self.d_head).transpose(1, 2)

        # rotate q, k
        q_rotated = torch.einsum("bhci,cij->bhcj", q, self.R)
        k_rotated = torch.einsum("bhci,cij->bhcj", k, self.R)

        # calculate attention
        attn_numerator = torch.exp((q_rotated @ k_rotated.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head)))
        attn_denominator = torch.exp((q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head)))
        attn_denominator = torch.sum(attn_denominator, dim=-1, keepdim=True)

        attn = attn_numerator / attn_denominator

        # mask attention to make it causal
        attn = attn.masked_fill(self.attn_mask[:, :, :C, :C] == 0, 0.0)

        # dropout
        attn = self.dropout(attn)

        # batch_size, n_heads, context_length, d_heads
        y = attn @ v

        # re-assemble all heads
        y = y.transpose(1, 2).contiguous().view(B, C, D)

        # out projection
        y = self.out(y)

        # dropout
        y = self.dropout(y)

        return y


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.fc = nn.Linear(d_model, 4 * d_model)
        self.act = SwiGLU(4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class TLMBlock(nn.Module):
    def __init__(
        self,
        context_length: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.attn_rms = RMSNorm((context_length, d_model))
        self.attn = RoPECausalAttention(context_length, d_model, n_heads, dropout=dropout)
        self.ffn_rms = RMSNorm((context_length, d_model))
        self.ffn = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-normalizatoin
        x = self.attn_rms(x)

        # attention and skip connection
        x = x + self.attn(x)

        # pre-normalization
        x = self.ffn_rms(x)

        # ffn and skip connection
        x = x + self.ffn(x)

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

        self.tlm = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, d_model),
                h=nn.ModuleList([TLMBlock(context_length, d_model, n_heads, dropout=dropout) for _ in range(n_layers)]),
                ln_f=RMSNorm((context_length, d_model)),
            )
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        print(f"number of parameteres: {(self._count_params() / 1e9):.2f}B.")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tlm.wte(x)
        for block in self.tlm.h:
            x = block(x)
        x = self.tlm.ln_f(x)
        x = self.lm_head(x)

        return x

    def loss_fn(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yhat = yhat.view(-1, yhat.size(-1))
        y = y.view(-1)

        loss = F.cross_entropy(yhat, y, ignore_index=-100)

        return loss

    def generate(self, text: str, tokenizer: PreTrainedTokenizerBase, max_length: int, device: str) -> str:
        input_ids = tokenizer(text)["input_ids"]

        while len(input_ids) < max_length:
            model_input_ids = tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"].unsqueeze(dim=0)

            outputs = self.forward(model_input_ids.to(device)).detach()
            outputs = outputs.argmax(dim=-1)[0]

            last_output = outputs[len(input_ids) - 1]
            input_ids.append(last_output.item())

            if last_output == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids)
