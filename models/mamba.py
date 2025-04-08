from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model).to(device)
        self.layers = nn.ModuleList([ResidualBlock(args).to(device) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model).to(device)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False).to(device)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

        # self.output_layer = nn.Linear(args.vocab_size, 2)

        self.criteration = nn.CrossEntropyLoss()
    # def forward(self, input_ids):
    def forward(self, input_ids, labels, attention_mask, token_type_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)
        """
        x = self.embedding(input_ids).to(device)

        for layer in self.layers:
            x = layer(x).to(device)

        x = self.norm_f(x).to(device)
        x = self.lm_head(x).to(device)

        logits = self.output_layer(x).to(device)

        loss = self.criteration(logits, labels)

        return loss,logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu')

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args).to(device)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args).to(device)
        self.norm = RMSNorm(args.d_model).to(device)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)
        """
        output = self.mixer(self.norm(x).to(device)).to(device) + x.to(device)

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias).to(device)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        ).to(device)

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False).to(device)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True).to(device)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner).to(device)
        self.A_log = nn.Parameter(torch.log(A).to(device))
        self.D = nn.Parameter(torch.ones(args.d_inner).to(device))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias).to(device)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x).to(device)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l').to(device)
        x = self.conv1d(x)[:, :, :l].to(device)
        x = rearrange(x, 'b d_in l -> b l d_in').to(device)

        x = F.silu(x).to(device)

        y = self.ssm(x).to(device)

        y = y * F.silu(res).to(device)

        output = self.out_proj(y).to(device)

        return output

    def ssm(self, x):
        """Runs the SSM.

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)
        """
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float()).to(device)  # shape (d_in, n)
        D = self.D.float().to(device)

        x_dbl = self.x_proj(x).to(device)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta)).to(device)  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B.to(device), C.to(device), D).to(device)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm.

        Args:
            u: shape (b, l, d_in)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n')).to(device)
        deltaB_u = einsum(delta.to(device), B.to(device), u.to(device), 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i].to(device) * x + deltaB_u[:, i].to(device)
            y = einsum(x, C[:, i, :].to(device), 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1).to(device)  # shape (b, l, d_in)

        y = y + u.to(device) * D.to(device)

        return y

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.to(device)

        return output

#Simple, minimal implementation of Mamba in one file of PyTorch.
