import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel
import numpy as np
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel,MixerModel
from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf
from collections import namedtuple
import torch
from configs.config import MambaConfig
import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BidirectionalMambaModel(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size, ssm_cfg=None, **kwargs):
        super().__init__()
        self.forward_mamba = MixerModel(768, 24, 50277, ssm_cfg=ssm_cfg, **kwargs)
        self.backward_mamba = MixerModel(768, 24, 50277, ssm_cfg=ssm_cfg, **kwargs)

    def forward(self, input_ids):
        # 正向传播
        forward_hidden_states = []
        x = self.forward_mamba.embedding(input_ids)
        for layer in self.forward_mamba.layers:
            x, _ = layer(x)
            forward_hidden_states.append(x)

        # 反向传播
        reversed_input_ids = torch.flip(input_ids, dims=[1])
        backward_hidden_states = []
        x = self.backward_mamba.embedding(reversed_input_ids)
        for layer in self.backward_mamba.layers:
            x, _ = layer(x)
            backward_hidden_states.append(x)

        # 合并前向和后向的隐藏状态
        hidden_states = [torch.cat((fwd, bwd), dim=-1) for fwd, bwd in
                         zip(forward_hidden_states, backward_hidden_states)]

        return hidden_states
