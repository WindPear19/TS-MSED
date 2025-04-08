import torch
from torch import nn
from transformers import BertPreTrainedModel,BertModel
import numpy as np
import torch.nn.functional as F
from models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf
from collections import namedtuple
import torch
from configs.config import MambaConfig


from mamba_ssm import Mamba

# class SpatialAttentionModule(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv1 = nn.Conv2d(32, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


class MambaClassificationHead(nn.Module):
	def __init__(self, d_model, num_classes, **kwargs):
		super(MambaClassificationHead, self).__init__()

		# Use a linear layer to perform classification based on the input with size d_model and the number of classes to classify num_classes.
		self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

	def forward(self, hidden_states):
		return self.classification_head(hidden_states)


class MambaTextClassification(MambaLMHeadModel):
	def __init__(
			self,
			config: MambaConfig,
			initializer_cfg=None,
			device=None,
			dtype=None,
	) -> None:
		super().__init__(config, initializer_cfg, device, dtype)

		self.embedding = nn.Embedding(config.vocab_size, config.d_model)

		# self.spatial_attention = SpatialAttentionModule(kernel_size=7)

#		Create a classification head using MambaClassificationHead with input size of d_model and number of classes 2.
		self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=2)

		# self.attention_layer = MLAttention(hidden_size=768)
		del self.lm_head
		# self.classification_head = MambaClassificationHead(d_model=768, num_classes=2)

	# def forward(self, input_ids, ,token_type_ids,attention_mask=None, labels=None):
	def forward(self, input_ids, labels, attention_mask, token_type_ids):

		forward_hidden_states = self.backbone(input_ids)

		# x_flip = torch.flip(input_ids, dims=[1])
		# backward_hidden_states = self.backbone(x_flip)

		# forward_hidden_states = self.spatial_attention(forward_hidden_states)

		# hidden_states = forward_hidden_states + backward_hidden_states
		mean_hidden_states = forward_hidden_states.mean(dim=1)

		logits = self.classification_head(mean_hidden_states)

		ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(logits, labels)

		# # 计算损失LDMLOSS
		# cls_num_list = [2000, 2000]
		# ldam_loss_fct = LDAMLoss(cls_num_list=cls_num_list)
		# loss = ldam_loss_fct(logits, labels)

		return loss, logits

	def predict(self, text, tokenizer, id2label=None):
		input_ids = torch.tensor(tokenizer(text)['input_ids'], device="cuda")[None]
		with torch.no_grad():
			logits = self.forward(input_ids).logits[0]
			label = np.argmax(logits.cpu().numpy())

		if id2label is not None:
			return id2label[label]
		else:
			return label

	@classmethod
	def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
		# Load the configuration from the pre-trained model.
		config_data = load_config_hf(pretrained_model_name)
		config = MambaConfig()

		# Initialize the model from the configuration and move it to the desired device and data type.
		model = cls(config, device=device, dtype=dtype, **kwargs)

		# Load the state of the pre-trained model.
		model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
		model.load_state_dict(model_state_dict, strict=False)

		# Print the newly initialized embedding parameters.
		print(" Newly initialized embedding :",
			  set(model.state_dict().keys()) - set(model_state_dict.keys())
			  )

		return model.to(device)

