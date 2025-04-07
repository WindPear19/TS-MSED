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


class MambaClassificationHead(nn.Module):
	def __init__(self, d_model, num_classes, **kwargs):
		super(MambaClassificationHead, self).__init__()

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

		self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=2)


		del self.lm_head

	def forward(self, input_ids, labels, attention_mask, token_type_ids):

		forward_hidden_states = self.backbone(input_ids)


		mean_hidden_states = forward_hidden_states.mean(dim=1)

		logits = self.classification_head(mean_hidden_states)

		ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(logits, labels)

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

