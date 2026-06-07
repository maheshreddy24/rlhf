import torch.nn as nn
import torch
import numpy as np
from transformers import AutoModelForCausalLM

class RewardHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    # the init is taken from the N+ implementation
    def _post_init(self):
        nn.init.normal_(self.reward.weight, std=(1.0 / np.sqrt(self.hidden_size + 1)))
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        return self.reward(hidden_states)

class GPT2RewardHead(nn.Module):
    def __init__(self, sft_model_path):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(sft_model_path)
        self.reward_head = RewardHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        reward = self.reward_head(last_hidden_state).squeeze(-1) #2*bs, seq_len 
        # ic(reward.shape)
        return reward