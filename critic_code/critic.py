import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import ast
import re

class VLMDoubleCritic(torch.nn.Module):
    def __init__(self, device, critic_lm, cache_dir, in_dim, out_dim):
        """
        VLM critic using image features
        """
        super(VLMDoubleCritic, self).__init__()
        self.device = device
        
        self.lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'

        # for q
        self.q_critic1 = nn.Sequential(
                                    nn.Linear(1536*2, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        self.q_critic2 = nn.Sequential(
                                    nn.Linear(1536*2, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        
        # for v
        self.v_critic1 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        self.v_critic2 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        

    def forward(self, batch, detach_model=False, next_state=False):
        # state_inputs = {
        #     'input_ids': batch['state_input_ids'].to(self.device),
        #     'attention_mask': batch['state_attention_mask'].to(self.device)
        # }
        # action_inputs = {
        #     'input_ids': batch['action_input_ids'].to(self.device),
        #     'attention_mask': batch['action_attention_mask'].to(self.device)
        # }
        # next_state_inputs = {
        #     'input_ids': batch['next_state_input_ids'].to(self.device),
        #     'attention_mask': batch['next_state_attention_mask'].to(self.device)
        # }

        # if detach_model:
        #     with torch.no_grad():
        #         state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        #         action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        # else:
        #     state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        #     action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

        # q_states = torch.cat([state, action], dim = 1)
        # v_states = state if not next_state else self.lm(**next_state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

        # Tokenize states, actions, and next_states on the fly
        state_inputs = self.tokenizer(
            batch['s'], return_tensors="pt", padding=True, truncation=True, max_length=1280
        ).to(self.device)
        
        action_inputs = self.tokenizer(
            batch['a'], return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)

        if detach_model:
            with torch.no_grad():
                state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
                action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        else:
            state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

        q_states = torch.cat([state, action], dim = 1)

        # print("q_states shape:", q_states.shape)
        
        if next_state:
            next_state_inputs = self.tokenizer(
                batch['s_prime'], return_tensors="pt", padding=True, truncation=True, max_length=1280
            ).to(self.device)
            v_states = self.lm(**next_state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        else:
            v_states = state

        # print("v_states shape:", v_states.shape)

        return self.q_critic1(q_states), self.q_critic2(q_states), self.v_critic1(v_states), self.v_critic2(v_states)
    

class VLMDoubleCriticActionMerged(torch.nn.Module):
    def __init__(self, device, critic_lm, cache_dir, in_dim, out_dim):
        """
        VLM critic using image features
        """
        super(VLMDoubleCriticActionMerged, self).__init__()
        self.device = device
        
        self.lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'

        # for q
        self.q_critic1 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        self.q_critic2 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        
        # for v
        self.v_critic1 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        self.v_critic2 = nn.Sequential(
                                    nn.Linear(1536, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1024),\
                                    nn.ReLU(),\
                                    nn.Linear(1024, 1)).to(device)
        

    def forward(self, batch, detach_model=False, next_state=False):
        state_inputs = {
            'input_ids': batch['state_input_ids'].to(self.device),
            'attention_mask': batch['state_attention_mask'].to(self.device)
        }
        action_inputs = {
            'input_ids': batch['action_input_ids'].to(self.device),
            'attention_mask': batch['action_attention_mask'].to(self.device)
        }
        next_state_inputs = {
            'input_ids': batch['next_state_input_ids'].to(self.device),
            'attention_mask': batch['next_state_attention_mask'].to(self.device)
        }

        if detach_model:
            with torch.no_grad():
                state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
                action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        else:
            state = self.lm(**state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            action = self.lm(**action_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

        q_states = torch.cat([state, action], dim = 1)
        v_states = state if not next_state else self.lm(**next_state_inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]

        return self.q_critic1(q_states), self.q_critic2(q_states), self.v_critic1(v_states), self.v_critic2(v_states)
    