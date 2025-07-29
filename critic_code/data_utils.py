from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
# IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from agent_system.environments import make_envs
import yaml
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
from functools import partial
from agent_system.environments.env_manager import AlfWorldEnvironmentManager

ending = "\n\nNow it's your turn to take an action.\nYou should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. \nOnce you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\n"


class SupervisedDataset(Dataset):
    """Dataset for offline Q-learning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        results_dir = pathlib.Path('/checkpoint/ai_society/jtsu/verl-agent/results')
        all_dataset = []

        for result_file in sorted(results_dir.glob('dataset_*.json')):
            with open(result_file, 'r') as f:
                dataset = json.load(f)
                all_dataset.extend(dataset)
        # all_dataset: list, each element is a dict with keys 's', 'a', 'r', 's_prime',

        self.data = []
        # for item in all_dataset:
        #     # Tokenize state, action, and next_state
        #     tokenized_state = self.tokenizer(item['s'][:-len(ending)], return_tensors="pt", padding="max_length", truncation=True, max_length=1280)
        #     tokenized_action = self.tokenizer(item['a'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        #     tokenized_next_state = self.tokenizer(item['s_prime'][:-len(ending)], return_tensors="pt", padding="max_length", truncation=True, max_length=1280)

        #     self.data.append({
        #         'state_input_ids': tokenized_state.input_ids.squeeze(0),
        #         'state_attention_mask': tokenized_state.attention_mask.squeeze(0),
        #         'action_input_ids': tokenized_action.input_ids.squeeze(0),
        #         'action_attention_mask': tokenized_action.attention_mask.squeeze(0),
        #         'r': torch.tensor(item['r']),
        #         'next_state_input_ids': tokenized_next_state.input_ids.squeeze(0),
        #         'next_state_attention_mask': tokenized_next_state.attention_mask.squeeze(0),
        #     })
        for item in all_dataset:
            self.data.append({
                's': item['s'][:-len(ending)],
                'a': item['a'],
                'r': torch.tensor(item['r']),
                's_prime': item['s_prime'][:-len(ending)],
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i]


class SupervisedDatasetActionMerged(Dataset):
    """Dataset for offline Q-learning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDatasetActionMerged, self).__init__()

        results_dir = pathlib.Path('/checkpoint/ai_society/jtsu/verl-agent/results')
        all_dataset = []

        for result_file in sorted(results_dir.glob('dataset_*.json')):
            with open(result_file, 'r') as f:
                dataset = json.load(f)
                all_dataset.extend(dataset)
        # all_dataset: list, each element is a dict with keys 's', 'a', 'r', 's_prime',

        self.data = []
        for item in all_dataset:
            # Tokenize state, action, and next_state
            tokenized_state_and_action = self.tokenizer(item['s'][:-len(ending)] + "\nAction: " + item['a'], return_tensors="pt", padding=True, truncation=True)
            tokenized_next_state = self.tokenizer(item['s_prime'][:-len(ending)], return_tensors="pt", padding=True, truncation=True)

            self.data.append({
                'state_input_ids': tokenized_state_and_action.input_ids.squeeze(0),
                'state_attention_mask': tokenized_state_and_action.attention_mask.squeeze(0),
                'reward': torch.tensor(item['r']),
                'next_state_input_ids': tokenized_next_state.input_ids.squeeze(0),
                'next_state_attention_mask': tokenized_next_state.attention_mask.squeeze(0),
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i]