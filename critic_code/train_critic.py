from hydra import compose, initialize
from omegaconf import OmegaConf
import os
from datetime import timedelta
from data_utils import SupervisedDataset
from trainer import QTrainer
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
# from utils import colorful_print
import argparse
import copy
import torch
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence
import argparse
import copy
from accelerate import Accelerator
from critic import VLMDoubleCritic

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
import wandb
# IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from agent_system.environments import make_envs
import yaml
from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
from functools import partial
from agent_system.environments.env_manager import AlfWorldEnvironmentManager
import json

os.environ["ALFWORLD_DATA"] = '/checkpoint/ai_society/jtsu/alfworld/data'

parser = argparse.ArgumentParser(description="Train the critic model.")
parser.add_argument('--critic', type=str, default='VLMDoubleCritic', help='Type of critic model to use.')
parser.add_argument('--target_critic', type=str, default='VLMDoubleCritic', help='Type of target critic model to use.')
parser.add_argument('--critic_lm', type=str, default='/checkpoint/ai_society/jtsu/hf_models/Qwen2.5-1.5B-Instruct', help='Pretrained language model for the critic.')
parser.add_argument('--cache_dir', type=str, default='/checkpoint/ai_society/jtsu/hf_models/cache', help='Cache directory for the pretrained model.')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--tau', type=float, default=0.1, help='Soft update parameter for target critic.')
parser.add_argument('--max_grad_norm', type=float, default=0.01, help='Maximum gradient norm for clipping.')
# parser.add_argument('--num_action_resampling', type=int, default= 10, help='Number of action resampling steps.')
parser.add_argument('--save_path', type=str, default='/checkpoint/ai_society/jtsu/verl-agent/critic', help='Path to save the trained critic model.')
parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging.')
parser.add_argument('--wandb_project', type=str, default='qv-critic', help='Wandb project name.')
parser.add_argument('--wandb_run_name', type=str, default='attempt', help='Wandb run name.')
parser.add_argument('--store_model_name', type=str, default='qv_critic', help='Model name to store.')
parser.add_argument('--reweighting', type=float, default=None, help='Reweighting factor for the loss.')
parser.add_argument('--freeze', type=bool, default=None, help='Whether to freeze the model parameters.')

args = parser.parse_args()

if args.use_wandb:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = transformers.AutoModel.from_pretrained(args.critic_lm, cache_dir=args.cache_dir).to(device)
model = VLMDoubleCritic(
    device=device,
    critic_lm=args.critic_lm,
    cache_dir=args.cache_dir,
    in_dim=1536, # Example dimension, adjust if necessary
    out_dim=1    # Example dimension, adjust if necessary
).to(device)

# freeze the model parameters except for the critic layers if specified
if args.freeze:
    print("Freezing model parameters except for critic layers.")
    for name, param in model.named_parameters():
        if 'q_critic' in name or 'v_critic' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

tokenizer = transformers.AutoTokenizer.from_pretrained(args.critic_lm)
tokenizer.truncation_side = 'left'

q_offline_data = SupervisedDataset(
    tokenizer=tokenizer
)

print("Dataset Prepared.")

# only take a subset of the dataset for training
# q_offline_data.data = q_offline_data.data[:256]  # Adjust
# print(f"Using {len(q_offline_data.data)} samples for training.")

dataloader = torch.utils.data.DataLoader(
    q_offline_data,
    batch_size=args.batch_size,
    shuffle=True
)

print("Dataloader Ready.")

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], 
                            project_dir = args.save_path,
                            gradient_accumulation_steps=args.grad_accum_steps)

trainer = QTrainer(
    critic=model,
    # make a copy of the critic model weights for target critic
    target_critic = copy.deepcopy(model).to(device),
    accelerator=accelerator,
    tokenizer=tokenizer,
    detach_model=True,
    reweighting=args.reweighting
)

print("Trainer Initialized.")

os.makedirs(args.save_path, exist_ok=True)

if os.path.exists(os.path.join(args.save_path, f'{args.store_model_name}.pt')):
    print("Loading from offline critic.")
    trainer.load(os.path.join(args.save_path, f'{args.store_model_name}.pt'))

print(">>>Training critic")
for i in range(args.epochs):
    info = trainer.update_critic(dataloader)
    if args.use_wandb and accelerator.is_main_process:
        # Add a prefix to distinguish epoch-level average stats
        epoch_info = {f"epoch_{k}": v for k, v in info.items()}
        wandb.log(epoch_info)
    trainer.save(os.path.join(args.save_path, f'{args.store_model_name}_epoch_{i+1}.pt'))
accelerator.wait_for_everyone()
trainer.save(os.path.join(args.save_path, f'{args.store_model_name}.pt'))

if args.use_wandb:
    wandb.finish()