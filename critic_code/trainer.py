import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
# from digiq.data import ReplayBufferDataset
import random
from utils import colorful_print
import time
import wandb

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class QTrainer():
    def __init__(self,\
        critic,\
        target_critic,\
        accelerator,\
        tokenizer,\
        critic_lr: float = 1e-3,\
        lm_lr: float = 1e-5,\
        grad_accum_steps: int = 8,\
        gamma: float = 0.9,\
        tau: float = 0.1,\
        epochs: int = 3,\
        max_grad_norm: float=0.01,
        actor_epochs: int = 3,
        advantage_estimation: str = "bellman",
        num_action_resampling: int = 4,
        task_set = "",
        actor_always_include_original_action = True,
        actor_loss_type = "best-of-n",
        pg_multiplier = 10.0,
        awr_beta = 0.05,
        detach_model = False,
        reweighting: float = None,
    ):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.critic = critic
        self.target_critic = target_critic
        self.tokenizer = tokenizer
        # self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()), lr = critic_lr)
        self.task_set = task_set
        self.advantage_estimation = advantage_estimation
        self.detach_model = detach_model
        
        # self.actor_loss_type = actor_loss_type
        self.pg_multiplier = pg_multiplier
        self.awr_beta = awr_beta
        self.criterion = torch.nn.MSELoss(reduction='none')
        
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.softmax = torch.nn.Softmax(dim = -1)
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.num_action_resampling = num_action_resampling
        self.device = self.accelerator.unwrap_model(self.critic).device
        self.reweighting = reweighting

    def prepare(self, dataloader):
        # self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
        # self.critic_optimizer = self.accelerator.prepare(self.critic_optimizer)
        self.critic, self.target_critic, self.critic_optimizer, self.dataloader = self.accelerator.prepare(
            self.critic, self.target_critic, self.critic_optimizer, dataloader
        )

    def critic_loss(self, batch, validation=False, **kwargs):
        reward = torch.Tensor(batch['r']).to(self.device)
        done = (reward != 0).float()

        # both mc and bellman should obtain Q and V from the dataset (not from the model)
        q1, q2, v1, v2 = self.critic(batch, detach_model=self.detach_model)
        with torch.no_grad():
            q1_target, q2_target, _, _ = self.target_critic(batch)
            # action is dummy in the line below
            _, _, v1_target, v2_target = self.target_critic(batch, next_state=True)
        q1, q2, v1, v2, q1_target, q2_target, v1_target, v2_target = q1.flatten(), q2.flatten(), v1.flatten(), v2.flatten(), q1_target.flatten(), q2_target.flatten(), v1_target.flatten(), v2_target.flatten()
        v1_target = reward + (1 - done)*v1_target*self.gamma
        v2_target = reward + (1 - done)*v2_target*self.gamma

        # q1_loss = self.criterion(q1, v1_target)
        # q2_loss = self.criterion(q2, v2_target)
        # v1_loss = self.criterion(v1, q1_target)
        # v2_loss = self.criterion(v2, q2_target)

        if self.reweighting is not None:
            weights = done * self.reweighting + 1 # default self.reweighting = 99, implies 100 for done, 1 for not done

            q1_loss = (self.criterion(q1, v1_target) * weights).mean()
            q2_loss = (self.criterion(q2, v2_target) * weights).mean()
            v1_loss = (self.criterion(v1, q1_target) * weights).mean()
            v2_loss = (self.criterion(v2, q2_target) * weights).mean()
        else:
            q1_loss = self.criterion(q1, v1_target).mean()
            q2_loss = self.criterion(q2, v2_target).mean()
            v1_loss = self.criterion(v1, q1_target).mean()
            v2_loss = self.criterion(v2, q2_target).mean()

        if not validation:
            self.accelerator.backward(v1_loss+v2_loss+q1_loss+q2_loss)
        q1_loss, q2_loss = q1_loss.detach().cpu().item(), q2_loss.detach().cpu().item()
        v1_loss, v2_loss = v1_loss.detach().cpu().item(), v2_loss.detach().cpu().item()
        q1, q2, v1, v2 = q1.detach().cpu(), q2.detach().cpu(), v1.detach().cpu(), v2.detach().cpu()
        # v_max, q_max = v_max.detach().cpu(), q_max.detach().cpu()

        # calculate the probability for logging purpose
        info = {
                "q1.loss": q1_loss,\
                "q2.loss": q2_loss,\
                "q1.mean": torch.mean(q1).item(),\
                "q1.min": torch.min(q1).item(),\
                "q1.max": torch.max(q1).item(),\
                "q1.std": torch.std(q1).item(),\
                "q2.mean": torch.mean(q2).item(),\
                "q2.min": torch.min(q2).item(),\
                "q2.max": torch.max(q2).item(),\
                "q2.std": torch.std(q2).item(),\
                "qmin.std": torch.std(torch.minimum(q1, q2)).item(),\
                "v1.loss": v1_loss,\
                "v2.loss": v2_loss,\
                "v1.mean": torch.mean(v1).item(),\
                "v1.min": torch.min(v1).item(),\
                "v1.max": torch.max(v1).item(),\
                "v1.std": torch.std(v1).item(),
                "v2.mean": torch.mean(v2).item(),
                "v2.max": torch.max(v2).item(),
                "v2.min": torch.min(v2).item(),
                "v2.std": torch.std(v2).item(),
                "vmin.std": torch.std(torch.minimum(v1, v2)).item(),\
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    # def actor_loss(self, observation, action_list, image_features, mc_return, reward, q_rep_out, 
    #                q_rep_out_list, validation=False, **kwargs):
    #     # print(observation[0])
        
    #     num_action_resampling = self.num_action_resampling

    #     mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
    #     reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()

    #     with torch.no_grad():
    #         advantage_action_pairs = []
    #         action_list = [action.split("<split>") for action in action_list]

    #         action_id_list = random.sample(range(q_rep_out_list.shape[1]), num_action_resampling)
    #         if self.actor_always_include_original_action:
    #             action_id_list[0] = 0
    #         for action_id in action_id_list:
    #             # action_list: [4, 64]
    #             # q_rep_out_list: [4, 64, 4096]
    #             pi_action = [action_list_for_this_batch[action_id] for action_list_for_this_batch in action_list]
    #             q_rep_out = q_rep_out_list[:, action_id, :]
    #             q1, q2, v1, v2 = self.critic(observation, image_features, pi_action, q_rep_out, detach_model=False)
    #             if self.learn_metric == "classification":
    #                 # classification uses CrossEntropyLoss, so we need to apply softmax for aggregation
    #                 q1 = self.softmax(q1)[:, 1]
    #                 q2 = self.softmax(q2)[:, 1]
    #                 v1 = self.softmax(v1)[:, 1]
    #                 v2 = self.softmax(v2)[:, 1]

    #             q = torch.maximum(q1, q2).flatten()
    #             v = torch.maximum(v1, v2).flatten()
    #             advantage = q - v
                
    #             for batch_position in range(len(pi_action)):
    #                 if not self.agent.is_action_valid(pi_action[batch_position]):
    #                     colorful_print(f"Invalid action {pi_action[batch_position]} detected, setting advantage to zero", fg='red')
    #                     advantage[batch_position] = 0

    #             advantage_action_pairs.append((advantage, pi_action))

    #         batch_size = len(q)
    #         max_index = torch.stack([adv[0] for adv in advantage_action_pairs], dim=1).argmax(dim=1)

    #         advantage = torch.zeros(batch_size)
    #         pi_action = [""] * batch_size

    #         for i in range(batch_size):
    #             advantage[i] = advantage_action_pairs[max_index[i]][0][i]
    #             pi_action[i] = advantage_action_pairs[max_index[i]][1][i]

    #     advantage = torch.clamp(advantage, 0, 1)
    #     if self.task_set == "general":
    #         threshold = 0.10
    #     elif self.task_set == "webshop":
    #         threshold = 0.05
    #     else:
    #         raise ValueError(f"Unknown task set {self.task_set}")
        
    #     if self.actor_loss_type == "best-of-n":
    #         advantage = (advantage > threshold).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
    #     elif self.actor_loss_type == "pg":
    #         advantage = (advantage*self.pg_multiplier).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
    #     elif self.actor_loss_type == "awr":
    #         advantage = torch.exp(advantage/self.awr_beta).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
    #     elif self.actor_loss_type == "sft":
    #         advantage = torch.ones_like(advantage).to(dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        
    #     learned_actions = []
    #     for i in range(len(advantage)):
    #         if advantage[i] == 1:
    #             learned_actions.append(pi_action[i])

    #     image_features = image_features.to(self.agent.device)
    #     log_prob = self.agent.get_pi_theta_log_prob(observation, image_features, pi_action).sum(dim = 1).flatten()
    #     advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
    #     advantages = advantage.flatten()
    #     pg_loss = -torch.mean(log_prob.flatten()*advantages)
    #     value_loss = torch.zeros_like(pg_loss)
    #     if not validation:
    #         self.accelerator.backward(pg_loss+value_loss)
    #     advantages = advantages.detach().cpu()
    #     info =  {"pg.loss": pg_loss.detach().cpu().item(),
    #             "advantages.mean": advantages.mean(),
    #             "advantages.max": torch.max(advantages),
    #             "advantages.min": torch.min(advantages),
    #             "advantages.std": torch.std(advantages),}
    #     if validation:
    #         validation_info = {}
    #         for k,v in info.items():
    #             validation_info["validation."+k] = v
    #         return validation_info
    #     return info

    def soft_update_target_critic(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_critic(self, dataloader):
        self.step += 1
        info = {}
        
        # for epoch in tqdm(range(self.epochs), disable= not self.accelerator.is_main_process):
        info_list = []
        for batch_idx, batch in enumerate(self.dataloader):
            print(f"Batch {batch_idx+1}/{len(self.dataloader)}")
            with self.accelerator.accumulate(self.critic):
                self.critic_optimizer.zero_grad()
                batch_info = self.critic_loss(batch)
                
                # Log per-batch info to wandb if it's the main process
                if self.accelerator.is_main_process:
                    wandb.log(batch_info)
                info_list.append(batch_info)
                self.accelerator.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        info.update(dict_mean(info_list))
        
        # update target network each epoch
        if self.advantage_estimation == "bellman":
            self.soft_update_target_critic(tau=self.tau)
    
        # if validation_buffer is not None:
        #     info_list = []
        #     val_dataset = ReplayBufferDataset(validation_buffer)
        #     val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=self.grad_accum_steps * validation_buffer.batch_size)
        #     val_dataloader = DataLoader(val_dataset, batch_size=validation_buffer.batch_size, sampler=val_sampler)
        #     val_dataloader = self.accelerator.prepare(val_dataloader)
        #     with torch.no_grad():
        #         for batch in tqdm(val_dataloader, disable=True):
        #             info_list.append(self.critic_loss(validation=True, **batch))
        #     info.update(dict_mean(info_list))
        return info
        
    # def update_policy(self, replay_buffer, validation_buffer = None, no_update_actor=False):
    #     self.step += 1
    #     info = {}
        
    #     # Create the dataset and DataLoader once per update.
    #     dataset = ReplayBufferDataset(replay_buffer)
    #     sampler = RandomSampler(dataset, replacement=True, num_samples=self.grad_accum_steps * replay_buffer.batch_size)
    #     dataloader = DataLoader(dataset, batch_size=replay_buffer.batch_size, sampler=sampler)
    #     dataloader = self.accelerator.prepare(dataloader)
        
    #     if not no_update_actor:
    #         print(">>>Training phase of actor")
            
    #         info_list = []
    #         for epoch in tqdm(range(self.actor_epochs), disable= not self.accelerator.is_main_process):
    #             for batch in dataloader:
    #                 with self.accelerator.accumulate(self.agent.model):
    #                     info_list.append(self.actor_loss(**batch))
    #                     self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
    #                     self.lm_optimizer.step()
    #                     self.lm_optimizer.zero_grad()
    #         info.update(dict_mean(info_list))
            
    #     if validation_buffer is not None:
    #         print(">>>Validation phase of actor")
    #         info_list = []
    #         val_dataset = ReplayBufferDataset(validation_buffer)
    #         val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=self.grad_accum_steps * validation_buffer.batch_size)
    #         val_dataloader = DataLoader(val_dataset, batch_size=validation_buffer.batch_size, sampler=val_sampler)
    #         val_dataloader = self.accelerator.prepare(val_dataloader)
    #         info_list = []
    #         with torch.no_grad():
    #             for batch in tqdm(val_dataloader, disable=True):
    #                 info_list.append(self.actor_loss(validation=True, **batch))
    #         info.update(dict_mean(info_list))
            
    #     return info

    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)

    def load(self, path):
        self.accelerator.load_state(path)
        
        