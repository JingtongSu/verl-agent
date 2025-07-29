#!/bin/bash
#SBATCH --account=ai_society
#SBATCH --job-name=qc_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output='textfiles/output_normal_test.log'
#SBATCH --error='textfiles/output_normal_test.err'
#SBATCH --mem=0
#SBATCH --time=24:00:00
export ALFWORLD_DATA=/checkpoint/ai_society/jtsu/alfworld/data
# python train_critic.py --use_wandb --wandb_run_name reweighted --save_path /checkpoint/ai_society/jtsu/verl-agent/critic_reweighted/ --batch_size 32
# python train_critic.py --use_wandb --wandb_run_name normal --save_path /checkpoint/ai_society/jtsu/verl-agent/critic_normal/ --batch_size 8
# python train_critic.py --use_wandb --wandb_run_name reweighted --save_path /checkpoint/ai_society/jtsu/verl-agent/critic_reweighted/ --batch_size 32 --store_model_name qv_critic_reweighted
# python train_critic.py --use_wandb --wandb_run_name normal --save_path /checkpoint/ai_society/jtsu/verl-agent/test/ --batch_size 256 --store_model_name qv_critic_normal

python train_critic.py --use_wandb --wandb_run_name freeze_test --save_path /checkpoint/ai_society/jtsu/verl-agent/freeze_test_no_reweight/ --batch_size 256 --store_model_name qv_critic_freeze_noreweight --freeze True
