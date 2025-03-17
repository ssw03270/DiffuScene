#!/usr/bin/bash
#SBATCH -J diffusion_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

# 현재 작업 디렉토리 확인
pwd
which python
hostname

# scripts 폴더로 이동
cd ./scripts

# 결과 디렉토리 지정
exp_dir="outputs"

#### bedrooms
config="../config/uncond/diffusion_bedrooms_instancond_lat32_v.yaml"
exp_name="diffusion_bedrooms_instancond_lat32_v"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name --with_wandb_logger

#### diningrooms
config="../config/uncond/diffusion_diningrooms_instancond_lat32_v.yaml"
exp_name="diffusion_diningrooms_instancond_lat32_v"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name --with_wandb_logger

#### livingrooms
config="../config/uncond/diffusion_livingrooms_instancond_lat32_v.yaml"
exp_name="diffusion_livingrooms_instancond_lat32_v"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name --with_wandb_logger

exit 0