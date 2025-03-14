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

exp_dir="../outputs"

####'bedrooms'
config="../config/text/diffusion_bedrooms_instancond_lat32_v_bert.yaml"
exp_name="diffusion_bedrooms_instancond_lat32_v_bert"
model_name="model_6000"
weight_file=$exp_dir/$exp_name/$model_name.pt
threed_future='../dataset/3d_front_processed/threed_future_model_bedroom.pkl'

python  generate_diffusion.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
    --without_screen  --n_sequences 1000 --render_top2down --save_mesh --no_texture --without_floor  --clip_denoised --retrive_objfeats
