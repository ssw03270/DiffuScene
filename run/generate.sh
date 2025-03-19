#!/usr/bin/bash
#SBATCH -J diffusion_train
#SBATCH --gres=gpu:4
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

####'bedrooms'
config="../config/uncond/diffusion_bedrooms_instancond_lat32_v.yaml"
exp_name="diffusion_bedrooms_instancond_lat32_v"
epoch="model_10000"
weight_file="$exp_dir/$exp_name/$epoch"
threed_future='../dataset/3d_front_processed/threed_future_model_bedroom.pkl'

python  generate_diffusion.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
    --without_screen  --n_sequences 1000 --render_top2down --save_mesh --no_texture --without_floor  --clip_denoised --retrive_objfeats

#
#####'diningrooms'
#config="../config/uncond/diffusion_diningrooms_instancond_lat32_v.yaml"
#exp_name="diningrooms_uncond"
#weight_file="$exp_dir/$exp_name/$exp_name.pt"
#threed_future='/cluster/balrog/jtang/3d_front_processed/diningrooms/threed_future_model_diningroom.pkl'
#
#python  generate_diffusion.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
#    --without_screen  --n_sequences 1000 --render_top2down --save_mesh --no_texture --without_floor  --clip_denoised --retrive_objfeats
#
#
#####'livingrooms'
#config="../config/uncond/diffusion_livingrooms_instancond_lat32_v.yaml"
#exp_name="livingrooms_uncond"
#weight_file="$exp_dir/$exp_name/$exp_name.pt"
#threed_future='/cluster/balrog/jtang/3d_front_processed/livingrooms/threed_future_model_livingroom.pkl'
#
#python  generate_diffusion.py $config  $exp_dir/$exp_name/gen_top2down_notexture_nofloor $threed_future  --weight_file $weight_file \
#    --without_screen  --n_sequences 1000 --render_top2down --save_mesh --no_texture --without_floor  --clip_denoised --retrive_objfeats