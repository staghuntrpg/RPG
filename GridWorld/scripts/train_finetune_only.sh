#!/bin/sh
env="StagHuntGW"
algo="paper-502"
seed_max=3

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_finetune_only.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 256 --num_mini_batch 1 --num_agents 2 --ppo_epoch 4 --episode_length 50 --lr 1e-3 --value_loss_coef 1 --num_env_steps 14000000
done
