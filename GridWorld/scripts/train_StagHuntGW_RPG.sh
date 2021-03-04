#!/bin/sh
env="StagHuntGW"
algo="paper-new"
seed_max=3

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_RPG.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 256 --num_mini_batch 1 --num_agents 2 --ppo_epoch 4 --episode_length 50 --lr 1e-3 --value_loss_coef 1 --num_env_steps 7000000 --reward_randomization --num_policy_candidates 1
done