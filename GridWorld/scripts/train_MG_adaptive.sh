#!/bin/sh
env="StagHunt"
algo="RR"
seed_max=8

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    let "seed_new=seed+2"
    echo "seed is ${seed_new}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --seed ${seed_new} --n_rollout_threads 512 --num_mini_batch 1 --num_agents 1 --ppo_epoch 4 --episode_length 10 --history_length 1 --lr 1e-3 --value_loss_coef 1 --num_env_steps 4000000 --recurrent_policy --data_chunk_length 10 --num_policy_candidates 4 --critic_full_obs
done