#!/bin/sh
env="StagHunt"
algo="paper-original"
seed_max=10

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 512 --num_mini_batch 1 --num_agents 2 --ppo_epoch 4 --episode_length 10 --history_length 1 --lr 1e-3 --value_loss_coef 1 --num_env_steps 4000000 --recurrent_policy --data_chunk_length 10
done