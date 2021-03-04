#!/bin/sh
env="multi_StagHuntGW"
algo="population_based"
seed_max=15
echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train.py --env_name ${env} --algorithm_name ${algo} --seed ${seed} --n_rollout_threads 256 --num_mini_batch 1 --num_agents 3 --ppo_epoch 4 --episode_length 50 --lr 1e-3 --value_loss_coef 1 --num_env_steps 21000000
done
