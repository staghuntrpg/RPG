#!/bin/sh
env="StagHuntGWAdaptive"
seed_max=3

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=5 python eval/eval_GW_adaptive.py --env_name ${env} --seed ${seed} --num_agents 2 --episode_length 50 --model_dir "/home/yuchao/project/mappo-ssd/results/StagHuntGW/paper-extra/" --num_policy_candidates 6 --recurrent_policy --data_chunk_length 10 --critic_full_obs
done