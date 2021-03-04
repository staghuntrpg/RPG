#!/bin/sh
env="StagHuntSingle"
seed_max=10

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=5 python eval/eval_MG_single.py --env_name ${env} --seed ${seed} --num_agents 1 --episode_length 10 --history_length 1 --model_dir "/home/yuchao/project/mappo-ssd/results/StagHunt/paper-extra/" --num_opponents 4 --recurrent_policy --data_chunk_length 10 --critic_full_obs
done