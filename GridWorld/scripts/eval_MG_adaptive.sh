#!/bin/sh
env="StagHuntAdaptive"
seed_max=6

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=5 python eval/eval_MG_adaptive.py --env_name ${env} --seed ${seed} --num_agents 1 --episode_length 10 --history_length 1 --model_dir "/home/yuchao/project/mappo-ssd/results/StagHunt/RR/" --num_policy_candidates 4 --recurrent_policy --data_chunk_length 10 --critic_full_obs
done