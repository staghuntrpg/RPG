#!/bin/sh
env="StagHuntGW"
seed_max=5

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=5 python eval/eval.py --env_name ${env} --seed ${seed} --num_agents 2 --episode_length 50 --model_dir "/home/yuchao/project/" --eval_episodes 100
done
