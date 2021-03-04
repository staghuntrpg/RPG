import argparse


def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='RPG.')

    # prepare
    parser.add_argument("--algorithm_name", type=str, default='check')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True)
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=1)
    parser.add_argument("--num_env_steps", type=int, default=10e9, help='number of environment steps to train (default: 10e6)')
    
    # adaptive
    parser.add_argument("--num_policy_candidates", type=int, default=10)
    parser.add_argument("--policy_finetune", action='store_true', default=False)
    parser.add_argument("--policy_adaptive", action='store_true', default=False) 
    
    # rnn
    parser.add_argument("--naive_recurrent_policy", action='store_true', default=False, help='use a naive recurrent policy')
    parser.add_argument("--recurrent_policy", action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument("--lstm", action='store_true', default=False, help='use a lstm policy')
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--critic_full_obs", action='store_true', default=False)
    
    # env
    parser.add_argument("--env_name", type=str, default='StagHunt')
    parser.add_argument("--num_agents", type=int, default=2)

    # matrix game
    parser.add_argument("--limit_game", action='store_true', default=False)
    parser.add_argument("--game_length", type=int, default=10)
    parser.add_argument("--history_length", type=int, default=1)
    parser.add_argument("--num_opponents", type=int, default=3)

    # reward
    parser.add_argument("--reward_randomization", action='store_true', default=False)
    parser.add_argument("--share_reward", action='store_true', default=False)
    parser.add_argument("--shape_reward", action='store_true', default=False)
    parser.add_argument("--shape_beta", type=float, default=0.8, help='use how much global reward')
    
    # network
    parser.add_argument("--share_policy", action='store_true', default=False, help='agent share the same policy')
    parser.add_argument("--hidden_size", type=int, default=64)
    
    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=10, help='number of ppo epochs (default: 4)')    
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=32, help='number of batches for ppo (default: 32)')   
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--lr", type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use-gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae-lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='compute returns taking into account time limits')
    
    # replay buffer
    parser.add_argument("--episode_length", type=int, default=10, help='number of forward steps in A2C (default: 5)')

    # run
    parser.add_argument("--use-linear-lr-decay", action='store_false', default=True, help='use a linear schedule on the learning rate')
    
    # save
    parser.add_argument("--save_interval", type=int, default=10)
    
    # log
    parser.add_argument("--log_interval", type=int, default=1)

    #eval
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--ifi", type=float, default=0.333333)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default='/home/yuchao/project/')
    
    
    args = parser.parse_args()

    return args
