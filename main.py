import os
import json
import wandb
import argparse

from utils.logging import get_flag_dict, get_exp_name, setup_wandb

def main(args):

    exp_name = get_exp_name(args.env_name)
    setup_wandb(project='hrl-arenaX', group=args.run_group, name=exp_name)

    args.save_dir = os.path.join(args.save_dir, wandb.run.project, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    flag_dict = get_flag_dict()

    with open(os.path.join(args.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    # Environment
    parser.add_argument('--env_name', type=str, default='FrankaikGolfCourseEnv-v0', help='Environment (dataset) name.')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Dataset directory.')
    parser.add_argument('--dataset_replace_interval', type=int, default=1000, help='Dataset replace interval.')
    parser.add_argument('--num_datasets', type=int, default=None, help='Number of datasets to use.')

    # Save / restore
    parser.add_argument('--save_dir', type=str, default='exp/', help='Save directory.')
    parser.add_argument('--restore_path', type=str, default=None, help='Restore path.')
    parser.add_argument('--restore_epoch', type=int, default=None, help='Restore epoch.')

    # Training steps and logging
    parser.add_argument('--offline_steps', type=int, default=5000000, help='Number of offline steps.')
    parser.add_argument('--log_interval', type=int, default=10000, help='Logging interval.')
    parser.add_argument('--eval_interval', type=int, default=250000, help='Evaluation interval.')
    parser.add_argument('--save_interval', type=int, default=5000000, help='Saving interval.')

    # Evaluation
    parser.add_argument('--eval_episodes', type=int, default=15, help='Number of episodes for each task.')
    parser.add_argument('--eval_temperature', type=float, default=0, help='Actor temperature for evaluation.')
    parser.add_argument('--eval_gaussian', type=float, default=None, help='Action Gaussian noise for evaluation.')

    args = parser.parse_args()

    main(args)