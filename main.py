import os
import json
import wandb
import time
import tqdm
import random
import argparse
import numpy as np
from collections import defaultdict

from agents import agents

import jax
import jax.numpy as jnp
from utils.buffers import GCDataset, Dataset
from utils.env_utils import make_env_and_datasets
from utils.logging import get_exp_name, setup_wandb, get_wandb_video
from utils.evaluation import *
from utils.flax_utils import save_agent

def sanitize_metrics(metrics):
    sanitized = {}
    for k, v in metrics.items():
        if isinstance(v, (jnp.ndarray, float, int)):
            sanitized[k] = float(v)
        else:
            sanitized[k] = v
    return sanitized

def main(args):

    exp_name = get_exp_name(args.env_name)
    setup_wandb(project='hrl-arenaX', group=args.run_group, name=exp_name)

    args.save_dir = os.path.join(args.save_dir, wandb.run.project, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    env, train_dataset, val_dataset = make_env_and_datasets(args.env_name)
    random.seed(args.seed)
    np.random.seed(args.seed)

    (agent_class, agent_config) = agents[args.agents]

    train_dataset = GCDataset(Dataset.create(**train_dataset),agent_config)
    val_dataset = GCDataset(Dataset.create(**val_dataset),agent_config)
    example_batch = train_dataset.sample(1)

    agent = agent_class.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        {},
    )

    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, args.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(agent_config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % args.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(agent_config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(sanitize_metrics(train_metrics), step=i)

        # Evaluate agent.
        if i == 1 or i % args.eval_interval == 0:
            if args.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=agent_config,
                    num_eval_episodes=args.eval_episodes,
                    num_video_episodes=args.video_episodes,
                    video_frame_skip=args.video_frame_skip,
                    eval_temperature=args.eval_temperature,
                    eval_gaussian=args.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if args.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(sanitize_metrics(eval_metrics), step=i)

        # Save agent.
        if i % args.save_interval == 0:
            save_agent(agent, args.save_dir, i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_group', type=str, default='Debug', help='Run group.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--agents', type=str, default="gcbc", help='Agent to load.')

    # Environment
    parser.add_argument('--env_name', type=str, default='humanoidmaze-medium-navigate-v0', help='Environment (dataset) name.')
    parser.add_argument('--dataset_dir', type=str, default="~/.ogbench/data", help='Dataset directory.')
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
    parser.add_argument('--eval_episodes', type=int, default=2, help='Number of episodes for each task.')
    parser.add_argument('--eval_temperature', type=float, default=0, help='Actor temperature for evaluation.')
    parser.add_argument('--eval_gaussian', type=float, default=None, help='Action Gaussian noise for evaluation.')
    parser.add_argument('--eval_tasks', type=float, default=None, help='Number of tasks to evaluate (None for all).')
    parser.add_argument('--eval_on_cpu', type=float, default=1.0, help='Whether to evaluate on CPU.')
    parser.add_argument('--video_episodes', type=int, default=1, help='Number of video episodes for each task.')
    parser.add_argument('--video_frame_skip', type=int, default=3, help='Frame skip for videos.')
    args = parser.parse_args()

    main(args)