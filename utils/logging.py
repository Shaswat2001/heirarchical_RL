import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb

def get_exp_name(env_name):

    base_name = env_name.split("-")[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{base_name}_{timestamp}"

    return exp_name

def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        project=project,
        entity=entity,  
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run
