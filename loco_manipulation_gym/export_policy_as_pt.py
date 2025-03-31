from __init__ import LOCO_MANI_GYM_ROOT_DIR
import os
import subprocess

import isaacgym
from envs import *
from utils import get_args, task_registry, get_load_path, class_to_dict
from modules import ActorCritic, ActorCriticRecurrent

import numpy as np
import torch
import copy
from modules import ActorCriticRMA,ActorCriticBarlowTwins

from envs.tita_noarm.tita_noarm_config import TitaNoArmRoughCfg, TitaNoArmRoughCfgPPO
from envs.tita_noarm.tita_noarm_robot import TitaNoArm
from termcolor import cprint

def export_policy_as_pt(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    log_root = os.path.join(LOCO_MANI_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=train_cfg.runner.checkpoint)
    cprint(f'Loading Model: {resume_path}', 'green')
    
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    
    policy = actor_critic_class(
        env_cfg.env.num_propriceptive_obs,
        env_cfg.env.num_privileged_obs,
        env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy)
    ).to('cpu')
    
    loaded_dict = torch.load(resume_path)
    policy.load_state_dict(loaded_dict['model_state_dict'])
    policy.eval()
    
    export_path = os.path.join(os.path.dirname(resume_path), 'exported', 'policy_1.pt')
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    torch.save(policy, export_path)
    cprint(f'Successfully exported deployable model to: {export_path}', 'green')
    
    test_input = torch.randn(1, env_cfg.env.num_propriceptive_obs)
    output = policy.actor(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape} (should match action dim: {env_cfg.env.num_actions})")


if __name__ == '__main__':
    task_registry.register("tita_noarm", TitaNoArm, TitaNoArmRoughCfg(), TitaNoArmRoughCfgPPO())
    args = get_args()
    export_policy_as_pt(args)
