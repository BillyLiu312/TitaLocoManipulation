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

def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    print("train_cfg.runner.................:", train_cfg.runner)

    log_root = os.path.join(LOCO_MANI_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    cprint(f'Loading Model: {resume_path}', 'green')
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    # export policy as an onnx file
    path = os.path.join(LOCO_MANI_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    engine_path =  os.path.join(LOCO_MANI_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies/policy.engine')

    if os.path.exists(engine_path):
        print(f"Engine file already exists: {engine_path}")
    else:   
        print("Exported policy as onnx script to: ", engine_path)
        convert_onnx_to_engine(engine_path)

def convert_onnx_to_engine(engine_path):
    onnx_path = engine_path.replace(".engine", ".onnx")
    trtexec_path = "/usr/local/TensorRT-10.4.0.26/bin/trtexec"

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"  # Use FP16 precision if supported
    ]

    try:
        print("Converting ONNX to TensorRT engine...")
        subprocess.run(command, check=True)
        print("Converted TensorRT engine saved to:", engine_path)
    except subprocess.CalledProcessError as e:
        print("Error during ONNX to TensorRT conversion:", e)


if __name__ == '__main__':
    task_registry.register("tita_noarm", TitaNoArm, TitaNoArmRoughCfg(), TitaNoArmRoughCfgPPO())
    args = get_args()
    export_policy_as_onnx(args)
