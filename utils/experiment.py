import importlib
import os
import random

import numpy as np
import torch

from dataset.chico import NORMAL_ACTIONS, PoseDataset
from dataset.comad_hr import COMAD_SUBTASKS, CoMaD_HR
from dataset.harper_3d_2 import Harper3D

DATASET_CONFIGS = {
    'harper': 'config.harper_config',
    'comad': 'config.comad_config',
    'chico': 'config.chico_config',
}

HARPER_ACTIONS = [
    'act1_0', 'act1_45', 'act1_90', 'act1_180', 'act2',
    'act3', 'act4', 'act5', 'act6', 'act7',
    'act8', 'act9', 'act10', 'act11', 'act12',
]


def load_config(dataset):
    module = importlib.import_module(DATASET_CONFIGS[dataset])
    return module.config


def apply_runtime_options(config, args):
    config.motion_fc_in.temporal_fc = args.temporal_only
    config.motion_fc_out.temporal_fc = args.temporal_only
    config.motion_mlp.norm_axis = args.layer_norm_axis
    config.motion_mlp.spatial_fc_only = args.spatial_fc
    if args.with_normalization is not None:
        config.motion_mlp.with_normalization = args.with_normalization
    config.motion_mlp.num_layers = args.num
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    if args.total_iters is not None:
        config.cos_lr_total_iters = args.total_iters
    if args.save_every is not None:
        config.save_every = args.save_every


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)


def set_motion_target_length(config, length):
    config.motion.target_length = length
    config.motion.harper_target_length = length
    config.motion.h36m_target_length = length


def action_list(dataset):
    if dataset == 'harper':
        return HARPER_ACTIONS
    if dataset == 'comad':
        return COMAD_SUBTASKS
    if dataset == 'chico':
        return NORMAL_ACTIONS
    raise ValueError(f'Unknown dataset: {dataset}')


def normalize_action_arg(dataset, action):
    if action in (None, 'all'):
        return 'all'
    if dataset == 'chico' and not action.endswith('.pkl'):
        return [action]
    return [action]


def build_dataset(dataset, data_root, config, split, target_length, action='all'):
    if dataset == 'harper':
        return Harper3D(
            data_path=data_root,
            split=split,
            n_input=config.motion.input_length,
            n_output=target_length,
            sample=1,
            action=normalize_action_arg(dataset, action),
        )

    if dataset == 'comad':
        subtasks = COMAD_SUBTASKS if action in (None, 'all') else [action]
        return CoMaD_HR(
            data_dir=data_root,
            input_n=config.motion.input_length,
            output_n=target_length,
            split=split,
            subtask=subtasks,
            sample_rate=120,
            output_rate=30,
        )

    if dataset == 'chico':
        actions = NORMAL_ACTIONS if action in (None, 'all') else normalize_action_arg(dataset, action)
        return PoseDataset(
            data_path=data_root,
            split=split,
            input_time_frames=config.motion.input_length,
            output_time_frames=target_length,
            actions=actions,
            win_stride=1,
        )

    raise ValueError(f'Unknown dataset: {dataset}')


def get_dct_matrix(num_frames):
    dct_m = np.eye(num_frames)
    for k in np.arange(num_frames):
        for i in np.arange(num_frames):
            w = np.sqrt(2 / num_frames)
            if k == 0:
                w = np.sqrt(1 / num_frames)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / num_frames)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def get_out_print_frame(out_n):
    if out_n == 30:
        return [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    if out_n == 15:
        return [2, 5, 8, 11, 14]
    if out_n == 10:
        return list(range(10))
    if out_n == 25:
        return [1, 3, 7, 9, 13, 17, 21, 24]
    return [out_n - 1]
