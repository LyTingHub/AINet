import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

COMAD_SUBTASKS = ['cabinet', 'cart', 'handover', 'react', 'tabletop']


def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def missing_data(tensor):
    return torch.any(torch.all(tensor == 0, dim=-1))


def fix_orientation(tensor):
    tensor = tensor.clone()
    tensor[:, :, [0, 1, 2]] = tensor[:, :, [0, 2, 1]]
    tensor[:, :, 0] *= -1
    return tensor


def gen_velocity(motion):
    if torch.is_tensor(motion):
        dm = torch.zeros_like(motion)
    else:
        dm = np.zeros_like(motion)
    dm[:, 1:] = motion[:, 1:] - motion[:, :-1]
    dm[:, 0] = dm[:, 1]
    return dm


class CoMaD_HR(Dataset):
    """CoMaD human-robot 3D skeleton loader used in the AINet paper."""

    def __init__(
        self,
        data_dir,
        input_n=60,
        output_n=30,
        sample_rate=120,
        output_rate=30,
        split='train',
        subtask=None,
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        assert os.path.isdir(data_dir), f"Path {data_dir} does not exist."
        assert sample_rate % output_rate == 0, "sample_rate must be divisible by output_rate."

        self.data_dir = data_dir
        self.input_n = input_n
        self.output_n = output_n
        self.sequence_len = input_n + output_n
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.subtask = COMAD_SUBTASKS if subtask is None else subtask

        self.human_input, self.human_output = [], []
        self.robot_input, self.robot_output = [], []
        self.add_comad_dataset()
        assert len(self.robot_input) > 0, "No CoMaD samples were found. Check data_root and subtask names."

    def add_comad_dataset(self):
        for subtask in self.subtask:
            self.add_comad_dataset_subtask(subtask)

    def add_comad_dataset_subtask(self, subtask):
        subtask_dir = os.path.join(self.data_dir, self.split, subtask, 'HR')
        assert os.path.isdir(subtask_dir), f"Missing CoMaD subtask directory: {subtask_dir}"

        downsample_rate = self.sample_rate // self.output_rate
        for episode in sorted(os.listdir(subtask_dir)):
            episode_dir = os.path.join(subtask_dir, episode)
            json_path = os.path.join(episode_dir, 'data.json')
            if not os.path.isfile(json_path):
                continue

            json_data = read_json(json_path)
            human_tensor = fix_orientation(torch.tensor(json_data['Person_1']))[::downsample_rate]
            robot_tensor = fix_orientation(torch.tensor(json_data['Robot']))[::downsample_rate]

            for start_frame in range(human_tensor.shape[0] - self.sequence_len + 1):
                end_frame = start_frame + self.sequence_len
                if missing_data(human_tensor[start_frame:end_frame]) or missing_data(robot_tensor[start_frame:end_frame]):
                    continue
                self.human_input.append(human_tensor[start_frame:start_frame + self.input_n])
                self.human_output.append(human_tensor[start_frame + self.input_n:end_frame])
                self.robot_input.append(robot_tensor[start_frame:start_frame + self.input_n])
                self.robot_output.append(robot_tensor[start_frame + self.input_n:end_frame])

    def __len__(self):
        return len(self.robot_input)

    def __getitem__(self, idx):
        motion_input = torch.cat(
            [self.human_input[idx].reshape(self.input_n, -1), self.robot_input[idx].reshape(self.input_n, -1)],
            dim=-1,
        ).float()
        motion_output = torch.cat(
            [self.human_output[idx].reshape(self.output_n, -1), self.robot_output[idx].reshape(self.output_n, -1)],
            dim=-1,
        ).float()
        vel_input = gen_velocity(motion_input.reshape([self.input_n, 21, 3]))
        vel_output = gen_velocity(motion_output.reshape([self.output_n, 21, 3]))
        return motion_input, motion_output, vel_input, vel_output
