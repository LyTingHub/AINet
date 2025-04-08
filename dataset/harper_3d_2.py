import os
from glob import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
import numpy as np


def load_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


class Harper3D(Dataset):
    """
    Data loader for the Harper (3D) dataset.
    This data loader is designed to provide data for forecasting, but can easily adapted as per your needs.
    """

    def __init__(self, data_path: str, split: str, n_input: int, n_output: int, sample: int, action='all', subject='all') -> None:
        '''

        :param data_path:
        :param split:
        :param n_input:
        :param n_output:
        :param sample:
        :param action:   'all' or List
        :param subject:
        '''
        # Sanity checks
        assert os.path.exists(data_path), f"Path {data_path} does not exist. Please download the dataset first"
        assert split in ["train", "test"], f"Split {split} not recognized. Use either 'train' or 'test'"
        data_folder = os.path.join(data_path, split)
        assert os.path.exists(
            data_folder), f"Path {data_folder} does not exist. It is in the correct format? Refer to the README"
        assert n_input > 0 and isinstance(n_input, int), f"n_input must be an integer greater than 0"
        assert n_output > 0 and isinstance(n_output, int), f"n_output must be an integer greater than 0"

        # Load data
        self.n_input = n_input
        self.n_output = n_output
        self.sample_rate = sample
        self.pkls_files: list[str] = glob(os.path.join(data_folder, "*.pkl"))  #
        self.all_sequences: list[dict[int, dict]] = [load_pkl(f) for f in self.pkls_files]  #

        # print(self.pkls_files)
        # self._get_subject_action()
        self.action = ['act1_0', 'act1_45', 'act1_90', 'act1_180', 'act2', 'act3', 'act4', 'act5', 'act6', 'act7', 'act8', 'act9', 'act10', 'act11', 'act12']
        self.actname2int = dict(zip(self.action, range(15)))
        self.actOneHot = F.one_hot(torch.arange(15), num_classes=15)
        self.test_subject = ["t", "toa", "xu", "xy", "yf"]  # 5
        self.train_subject = ['avo', 'bn', 'cun', 'el', 'h', 'jk', 'j', 'mt', 'ric', 'ry', 'sh', 'son']
        self.dimension_use = np.arange((21 + 23) * 3)
        self.in_features = len(self.dimension_use)
        self.shift_step = 1
        self._get_sequences(subjects=subject, actions=action)
        assert len(self.all_sequences_windows) > 0, f"ERROR! len(dataset)==0"


    def __len__(self):
        return len(self.all_sequences_windows)

    def _get_sequences(self, actions='all', subjects='all'):
        # list of sliding windows on sequences
        # NOTE: if you need to select a specific subject and action, you can find it in every dictionary under "subject" and "action" keys
        self.all_sequences_windows = []
        if actions == 'all' and subjects == 'all':
            for sequence in self.all_sequences:
                sequence_list = [v for k, v in sequence.items()]  # list[dict]
                for i in range(0, len(sequence_list) - self.shift_step * (self.n_input + self.n_output) + 1, self.sample_rate):
                    self.all_sequences_windows.append(
                        sequence_list[i: i + (self.n_input + self.n_output) * self.shift_step:self.shift_step])
        else:
            for sequence in self.all_sequences:
                subject, action = sequence[0]['subject'], sequence[0]['action']
                if (subjects == 'all' or subject in subjects) and (actions == 'all' or action in actions):
                    sequence_list = [v for k, v in sequence.items()]   # list[dict]
                    for i in range(0, len(sequence_list) - self.shift_step * (self.n_input + self.n_output) + 1, self.sample_rate):
                        self.all_sequences_windows.append(
                            sequence_list[i: i + (self.n_input + self.n_output) * self.shift_step:self.shift_step])

    def _get_subject_action(self):
        self.subject = []
        self.action = []
        for sequence in self.all_sequences:
            if sequence[0]['subject'] not in self.subject:
                self.subject.append(sequence[0]['subject'])
            if sequence[0]['action'] not in self.action:
                self.action.append(sequence[0]['action'])
        print(self.subject)
        print(self.action)

    def __getitem__(self, idx):  # -> "dict[str, torch.Tensor|str]":
        curr_data = self.all_sequences_windows[idx]  # list
        human = torch.tensor([obs["human_joints_3d"] for obs in curr_data], dtype=torch.float32)
        spot = torch.tensor([obs["spot_joints_3d"] for obs in curr_data], dtype=torch.float32)
        seq_len, _, _ = human.shape  # seq_len, 21, 3
        human = human.reshape(seq_len, -1)
        spot = spot.reshape(seq_len, -1)
        data = torch.cat([human, spot], dim=-1)
        motion_input, motion_target = data[:self.n_input], data[self.n_input:]
        return motion_input, motion_target

def gen_velocity(m):
    dm = np.zeros_like(m)
    dm[:, 1:] = m[:, 1:] - m[:, :-1]
    dm[:, 0] = dm[:, 1]
    return dm

