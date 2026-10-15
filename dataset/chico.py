import os
import pickle

import numpy as np
from torch.utils.data import Dataset

NORMAL_ACTIONS = [
    'hammer.pkl',
    'lift.pkl',
    'place-hp.pkl',
    'place-lp.pkl',
    'polish.pkl',
    'span_heavy.pkl',
    'span_light.pkl',
]

ABNORMAL_ACTIONS = [
    'place-hp_CRASH.pkl',
    'place-lp_CRASH.pkl',
    'polish_CRASH.pkl',
    'span_heavy_CRASH.pkl',
    'span_light_CRASH.pkl',
]

NORMAL_ACTION_NAMES = [a[:-4] for a in NORMAL_ACTIONS]
ABNORMAL_ACTION_NAMES = [a[:-4] for a in ABNORMAL_ACTIONS]


def gen_velocity(motion):
    dm = np.zeros_like(motion)
    dm[:, 1:] = motion[:, 1:] - motion[:, :-1]
    dm[:, 0] = dm[:, 1]
    return dm


class PoseDataset(Dataset):
    """CHICO human-robot pose forecasting split used in the AINet paper."""

    def __init__(
        self,
        data_path,
        split='train',
        input_time_frames=50,
        output_time_frames=25,
        win_stride=1,
        collision=False,
        actions=None,
        scale=1000,
    ):
        super().__init__()
        assert os.path.isdir(data_path), f"Path {data_path} does not exist."

        self.data_path = data_path
        self.split = split
        self.in_tf = input_time_frames
        self.out_tf = output_time_frames
        self.win_size = input_time_frames + output_time_frames
        self.win_stride = self.win_size if win_stride == 0 else win_stride
        self.scale = scale

        self.subjects = self._subjects_for_split(split)
        allowed_actions = ABNORMAL_ACTIONS if collision else NORMAL_ACTIONS
        allowed_names = ABNORMAL_ACTION_NAMES if collision else NORMAL_ACTION_NAMES
        actions = allowed_actions if actions is None else actions
        self.actions = self._normalize_actions(actions, allowed_actions, allowed_names)
        assert self.actions, 'There should be at least one CHICO action.'

        self.human_windows, self.robot_windows = self.build_dataset()
        assert self.human_windows.shape[0] > 0, "No CHICO samples were found. Check data_path, split, and actions."

    @staticmethod
    def _subjects_for_split(split):
        if split == 'train':
            return ['S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S01', 'S16', 'S17']
        if split == 'validation':
            return ['S00', 'S04']
        if split == 'test':
            return ['S02', 'S03', 'S18', 'S19']
        raise ValueError("split must be 'train', 'validation', or 'test'")

    @staticmethod
    def _normalize_actions(actions, allowed_files, allowed_names):
        normalized = []
        for action in actions:
            filename = action if action.endswith('.pkl') else action + '.pkl'
            name = filename[:-4]
            if filename in allowed_files or name in allowed_names:
                normalized.append(filename)
        return normalized

    def __len__(self):
        return self.human_windows.shape[0]

    def __getitem__(self, index):
        motion_input = np.concatenate(
            (self.human_windows[index][:self.in_tf], self.robot_windows[index][:self.in_tf]),
            axis=-1,
        )
        motion_target = np.concatenate(
            (self.human_windows[index][self.in_tf:], self.robot_windows[index][self.in_tf:]),
            axis=-1,
        )
        vel_input = gen_velocity(motion_input.reshape([self.in_tf, 24, 3]))
        vel_output = gen_velocity(motion_target.reshape([self.out_tf, 24, 3]))
        return motion_input, motion_target, vel_input, vel_output

    def build_dataset(self):
        human_all_data = []
        robot_all_data = []

        for subject in self.subjects:
            sub_path = os.path.join(self.data_path, subject)
            assert os.path.isdir(sub_path), f"Missing CHICO subject directory: {sub_path}"
            sub_actions_paths = [
                os.path.join(sub_path, act)
                for act in sorted(os.listdir(sub_path))
                if act.endswith('.pkl') and act in self.actions
            ]
            for sub_actions_path in sub_actions_paths:
                hpose_np, rpose_np = self.retrieve_data(sub_actions_path)
                human_all_data.append((self.split_single_pose(hpose_np) / self.scale).astype(np.float32))
                robot_all_data.append((self.split_single_pose(rpose_np) / self.scale).astype(np.float32))

        return np.concatenate(human_all_data), np.concatenate(robot_all_data)

    @staticmethod
    def retrieve_data(action_path):
        with open(action_path, 'rb') as f:
            all_data = pickle.load(f)

        human_related_data = [x[0] for x in all_data]
        robot_related_data = [x[1] for x in all_data]
        return np.stack(human_related_data, axis=0), np.stack(robot_related_data, axis=0)

    def split_single_pose(self, single_pose_array):
        total_frames = single_pose_array.shape[0]
        iterations = np.ceil((total_frames - self.win_size) / self.win_stride).astype(int) + 1
        all_windows = []
        for segment in range(iterations):
            start_index = self.win_stride * segment
            end_index = start_index + self.win_size
            curr_win = single_pose_array[start_index:end_index]
            if curr_win.shape[0] == self.win_size:
                all_windows.append(curr_win.reshape(self.win_size, -1))

        return np.stack(all_windows, axis=0)
