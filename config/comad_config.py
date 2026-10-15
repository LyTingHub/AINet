# encoding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp
import sys
import time
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.print_coe = False
C.save_pred_gt = False
C.save_mpjpe = False
C.seed = 888
C.dataset = 'comad'
C.model_variant = 'comad'
C.n_joint = 21
C.human_joint = 11
C.robot_joint = 10
C.print_loss = 400

C.abs_dir = osp.dirname(osp.realpath(__file__))
C.repo_name = 'AINet'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log2'))
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, 'snapshot' + exp_time))
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'lib'))

C.motion = edict()
C.motion.input_length = 60
C.motion.input_length_dct = 60
C.motion.target_length_train = 10
C.motion.target_length_eval = 30
C.motion.dim1 = 33
C.motion.dim2 = 30

# Backward-compatible aliases used by the original experiment snapshots.
C.motion.h36m_input_length = C.motion.input_length
C.motion.h36m_input_length_dct = C.motion.input_length_dct
C.motion.h36m_target_length_train = C.motion.target_length_train
C.motion.h36m_target_length_eval = C.motion.target_length_eval

C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

C.pre_dct = False
C.post_dct = False
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = C.motion.dim1
C.motion_mlp.seq_len = C.motion.input_length_dct
C.motion_mlp.num_layers = 24
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

C.motion_fc_in = edict()
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

C.motion_fc_out = edict()
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

C.batch_size = 256
C.num_workers = 2
C.cos_lr_max = 1e-5
C.cos_lr_min = 5e-8
C.cos_lr_total_iters = 40000
C.weight_decay = 1e-4
C.model_pth = None
C.shift_step = 1
C.print_every = 1000
C.save_every = 1000
