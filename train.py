import argparse
import os
import json
import numpy as np
import copy

from config.harper_config import config as config
from network.AINet import AINet as Model

from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from dataset.harper_3d_2 import Harper3D
from test import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

n_joint = 21 + 23
human_joint = 21
robot_joint = 23
data_root = r'/data3/0lyt/data/30hz'

config.motion.harper_target_length = config.motion.harper_target_length_train
dataset = Harper3D(data_path=data_root, split="train", n_input=config.motion.harper_input_length,
                   n_output=config.motion.harper_target_length, sample=1)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.harper_target_length = eval_config.motion.harper_target_length_eval
eval_dataset = Harper3D(data_path=data_root, split="test", n_input=eval_config.motion.harper_input_length,
                        n_output=eval_config.motion.harper_target_length, sample=1)

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                             num_workers=1, drop_last=False,
                             sampler=sampler, shuffle=shuffle, pin_memory=True)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=24, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--work_dir', type=str, default=".", help='=work_dir')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
ensure_dir('./result')
acc_log = open("./result/" + args.exp_name, 'a')
torch.manual_seed(args.seed)
ensure_dir('./ckpt_LNv5_reproduce')
writer = SummaryWriter('./ckpt_LNv5_reproduce')

config.motion.harper_target_length = config.motion.harper_target_length_train
eval_config = copy.deepcopy(config)
eval_config.motion.harper_target_length = eval_config.motion.harper_target_length_eval

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


dct_m, idct_m = get_dct_matrix(config.motion.harper_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


classify_fn = torch.nn.NLLLoss()
d = 100
print(f"reg_loss * {d}")

def train_step(harper_motion_input, harper_motion_target, model, optimizer, nb_iter,
               total_iter, max_lr, min_lr):
    in_features = human_joint * 3
    b, seqlen, _ = harper_motion_input.shape

    harper_motion_input_ = harper_motion_input.clone()
    src1, src2 = harper_motion_input_[:, :, :in_features], harper_motion_input_[:, :, in_features:]
    src1 = torch.matmul(dct_m[:, :, :config.motion.harper_input_length], src1.cuda())
    src2 = torch.matmul(dct_m[:, :, :config.motion.harper_input_length], src2.cuda())


    motion_pred1, motion_pred2,alpha_s, alpha_t , beta_s, beta_t= model(src1.cuda(), src2.cuda(),
                                                           )
    reg_loss = torch.mean(torch.relu(-alpha_s) + torch.relu(alpha_s - 1) +
                          torch.relu(-alpha_t) + torch.relu(alpha_t - 1) +
                          torch.relu(-beta_s) + torch.relu(beta_s - 1) +
                          torch.relu(-beta_t) + torch.relu(beta_t - 1))

    motion_pred1 = torch.matmul(idct_m[:, :config.motion.harper_input_length, :], motion_pred1)
    motion_pred2 = torch.matmul(idct_m[:, :config.motion.harper_input_length, :], motion_pred2)

    if config.deriv_output:
        offset1 = harper_motion_input[:, -1:, :in_features].cuda()
        offset2 = harper_motion_input[:, -1:, in_features:].cuda()

        motion_pred1 = motion_pred1[:, :config.motion.harper_target_length] + offset1
        motion_pred2 = motion_pred2[:, :config.motion.harper_target_length] + offset2
    else:
        motion_pred1 = motion_pred1[:, :config.motion.harper_target_length]
        motion_pred2 = motion_pred2[:, :config.motion.harper_target_length]

    b, n, c = harper_motion_target.shape
    motion_pred1 = motion_pred1.reshape(b, n, human_joint, 3).reshape(-1, 3)
    motion_pred2 = motion_pred2.reshape(b, n, robot_joint, 3).reshape(-1, 3)

    harper_motion_target = harper_motion_target.cuda().reshape(b, n, n_joint, 3)  # .reshape(-1, 3)
    h_gt = harper_motion_target[:, :, :human_joint].reshape(-1, 3)
    r_gt = harper_motion_target[:, :, human_joint:].reshape(-1, 3)

    loss_h = torch.mean(torch.norm(motion_pred1 - h_gt, 2, 1))
    loss_r = torch.mean(torch.norm(motion_pred2 - r_gt, 2, 1))


    if config.use_relative_loss:
        motion_h_gt = h_gt.reshape(b, n, human_joint, 3)
        motion_pred1 = motion_pred1.reshape(b, n, human_joint, 3)
        dmotion_pred = gen_velocity(motion_pred1)
        dmotion_hgt = gen_velocity(motion_h_gt)
        dlossh = torch.mean(torch.norm((dmotion_pred - dmotion_hgt).reshape(-1, 3), 2, 1))
        loss_h += dlossh

        motion_r_gt = r_gt.reshape(b, n, robot_joint, 3)
        motion_pred2 = motion_pred2.reshape(b, n, robot_joint, 3)
        dmotion_pred = gen_velocity(motion_pred2)
        dmotion_rgt = gen_velocity(motion_r_gt)
        dlossr = torch.mean(torch.norm((dmotion_pred - dmotion_rgt).reshape(-1, 3), 2, 1))
        loss_r += dlossr
    else:
        loss_r = loss_r.mean()
        loss_h = loss_h.mean()
    reg_loss = reg_loss * d
    loss = loss_r + loss_h + reg_loss

    writer.add_scalar('Loss/loss_all', loss.detach().cpu().numpy(), nb_iter)
    writer.add_scalar('Loss/loss_r', loss_r.detach().cpu().numpy(), nb_iter)
    writer.add_scalar('Loss/loss_h', loss_h.detach().cpu().numpy(), nb_iter)
    writer.add_scalar('Loss/reg_loss', reg_loss.detach().cpu().numpy(), nb_iter)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr, loss_r, loss_h

if __name__=="__main__":
    model = Model(config)
    model.train()
    model.cuda()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.cos_lr_max,
                                 weight_decay=config.weight_decay)

    # ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file, 'train')
    link_file(config.log_file, config.link_log_file)

    print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

    if config.model_pth is not None:
        state_dict = torch.load(config.model_pth)
        model.load_state_dict(state_dict, strict=True)
        print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

    ##### ------ training ------- #####
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.
    ensure_dir(os.path.join(config.snapshot_dir, "./model"))
    while (nb_iter + 1) < (config.cos_lr_total_iters):

        for (harper_motion_input, harper_motion_target) in tqdm(dataloader):
            # B, N, 66   B,T,66
            loss, optimizer, current_lr, loss_h, loss_r = train_step(harper_motion_input, harper_motion_target,
                                                                                model, optimizer,
                                                                                  nb_iter,
                                                                                  config.cos_lr_total_iters,
                                                                                  config.cos_lr_max, config.cos_lr_min)
            avg_loss += loss
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every

                print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.print_loss == 0:
                print(nb_iter + 1)
                print(f"loss {loss}, loss_r {loss_r}, loss_h {loss_h}  regloss {loss-loss_h-loss_r}")
            if (nb_iter + 1) % config.save_every == 0:
                print(nb_iter + 1)
                torch.save(model.state_dict(), config.snapshot_dir + "/model" + '/model-iter-' + str(nb_iter + 1) + '.pth')
                model.eval()
                res_dict = test(eval_config, model, eval_dataloader)
                # print(acc_tmp)
                acc_log.write(''.join(str(nb_iter + 1) + '\n'))
                line = ''
                for key, value in res_dict.items():
                    line += str(key) + ',' + ','.join([str(a) for a in value]) + '\n'
                acc_log.write(''.join(line))
                model.train()

            if (nb_iter + 1) == config.cos_lr_total_iters:
                break
            nb_iter += 1
    acc_log.close()
    writer.close()
    shutil.copyfile("./result/" + args.exp_name, os.path.join(args.work_dir, args.exp_name))
    shutil.copyfile("./result/" + args.exp_name, os.path.join(config.snapshot_dir, args.exp_name))
