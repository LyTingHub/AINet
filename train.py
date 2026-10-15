import argparse
import copy
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network.AINet import AINet as Model
from test import test
from utils.experiment import (
    apply_runtime_options,
    build_dataset,
    get_dct_matrix,
    load_config,
    set_motion_target_length,
    set_seed,
)
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import ensure_dir, link_file

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['harper', 'comad', 'chico'], default='harper')
    parser.add_argument('--data-root', type=str, required=True, help='Dataset root directory.')
    parser.add_argument('--exp-name', type=str, default=None, help='Result log file name.')
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--temporal-only', action='store_true')
    parser.add_argument('--layer-norm-axis', type=str, default='spatial')
    parser.add_argument('--with-normalization', action='store_true', default=None)
    parser.add_argument('--without-normalization', action='store_false', dest='with_normalization')
    parser.add_argument('--spatial-fc', action='store_true')
    parser.add_argument('--num', type=int, default=24, help='Kept for compatibility with ablation scripts.')
    parser.add_argument('--weight', type=float, default=1.0, help='Multiplier before the paper reg-loss factor 100.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--total-iters', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--resume', type=str, default=None, help='Optional checkpoint to resume from.')
    return parser.parse_args()


def update_lr_multistep(nb_iter, optimizer):
    current_lr = 1e-5 if nb_iter > 30000 else 3e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def gen_velocity(motion):
    return motion[:, 1:] - motion[:, :-1]


def train_step(batch, model, optimizer, config, dct_m, idct_m, device, nb_iter, reg_weight):
    motion_input, motion_target = batch[0].to(device), batch[1].to(device)
    in_features = config.human_joint * 3

    src1 = torch.matmul(dct_m[:, :, :config.motion.input_length], motion_input[:, :, :in_features])
    src2 = torch.matmul(dct_m[:, :, :config.motion.input_length], motion_input[:, :, in_features:])
    motion_pred1, motion_pred2, alpha_s, alpha_t, beta_s, beta_t = model(src1, src2)

    reg_loss = torch.mean(
        torch.relu(-alpha_s) + torch.relu(alpha_s - 1)
        + torch.relu(-alpha_t) + torch.relu(alpha_t - 1)
        + torch.relu(-beta_s) + torch.relu(beta_s - 1)
        + torch.relu(-beta_t) + torch.relu(beta_t - 1)
    )

    motion_pred1 = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred1)
    motion_pred2 = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred2)

    if config.deriv_output:
        motion_pred1 = motion_pred1[:, :config.motion.target_length] + motion_input[:, -1:, :in_features]
        motion_pred2 = motion_pred2[:, :config.motion.target_length] + motion_input[:, -1:, in_features:]
    else:
        motion_pred1 = motion_pred1[:, :config.motion.target_length]
        motion_pred2 = motion_pred2[:, :config.motion.target_length]

    batch_size, target_len, _ = motion_target.shape
    motion_pred1 = motion_pred1.reshape(batch_size, target_len, config.human_joint, 3).reshape(-1, 3)
    motion_pred2 = motion_pred2.reshape(batch_size, target_len, config.robot_joint, 3).reshape(-1, 3)

    motion_target = motion_target.reshape(batch_size, target_len, config.n_joint, 3)
    h_gt = motion_target[:, :, :config.human_joint].reshape(-1, 3)
    r_gt = motion_target[:, :, config.human_joint:].reshape(-1, 3)
    loss_h = torch.mean(torch.norm(motion_pred1 - h_gt, 2, 1))
    loss_r = torch.mean(torch.norm(motion_pred2 - r_gt, 2, 1))

    if config.use_relative_loss:
        motion_h_gt = h_gt.reshape(batch_size, target_len, config.human_joint, 3)
        motion_pred1 = motion_pred1.reshape(batch_size, target_len, config.human_joint, 3)
        loss_h += torch.mean(torch.norm((gen_velocity(motion_pred1) - gen_velocity(motion_h_gt)).reshape(-1, 3), 2, 1))

        motion_r_gt = r_gt.reshape(batch_size, target_len, config.robot_joint, 3)
        motion_pred2 = motion_pred2.reshape(batch_size, target_len, config.robot_joint, 3)
        loss_r += torch.mean(torch.norm((gen_velocity(motion_pred2) - gen_velocity(motion_r_gt)).reshape(-1, 3), 2, 1))

    reg_loss = reg_weight * reg_loss
    loss = loss_r + loss_h + reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_lr = update_lr_multistep(nb_iter, optimizer)
    return loss.item(), current_lr, loss_r.detach(), loss_h.detach(), reg_loss.detach()


def main():
    args = parse_args()
    config = load_config(args.dataset)
    apply_runtime_options(config, args)
    set_seed(args.seed)
    set_motion_target_length(config, config.motion.target_length_train)

    exp_name = args.exp_name or f'{args.dataset}_result.txt'
    ensure_dir('./result')
    acc_log = open(os.path.join('./result', exp_name), 'a', encoding='utf-8')
    acc_log.write('Seed : ' + str(args.seed) + '\n')

    train_dataset = build_dataset(args.dataset, args.data_root, config, 'train', config.motion.target_length_train)
    eval_split = 'validation' if args.dataset == 'chico' else 'test'
    eval_config = copy.deepcopy(config)
    set_motion_target_length(eval_config, eval_config.motion.target_length_eval)
    eval_dataset = build_dataset(args.dataset, args.data_root, eval_config, eval_split, eval_config.motion.target_length_eval)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            'The training dataloader is empty. Reduce --batch-size or provide more training samples.'
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dct_np, idct_np = get_dct_matrix(config.motion.input_length_dct)
    dct_m = torch.tensor(dct_np).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_np).float().to(device).unsqueeze(0)

    ensure_dir(config.snapshot_dir)
    ensure_dir(os.path.join(config.snapshot_dir, 'model'))
    writer = SummaryWriter(config.snapshot_dir)

    model = Model(config).to(device)
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.cos_lr_max, weight_decay=config.weight_decay)
    logger = get_logger(config.log_file, 'train')
    link_file(config.log_file, config.link_log_file)
    print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))
    print_and_log_info(logger, '>>> total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0
    model.train()
    while nb_iter < config.cos_lr_total_iters:
        for batch in tqdm(dataloader):
            loss, current_lr, loss_r, loss_h, reg_loss = train_step(
                batch, model, optimizer, config, dct_m, idct_m, device, nb_iter, args.weight * 100
            )
            writer.add_scalar('Loss/loss_all', loss, nb_iter)
            writer.add_scalar('Loss/loss_r', loss_r.item(), nb_iter)
            writer.add_scalar('Loss/loss_h', loss_h.item(), nb_iter)
            writer.add_scalar('Loss/reg_loss', reg_loss.item(), nb_iter)
            writer.add_scalar('LR/train', current_lr, nb_iter)
            avg_loss += loss
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss /= config.print_every
                avg_lr /= config.print_every
                print_and_log_info(logger, f'Iter {nb_iter + 1} Summary:')
                print_and_log_info(logger, f'\t lr: {avg_lr} \t Training loss: {avg_loss}')
                avg_loss = 0.0
                avg_lr = 0.0

            if (nb_iter + 1) % config.print_loss == 0:
                print(f'Iter {nb_iter + 1}: loss {loss}, loss_r {loss_r}, loss_h {loss_h}, reg_loss {reg_loss}')

            if (nb_iter + 1) % config.save_every == 0:
                ckpt_path = os.path.join(config.snapshot_dir, 'model', f'model-iter-{nb_iter + 1}.pth')
                torch.save(model.state_dict(), ckpt_path)
                model.eval()
                res_dict = test(eval_config, model, eval_dataloader, device=device)
                acc_log.write(str(nb_iter + 1) + '\n')
                for key, value in res_dict.items():
                    acc_log.write(str(key) + ',' + ','.join([str(a) for a in value]) + '\n')
                acc_log.flush()
                model.train()

            nb_iter += 1
            if nb_iter >= config.cos_lr_total_iters:
                break

    acc_log.close()
    writer.close()


if __name__ == '__main__':
    main()
