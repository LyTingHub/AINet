import argparse
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from network.AINet import AINet
from utils.experiment import (
    action_list,
    build_dataset,
    get_dct_matrix,
    get_out_print_frame,
    load_config,
    set_motion_target_length,
)


def regress_pred(config, model, dataloader, dct_m, idct_m, device, action='all'):
    out_n = config.motion.target_length_eval
    in_features = config.human_joint * 3
    mpjpe_all = np.zeros([out_n])
    mpjpe_human = np.zeros([out_n])
    mpjpe_robot = np.zeros([out_n])
    num_samples = 0

    for batch in dataloader:
        motion_input, motion_target = batch[0].to(device), batch[1]
        batch_size = motion_input.shape[0]
        num_samples += batch_size
        outputs = []
        step = config.motion.target_length_train
        num_step = max(1, (out_n + step - 1) // step)

        for _ in range(num_step):
            with torch.no_grad():
                src1 = torch.matmul(dct_m[:, :, :config.motion.input_length], motion_input[:, :, :in_features])
                src2 = torch.matmul(dct_m[:, :, :config.motion.input_length], motion_input[:, :, in_features:])
                motion_pred1, motion_pred2, _, _, _, _ = model(src1, src2)
                motion_pred1 = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred1)[:, :step, :]
                motion_pred2 = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred2)[:, :step, :]
                output = torch.cat([motion_pred1, motion_pred2], dim=-1)
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            outputs.append(output.reshape(batch_size, step, -1))
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)

        motion_pred = torch.cat(outputs, axis=1)[:, :out_n]
        motion_target = motion_target.detach().cpu()
        _, target_len, _ = motion_target.shape
        motion_pred = motion_pred.reshape(batch_size, target_len, config.n_joint, 3).cpu()
        motion_target = motion_target.reshape(batch_size, target_len, config.n_joint, 3)

        mpjpe_all += torch.sum(torch.mean(torch.norm(motion_target - motion_pred, dim=3), dim=2), dim=0).numpy()
        mpjpe_human += torch.sum(
            torch.mean(
                torch.norm(
                    motion_target[:, :, :config.human_joint] - motion_pred[:, :, :config.human_joint],
                    dim=3,
                ),
                dim=2,
            ),
            dim=0,
        ).numpy()
        mpjpe_robot += torch.sum(
            torch.mean(
                torch.norm(
                    motion_target[:, :, config.human_joint:] - motion_pred[:, :, config.human_joint:],
                    dim=3,
                ),
                dim=2,
            ),
            dim=0,
        ).numpy()

    mpjpe_all /= num_samples
    mpjpe_human /= num_samples
    mpjpe_robot /= num_samples
    out_print_frame = get_out_print_frame(out_n)
    res_dict = {
        'mpjpe_all': mpjpe_all[out_print_frame] * 1000,
        'mpjpe_human': mpjpe_human[out_print_frame] * 1000,
        'mpjpe_robot': mpjpe_robot[out_print_frame] * 1000,
    }
    print(f'Error at each output frame:\nFrame number: {out_print_frame}\n{action} Error: {res_dict}')
    return res_dict


def test(config, model, dataloader, device=None, action='all'):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dct_np, idct_np = get_dct_matrix(config.motion.input_length_dct)
    dct_m = torch.tensor(dct_np).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_np).float().to(device).unsqueeze(0)
    return regress_pred(config, model, dataloader, dct_m, idct_m, device, action=action)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['harper', 'comad', 'chico'], default='harper')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--model-pth', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--action', type=str, default='all')
    parser.add_argument('--action-wise', action='store_true')
    parser.add_argument('--output', type=str, default=None, help='Optional file for comma-separated metrics.')
    return parser.parse_args()


def evaluate_one(args, config, model, device, action):
    eval_config = copy.deepcopy(config)
    set_motion_target_length(eval_config, eval_config.motion.target_length_eval)
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        eval_config,
        'test',
        eval_config.motion.target_length_eval,
        action=action,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return test(eval_config, model, dataloader, device=device, action=action)


def main():
    args = parse_args()
    config = load_config(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AINet(config).to(device)
    state_dict = torch.load(args.model_pth, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    actions = action_list(args.dataset) if args.action_wise else [args.action]
    output_file = open(args.output, 'w', encoding='utf-8') if args.output else None
    try:
        for action in actions:
            res_dict = evaluate_one(args, config, model, device, action)
            if output_file:
                output_file.write(str(action) + '\n')
                for key, value in res_dict.items():
                    output_file.write(str(key) + ',' + ','.join([str(a) for a in value]) + '\n')
    finally:
        if output_file:
            output_file.close()


if __name__ == '__main__':
    main()
