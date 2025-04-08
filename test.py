import argparse
import numpy as np
from config.harper_config import config as config
from network.AINet import AINet
from dataset.harper_3d_2 import Harper3D
import copy
import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#5', "#8", '#11', '#14', "#17", "#20", "#23", "#26", "#29"]
n_joint = 44
human_joint = 21


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
in_features = human_joint * 3

def regress_pred(model, pbar, num_samples, action='all'):
    out_n = config.motion.harper_target_length_eval
    mpjpe_all, mpjpe_human, mpjpe_robot = np.zeros([out_n]), np.zeros([out_n]), np.zeros([out_n])

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b, n, c = motion_input.shape
        num_samples += b
        outputs = []
        step = config.motion.harper_target_length_train
        if step >= out_n:
            num_step = 1
        else:
            num_step = config.motion.harper_target_length_eval // config.motion.harper_target_length_train + 1
        for idx in range(num_step):
            with torch.no_grad():
                motion_input_ = motion_input.clone()
                src1, src2 = motion_input_[:, :, :in_features], motion_input_[:, :, in_features:]
                src1 = torch.matmul(dct_m[:, :, :config.motion.harper_input_length], src1.cuda())
                src2 = torch.matmul(dct_m[:, :, :config.motion.harper_input_length], src2.cuda())

                motion_pred1, motion_pred2, _, _, _, _ = model(src1, src2)
                motion_pred1 = torch.matmul(idct_m[:, :config.motion.harper_input_length, :], motion_pred1)[:, :step, :]
                motion_pred2 = torch.matmul(idct_m[:, :config.motion.harper_input_length, :], motion_pred2)[:, :step, :]

                output = torch.cat([motion_pred1, motion_pred2], dim=-1)
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            output = output.reshape(-1, n_joint * 3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:, :out_n]  # b, 30, 66

        motion_target = motion_target.detach().cpu()
        b, nt, c = motion_target.shape
        motion_pred = motion_pred.reshape(b, nt, n_joint, 3).cpu()
        motion_target = motion_target.reshape(b, nt, n_joint, 3)

        tmp_joi = torch.sum(torch.mean(torch.norm(motion_target - motion_pred, dim=3), dim=2), dim=0)
        mpjpe_all += tmp_joi.cpu().data.numpy()

        tmp_joi = torch.sum(torch.mean(torch.norm(motion_target[:, :, :human_joint] - motion_pred[:, :, :human_joint], dim=3), dim=2), dim=0)
        mpjpe_human += tmp_joi.cpu().data.numpy()

        tmp_joi = torch.sum(torch.mean(torch.norm(motion_target[:, :, human_joint:] - motion_pred[:, :, human_joint:], dim=3), dim=2), dim=0)
        mpjpe_robot += tmp_joi.cpu().data.numpy()

    mpjpe_all, mpjpe_human, mpjpe_robot = mpjpe_all / num_samples, mpjpe_human / num_samples, mpjpe_robot / num_samples

    out_print_frame = get_out_print_frame(out_n)
    res_dic = {"mpjpe_all": mpjpe_all[out_print_frame]*1000,
               "mpjpe_human": mpjpe_human[out_print_frame]*1000,
               "mpjpe_robot": mpjpe_robot[out_print_frame]*1000}

    print(f'Error at each output frame:\n Frame number:{out_print_frame}\n {action} Error:{res_dic}')

    return res_dic


def test(config, model, dataloader):
    m_p3d_harper = np.zeros([config.motion.harper_target_length])
    titles = np.array(range(config.motion.harper_target_length)) + 1
    num_samples = 0

    pbar = dataloader
    res_dict = regress_pred(model, pbar, num_samples)
    return res_dict


def get_out_print_frame(out_n):
    if out_n == 30:
        out_print_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    elif out_n == 15:
        out_print_frame = [2, 5, 8, 11, 14]
    elif out_n == 10:
        out_print_frame = range(10)
    elif out_n == 25:  # ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
        out_print_frame = [1, 3, 7, 9, 13, 17, 21, 24]
    else:
        out_print_frame = [out_n - 1]
    return out_print_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()


    model = AINet(config)
    args.model_pth = r''
    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    data_root = r'E:\Data\30hz'


    eval_config = copy.deepcopy(config)
    action_list = ['act1_0', 'act1_45', 'act1_90', 'act1_180', 'act2', 'act3', 'act4', 'act5', 'act6','act7', 'act8', 'act9', 'act10', 'act11', 'act12']
    acc_log = open("./best_ckpt_HARPER.txt", 'a+', encoding='utf-8')
    for action in action_list:
        print(action)
        eval_config.motion.harper_target_length = eval_config.motion.harper_target_length_eval
        eval_dataset = Harper3D(data_path=data_root, split="test", n_input=eval_config.motion.harper_input_length,
                                n_output=eval_config.motion.harper_target_length_eval, sample=1, action=[action])
        eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=4)
        acc_log.write(action+'\t'+str(len(eval_dataset))+'\n')

        res_dict = test(eval_config, model, eval_dataloader)
        print(res_dict)
        line = ''
        for key, value in res_dict.items():
            line += str(key) + ',' + ','.join([str(a) for a in value]) + '\n'
        acc_log.write(''.join(line))
