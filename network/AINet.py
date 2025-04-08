import copy

import torch
from torch import nn
from network.mlp import build_mlps
from einops.layers.torch import Rearrange
from network.LGSI import Local_Global_Spatial_Interaction as AIS

class AINet(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(AINet, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.in_features1 = self.config.motion.dim1
        self.in_features2 = self.config.motion.dim2
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')
        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        '''mlp block'''
        human_mlp_config = copy.deepcopy(self.config.motion_mlp)
        human_mlp_config.num_layers = 20
        self.human_pose_module1 = build_mlps(human_mlp_config)
        human_mlp_config.spatial_fc_only = True
        human_mlp_config.num_layers = 4
        self.human_pose_module2 = build_mlps(human_mlp_config)

        human_mlp_config = copy.deepcopy(self.config.motion_mlp)
        human_mlp_config.num_layers = 6
        self.human_traj_module1 = build_mlps(human_mlp_config)
        human_mlp_config.spatial_fc_only = True
        human_mlp_config.num_layers = 3
        self.human_traj_module2 = build_mlps(human_mlp_config)

        mlp_config2 = copy.deepcopy(self.config.motion_mlp)
        mlp_config2.hidden_dim = self.in_features2  # + joint_dim
        mlp_config2.num_layers = 20
        self.robot_pose_module1 = build_mlps(mlp_config2)
        mlp_config2.spatial_fc_only = True
        mlp_config2.num_layers = 4
        self.robot_pose_module2 = build_mlps(mlp_config2)

        mlp_config2 = copy.deepcopy(self.config.motion_mlp)
        mlp_config2.hidden_dim = self.in_features2
        mlp_config2.num_layers = 6
        self.robot_traj_module1 = build_mlps(mlp_config2)
        mlp_config2.spatial_fc_only = True
        mlp_config2.num_layers = 3
        self.robot_traj_module2 = build_mlps(mlp_config2)

        '''LSI'''
        local_joint_dim = 100
        self.LSI = AIS(in_features1=self.in_features1, in_features2=self.in_features2,
                               embed_dim=local_joint_dim, XIA_dim=local_joint_dim, XIA_head=25, XIA_axis='spatial')

        '''GSI'''
        global_joint_dim_ = 24
        self.GSI = AIS(in_features1=self.in_features1, in_features2=self.in_features2,
                                embed_dim=global_joint_dim_,
                                XIA_dim=global_joint_dim_, XIA_head=6, XIA_axis='spatial')

        if self.temporal_fc_in:
            self.motion_fc_in1 = nn.Linear(self.config.motion.harper_input_length_dct,
                                           self.config.motion.harper_input_length_dct)
            self.motion_fc_in2 = nn.Linear(self.config.motion.harper_input_length_dct,
                                           self.config.motion.harper_input_length_dct)
        else:
            self.motion_fc_in1 = nn.Linear(self.in_features1, self.in_features1)  
            self.motion_fc_in1_ = nn.Linear(self.in_features1, self.in_features1) 

            self.motion_fc_in2 = nn.Linear(self.in_features2, self.in_features2)
            self.motion_fc_in2_ = nn.Linear(self.in_features2, self.in_features2)

        if self.temporal_fc_out:
            self.motion_fc_out1 = nn.Linear(self.config.motion.harper_input_length_dct,
                                            self.config.motion.harper_input_length_dct)
            self.motion_fc_out2 = nn.Linear(self.config.motion.harper_input_length_dct,
                                            self.config.motion.harper_input_length_dct)
        else:
            self.motion_fc_out1 = nn.Linear(self.in_features1, self.in_features1)
            self.motion_fc_out1_ = nn.Linear(self.in_features1, self.in_features1)
            self.motion_fc_out2 = nn.Linear(self.in_features2, self.in_features2)
            self.motion_fc_out2_ = nn.Linear(self.in_features2, self.in_features2)

        self.context_LN =  nn.LayerNorm(global_joint_dim_*2)

        self.coemlp1 = nn.Sequential(
            nn.Linear(global_joint_dim_ * 2, global_joint_dim_),
            nn.Linear(global_joint_dim_, 1)
        )
        self.coemlp2 = nn.Sequential(
            nn.Linear(global_joint_dim_ * 2, global_joint_dim_),
            nn.Linear(global_joint_dim_, 1)
        )
        self.coemlp3 = nn.Sequential(
            nn.Linear(global_joint_dim_ * 2, global_joint_dim_),
            nn.Linear(global_joint_dim_, 1)
        )
        self.coemlp4 = nn.Sequential(
            nn.Linear(global_joint_dim_ * 2, global_joint_dim_),
            nn.Linear(global_joint_dim_, 1)
        )

        self.pose2traj_human = nn.Sequential(  # 63 69
            nn.Linear(self.in_features1, 16),
            nn.Linear(16, 3)
        )
        self.pose2traj_robot = nn.Sequential(  # 63 69
            nn.Linear(self.in_features2, 16),
            nn.Linear(16, 3)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out1_.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out1_.bias, 0)
        nn.init.xavier_uniform_(self.motion_fc_out2_.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out2_.bias, 0)

        nn.init.constant_(self.context_LN.bias, 0.5)

    def forward(self, motion_input1, motion_input2, nb_iter=10000):
        b, seqlen, _ = motion_input1.shape
        
        # Motion Encoder
        if self.temporal_fc_in:
            motion_feats1 = self.arr0(motion_input1)
            motion_feats1 = self.motion_fc_in1(motion_feats1)

            motion_feats2 = self.arr0(motion_input2)
            motion_feats2 = self.motion_fc_in2(motion_feats2)
        else:
            motion_feats1 = self.motion_fc_in1(motion_input1)
            motion_feats1 = self.motion_fc_in1_(motion_feats1)
            motion_feats1 = self.arr0(motion_feats1)

            motion_feats2 = self.motion_fc_in2(motion_input2)
            motion_feats2 = self.motion_fc_in2_(motion_feats2)
            motion_feats2 = self.arr0(motion_feats2)
            
        # LGSI
        p1_emd_G, p2_emd_G, p1_XIA_G, p2_XIA_G = self.GSI(motion_feats1.clone(), motion_feats2.clone())
        _, _, p1_XIA_L, p2_XIA_L = self.LSI(motion_feats1.clone(), motion_feats2.clone())
        context = torch.cat([p1_emd_G, p2_emd_G], dim=-1)  #
        context = self.context_LN(context)  #  + mean_shift
        alpha, alpha2, beta, beta2 = self.coemlp1(context), self.coemlp2(
            context), self.coemlp3(context), self.coemlp4(context)  # b, n, 1
        alpha, beta = alpha.permute(0, 2, 1), beta.permute(0, 2, 1)
        alpha2, beta2 = alpha2.permute(0, 2, 1), beta2.permute(0, 2, 1)  # b,1,n

        alpha_L = torch.clamp(alpha, 0, 1)
        alpha_G = torch.clamp(alpha2, 0, 1)
        alpha_Lum = alpha_L + alpha_G + 1e-8
        alpha_L = alpha_L / alpha_Lum
        alpha_G = alpha_G / alpha_Lum

        beta_L = torch.clamp(beta, 0, 1)
        beta_G = torch.clamp(beta2, 0, 1)
        beta_Lum = beta_L + beta_G + 1e-8
        beta_L = beta_L / beta_Lum
        beta_G = beta_G / beta_Lum
        motion_feats1 = alpha_L * p1_XIA_L + alpha_G * p1_XIA_G
        motion_feats2 = beta_G * p2_XIA_G + beta_L * p2_XIA_L

        motion_feats1 = self.human_traj_module2(motion_feats1)
        motion_feats1 = self.human_traj_module1(motion_feats1)

        motion_feats2 = self.robot_traj_module2(motion_feats2)
        motion_feats2 = self.robot_traj_module1(motion_feats2)

        traj1 = self.pose2traj_human(motion_feats1.permute(0, 2, 1))
        traj2 = self.pose2traj_robot(motion_feats2.permute(0, 2, 1))
        traj1, traj2 = traj1.reshape(b, seqlen, 1, 3), traj2.reshape(b, seqlen, 1, 3)
        motion_feats1 = motion_feats1.reshape(b, seqlen, -1, 3) + traj1
        motion_feats2 = motion_feats2.reshape(b, seqlen, -1, 3) + traj2
        motion_feats1 = motion_feats1.reshape(b, -1, seqlen)
        motion_feats2 = motion_feats2.reshape(b, -1, seqlen)
        motion_feats1 = self.human_pose_module2(motion_feats1)
        motion_feats1 = self.human_pose_module1(motion_feats1)

        motion_feats2 = self.robot_pose_module2(motion_feats2)
        motion_feats2 = self.robot_pose_module1(motion_feats2)

        if self.temporal_fc_out:
            motion_feats1 = self.motion_fc_out1(motion_feats1)
            motion_feats1 = self.arr1(motion_feats1)

            motion_feats2 = self.motion_fc_out2(motion_feats2)
            motion_feats2 = self.arr1(motion_feats2)
        else:
            motion_feats1 = self.arr1(motion_feats1)
            motion_feats1 = self.motion_fc_out1(motion_feats1)
            motion_feats1 = self.motion_fc_out1_(motion_feats1)
            motion_feats2 = self.arr1(motion_feats2)
            motion_feats2 = self.motion_fc_out2(motion_feats2)
            motion_feats2 = self.motion_fc_out2_(motion_feats2)
        return motion_feats1, motion_feats2, alpha, alpha2, beta, beta2



