from torch import nn

class Local_Global_Spatial_Interaction(nn.Module):
    def __init__(self, in_features1=45, in_features2=27, embed_dim=32, XIA_dim=50, XIA_head=5, XIA_axis='spatial'):
        super(Local_Global_Spatial_Interaction, self).__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.embed_dim = embed_dim
        self.p1_mlp = nn.Sequential(
            nn.Linear(self.in_features1, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.p2_mlp = nn.Sequential(
            nn.Linear(self.in_features2, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.p1_mlp_rev = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, self.in_features1),
        )
        self.p2_mlp_rev = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, self.in_features2),
        )

        self.update_p1 = XIA(embed_dim=XIA_dim, nb_h=XIA_head)  # seq=50
        self.update_p2 = XIA(embed_dim=XIA_dim, nb_h=XIA_head)
        self.XIA_axis = XIA_axis
        assert XIA_axis in ['spatial', 'temporal']

    def forward(self, motion_feats1, motion_feats2):
        if self.XIA_axis == 'spatial':
            # encode
            p1_XIA = self.p1_mlp(motion_feats1.permute(0, 2, 1))
            p2_XIA = self.p2_mlp(motion_feats2.permute(0, 2, 1))  # b, T, 32
            # XIA
            p1_XIA_ = self.update_p1(p1_XIA, p2_XIA)  # 对spatial
            p2_XIA_ = self.update_p2(p2_XIA, p1_XIA)  # b, T, 16
            # decode
            p1_XIA_S = self.p1_mlp_rev(p1_XIA_).permute(0, 2, 1)
            p2_XIA_S = self.p2_mlp_rev(p2_XIA_).permute(0, 2, 1)
            return p1_XIA, p2_XIA, p1_XIA_S+motion_feats1, p2_XIA_S+motion_feats2
        elif self.XIA_axis == 'temporal':
            p1_XIA = self.p1_mlp(motion_feats1.permute(0, 2, 1))
            p2_XIA = self.p2_mlp(motion_feats2.permute(0, 2, 1))  # b, T, 16
            p1_XIA_ = self.update_p1(p1_XIA.permute(0, 2, 1), p2_XIA.permute(0, 2, 1))
            p2_XIA_ = self.update_p2(p2_XIA.permute(0, 2, 1), p1_XIA.permute(0, 2, 1))  # b, T, 16
            p1_XIA_T = self.p1_mlp_rev(p1_XIA_.permute(0, 2, 1)).permute(0, 2, 1)  # 256, 60, 63
            p2_XIA_T = self.p2_mlp_rev(p2_XIA_.permute(0, 2, 1)).permute(0, 2, 1)
            return p1_XIA, p2_XIA, p1_XIA_T+motion_feats1, p2_XIA_T+motion_feats2


class XIA(nn.Module):
    def __init__(self, embed_dim=256, nb_h=8, dropout=0.1):
        super(XIA, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nb_h, dropout=dropout, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc[1].weight, gain=1e-8)
        nn.init.constant_(self.fc[1].bias, 0)
        nn.init.xavier_uniform_(self.fc[2].weight, gain=1e-8)
        nn.init.constant_(self.fc[2].bias, 0)

    def forward(self, k1, k2):
        query = k2
        key = k1
        value = k1

        k = self.self_attn(query, key, value=value)[0]
        k = self.fc(k)
        k = k + k1
        return k