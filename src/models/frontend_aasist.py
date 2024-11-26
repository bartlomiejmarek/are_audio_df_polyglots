import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.whisper_aasist import GraphAttentionLayer, HtrgGraphAttentionLayer, Residual_block, GraphPool

from src import frontends


class FrontendAASIST(nn.Module):
    @property
    def frontend_out_dim(self):
        return 3*128

    def __init__(self, device: str, num_classes: int = 1, **kwargs):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]


        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend = frontends.get_frontend(frontend_name)

        print(f"Using {frontend_name} frontend, {self.frontend_out_dim}")

        self.LL = nn.Linear(self.frontend_out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute front-end
        x = self._compute_frontend(x)

        x = self.LL(x)  # (bs,frame_number,feat_out_dim)

        # -------pre-trained Wav2vec model fine-tuning ------------------------##

        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)

        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)

        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)

        # ------------SAP for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)

        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)

        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SAP for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)

        output = self.out_layer(last_hidden)

        return output


    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        frontend = frontend.squeeze(1).permute(0,2,1)
        return frontend # (bs, frames, n_lfcc)


if __name__ == "__main__":
    print("Definition of model")
    device = "cuda"

    input_channels = 1
    config = {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }

    def count_parameters(model) -> int:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params
    model = FrontendAASIST(device=device, input_channels=1, frontend_algorithm=["lfcc"])
    model = model.to(device)
    print(count_parameters(model))
    import torch
    x = torch.rand(1, 64_600).to("cuda")
    model(x)