import torch.nn as nn
import torch
from model.modules.conv.spiralconv import SpiralConv
from occ_label_preparation.seg import *


# Init model weights
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0, 4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in), )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN"xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0) // 3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):

    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out


class H2ONet_GlobRotReg(nn.Module):

    def __init__(self):
        super(H2ONet_GlobRotReg, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, 256),
                                      nn.ReLU(inplace=True), nn.Linear(256, 6))

    def forward(self, j_x, r_x):
        x = j_x + r_x
        B, C = x.size(0), x.size(1)
        x = x.view(B, C, -1)  # (B, C, HW)
        x = self.conv_block(x)  # (B, C, HW)
        x = x.view(B, -1)  # (B, CHW)
        x = self.pre_block(x)
        pred_rot = self.fc_block(x)

        return pred_rot


class H2ONet_Decoder(nn.Module):

    def __init__(self, cfg, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(H2ONet_Decoder, self).__init__()
        self.cfg = cfg
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[-1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True)
        self.rot_reg = H2ONet_GlobRotReg()
        self.init_weights()

    def init_weights(self):
        self.rot_reg.apply(init_weights)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x, j_mid, r_mid):
        pred_glob_rot = self.rot_reg(j_mid.detach(), r_mid)
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)  # (B, N, C)
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])

        pred = self.head(x)

        return pred, pred_glob_rot


class OccPredNet(nn.Module):

    def __init__(self):
        super(OccPredNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.base_layer = nn.Sequential(
            nn.Linear(5 * 4 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.finger_cls_layer_list = nn.ModuleList()
        for i in range(5):
            self.finger_cls_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
            )
            self.finger_cls_layer_list.append(self.finger_cls_layer)

        self.glob_cls_layer = nn.Sequential(
            nn.Linear(64 * 5 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        B = x.size(0)
        thumb_feat = torch.index_select(x, dim=1, index=torch.IntTensor(thumb_joints_idx).to(x.device))  # (B, 4, C)
        index_feat = torch.index_select(x, dim=1, index=torch.IntTensor(index_joints_idx).to(x.device))  # (B, 4, C)
        middle_feat = torch.index_select(x, dim=1, index=torch.IntTensor(middle_joints_idx).to(x.device))  # (B, 4, C)
        ring_feat = torch.index_select(x, dim=1, index=torch.IntTensor(ring_joints_idx).to(x.device))  # (B, 4, C)
        little_feat = torch.index_select(x, dim=1, index=torch.IntTensor(little_joints_idx).to(x.device))  # (B, 4, C)

        finger_feat = torch.cat((thumb_feat, index_feat, middle_feat, ring_feat, little_feat), dim=0).permute(0, 2, 1)  # (5B, C=256, 4)
        finger_feat = self.pre_layer(finger_feat)  # (5B, C=64, 4)
        finger_feat = torch.cat(torch.chunk(finger_feat, 5, dim=0), dim=1).view(B, -1)  # (B, 5*64*4)
        finger_feat = self.base_layer(finger_feat)  # (B, 256)
        pred_finger_occ = []
        for i in range(5):
            finger_occ = self.finger_cls_layer_list[i](finger_feat)
            pred_finger_occ.append(finger_occ)  # (B, 2)

        palm_feat = torch.index_select(x, dim=1, index=torch.IntTensor(palm_joints_idx).to(x.device))  # (B, 5, C)
        palm_feat = self.pre_layer(palm_feat.permute(0, 2, 1)).view(B, -1)  # (B, 64*5)
        glob_feat = torch.cat((finger_feat, palm_feat), dim=1)  # (B, 2C=512)
        # print(finger_feat.size(), palm_feat.size())
        pred_glob_occ = self.glob_cls_layer(glob_feat)  # (B, 2)

        pred_occ = torch.stack((pred_finger_occ + [pred_glob_occ]), dim=-1)  # (B, 2, 6)

        return pred_occ


class OffsetPredNet(nn.Module):

    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        super(OffsetPredNet, self).__init__()
        self.block = nn.Sequential(
            meshconv(2 * in_channels, 2 * in_channels, indices),
            nn.ReLU(inplace=True),
            meshconv(2 * in_channels, in_channels, indices),
            nn.ReLU(inplace=True),
            meshconv(in_channels, in_channels, indices),
            nn.ReLU(inplace=True),
        )
        self.head = meshconv(in_channels, out_channels, indices)
        indices_list = [thumb_verts_idx, index_verts_idx, middle_verts_idx, ring_verts_idx, little_verts_idx]
        self.finger_index = [torch.from_numpy(np.array(i)).long().cuda() for i in indices_list]

    def forward(self, x, pred_occ):
        # pred_occ: (3B, 2, 6)
        pred_occ = torch.softmax(pred_occ, dim=1)  # (3B, 2, 6)
        pred_finger_occ = torch.chunk(pred_occ[:, 0, :5], 5, dim=1)  # 5 * (3B, 1)

        weights = torch.ones((x.size(0), 778)).to(x.device)  # (3B, 778)
        for index, non_occ_prob in zip(self.finger_index, pred_finger_occ):
            B, N = x.size(0), index.size(0)
            verts_index = index.unsqueeze(0).repeat(B, 1)  # (num_verts,) -> (3B, num_verts)
            batch_idx = torch.arange(B, dtype=torch.long, device=x.device).repeat_interleave(N)
            non_occ_prob = non_occ_prob.repeat(1, N).flatten()  # (3B, 1) -> (3B, num_verts) -> (num_verts * 3B,)

            verts_index = verts_index.flatten()  # (3B, num_verts) -> (num_verts * 3B)
            weights[batch_idx, verts_index] = non_occ_prob

        weights = torch.stack(torch.chunk(weights, 3, dim=0), dim=-1).unsqueeze(-1)  # (3B, 778) -> (B, 778, 3, 1)
        weights = torch.softmax(weights, dim=2)  # (B, 778, 3, 1)
        # x: (3B, 778, C)
        weighted_x = torch.stack(torch.chunk(x, 3, dim=0), dim=2)  # (3B, 778, C) -> (B, 778, 3, C)
        weighted_x = torch.sum(weights * weighted_x, dim=2)  # (B, 778, 3, 1) * (B, 778, 3, C) -> (B, 778, C)

        x = torch.cat((weighted_x, x[:x.size()[0] // 3, ...]), dim=2)  # (B, 778, C) -> (B, 778, 2C)
        x = self.block(x)
        pred_offset = self.head(x)  # (B, 778, 3)

        return pred_offset


class GlobRotReg_MF(nn.Module):

    def __init__(self):
        super(GlobRotReg_MF, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.pre_block = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        self.fc_block1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
            # nn.Linear(512, 16 * 6),
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )

    def forward(self, j_x, r_x, pred_occ):
        # pred_occ: (3B, 2, 6)
        x = j_x + r_x
        B = x.size(0)
        x = x.view(x.size()[0], x.size()[1], -1)  # (3B, C=1024, HW=16)
        x = self.conv_block(x)  # (3B, C=512, HW=16)
        x = x.view(x.size()[0], -1)  # (3B, CHW)
        x = self.pre_block(x)  # (3B, C)

        x_0, x_1, x_2 = torch.chunk(x, 3, dim=0)  # (B, C)

        pred_glob_rot = self.fc_block1(x)

        pred_occ = torch.argmax(pred_occ, dim=1)[:, :5]  # (3B, 5), 0: non-occ, 1: occ
        pred_glob_occ = torch.prod(pred_occ, dim=1, keepdim=True)  # (3B, 1)
        pred_glob_occ = 1 - pred_glob_occ  # (3B, 1), 1: non-occ, 0: occ
        pred_glob_occ_0, pred_glob_occ_1, pred_glob_occ_2 = torch.chunk(pred_glob_occ, 3, dim=0)  # (B, 1)

        weighted_x_0 = pred_glob_occ_0 * x_0 + (1 - pred_glob_occ_0) * (1 - pred_glob_occ_1) * (1 - pred_glob_occ_2) * x_0
        weighted_x_1 = (1 - pred_glob_occ_0) * pred_glob_occ_1 * x_1
        weighted_x_2 = (1 - pred_glob_occ_0) * (1 - pred_glob_occ_1) * pred_glob_occ_2 * x_2

        x_offset = torch.cat((x_0, weighted_x_0 + weighted_x_1 + weighted_x_2), dim=1)  # (B, 3)
        pred_glob_rot_0 = self.fc_block2(x_offset)

        return pred_glob_rot, pred_glob_rot_0


class H2ONet_MF_Decoder(nn.Module):

    def __init__(self, cfg, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        super(H2ONet_MF_Decoder, self).__init__()
        self.cfg = cfg
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[-1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True)
        self.occ_pred = OccPredNet()
        self.offset_pred = OffsetPredNet(in_channels=self.out_channels[0], out_channels=3, indices=self.spiral_indices[0], meshconv=meshconv)
        self.rot_reg = GlobRotReg_MF()
        self.init_weights()

    def init_weights(self):
        self.occ_pred.apply(init_weights)
        self.offset_pred.apply(init_weights)
        self.rot_reg.apply(init_weights)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x, j_mid, r_mid):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)  # (B, N, C)

        pred_occ = self.occ_pred(x.detach())

        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        # (B, 98, 256), (B, 195, 128), (B, 389, 64), (B, 778, 32)
        mid_de_feat = []
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
            mid_de_feat.append(x)

        # verts xyz decoder
        pred = self.head(x)

        # verts offset decoder
        pred_offset = self.offset_pred(x=mid_de_feat[-1], pred_occ=pred_occ.detach())
        pred[:x.size()[0] // 3, ...] = pred[:x.size()[0] // 3, ...] + pred_offset

        pred_glob_rot, pred_glob_offset_0 = self.rot_reg(j_mid.detach(), r_mid, pred_occ.detach())

        return pred, pred_occ, pred_glob_rot, pred_glob_offset_0

