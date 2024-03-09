import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from models2.PMTL import PMTL
from models2.Res2Net import res2net50_v1b_26w_4s


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ECT(nn.Module):
    def __init__(self, channel=1):
        super(ECT, self).__init__()
        in_channel = channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=1,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = out + identity
        out = torch.sigmoid(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        self.sa = SpatialAttention(kernel_size=3)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f, edge):
        if f.size() != edge.size():
            edge = F.interpolate(edge, f.size()[2:], mode='bilinear', align_corners=False)
        x = f * edge + f
        x = self.conv2d(x)
        s_weight = self.sa(x)
        x = x * s_weight
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.edge_extract = ECT()
        self.efm1 = EFM(256)
        self.efm2 = EFM(512)
        self.efm3 = EFM(1024)
        self.efm4 = EFM(2048)

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 128)
        self.reduce3 = Conv1x1(1024, 256)
        self.reduce4 = Conv1x1(2048, 512)

        self.pam1 = PAM_Module(in_dim=64)
        self.pam2 = PAM_Module(in_dim=128)
        self.pam3 = PAM_Module(in_dim=256)
        self.pam4 = PAM_Module(in_dim=512)

        self.pmtl = PMTL(c1_dim=64, c2_dim=128, c3_dim=256, c4_dim=512)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_score = nn.Sequential(
            nn.Linear(960, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def freezing(self):
        for name, param in self.named_parameters():
            if name.startswith('resnet'):
                param.requires_grad = False

    def unfreezing(self):
        for name, param in self.named_parameters():
            if name.startswith('resnet'):
                param.requires_grad = True

    def forward(self, x, y):
        x1, x2, x3, x4 = self.resnet(x)
        edge_att = self.edge_extract(y)

        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        feat1 = self.pam1(x1r)
        feat2 = self.pam2(x2r)
        feat3 = self.pam3(x3r)
        feat4 = self.pam4(x4r)

        feat = self.pmtl(feat1, feat2, feat3, feat4)
        feat = torch.flatten(feat, start_dim=1, end_dim=3)
        score = self.fc_score(feat)
        score = score.squeeze(dim=-1)

        return score