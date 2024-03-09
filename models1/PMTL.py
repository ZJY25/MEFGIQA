import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_3x3(in_plane, out_plane, stride, padding):
    return nn.Sequential(nn.Conv2d(in_plane, out_plane, (3, 3), stride, padding),
                         nn.BatchNorm2d(out_plane),
                         nn.ReLU(inplace=True))


def conv_1x1(in_plane, out_plane, stride, padding):
    return nn.Sequential(nn.Conv2d(in_plane, out_plane, (1, 1), stride, padding),
                         nn.BatchNorm2d(out_plane),
                         nn.ReLU(inplace=True))


class AFF(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(AFF, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x, y):
        out = x + y
        avg_out = self.avg_pool(out)
        max_out = self.max_pool(out)
        out_atten = self.sigmoid(self.conv_1(avg_out) + self.conv_2(max_out))
        out = out * out_atten + out
        return out


class fushion_block(nn.Module):
    def __init__(self, in_plane):
        super(fushion_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_plane, in_plane * 2, kernel_size=(1, 1), stride=(1, 1)),
                                  nn.BatchNorm2d(in_plane * 2))
                                  # nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.conv(x)
        x = x + y
        x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PMTL(nn.Module):
    def __init__(self, c1_dim, c2_dim, c3_dim, c4_dim):
        super(PMTL, self).__init__()
        self.x1_conv = conv_3x3(c1_dim, c1_dim, stride=(2, 2), padding=(1, 1))
        self.x21_conv = conv_3x3(c2_dim, c2_dim, stride=(1, 1), padding=(1, 1))
        self.x1x2_fushion = fushion_block(in_plane=c1_dim)

        self.x22_conv = conv_3x3(c2_dim, c2_dim, stride=(2, 2), padding=(1, 1))
        self.x31_conv = conv_3x3(c3_dim, c3_dim, stride=(1, 1), padding=(1, 1))
        self.x1x2x3_fushion = fushion_block(in_plane=c2_dim)

        self.x32_conv = conv_3x3(c3_dim, c3_dim, stride=(2, 2), padding=(1, 1))
        self.x41_conv = conv_3x3(c4_dim, c4_dim, stride=(1, 1), padding=(1, 1))
        self.x1x2x3x4_fushion = fushion_block(in_plane=c3_dim)

        self.x42_conv = conv_3x3(c4_dim, c4_dim, stride=(2, 2), padding=(1, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ca = CALayer(channel=960)

    def global_std_pool2d(self, x):
        """2D global standard variation pooling"""
        return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                         dim=2, keepdim=True)

    def forward(self, x1, x2, x3, x4):
        """
        :param x1: [B, 64, 56, 56]
        :param x2: [B, 128, 28, 28]
        :param x3: [B, 256, 14, 14]
        :param x4: [B, 512, 7, 7]
        :return:
        """
        f1 = self.x1_conv(x1)
        # f1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=False)
        f2 = self.x21_conv(x2)
        f12 = self.x1x2_fushion(f1, f2)
        f12 = self.x22_conv(f12)
        # f12 = F.interpolate(f12, x3.size()[2:], mode='bilinear', align_corners=False)
        f3 = self.x31_conv(x3)
        f123 = self.x1x2x3_fushion(f12, f3)
        f123 = self.x32_conv(f123)
        # f123 = F.interpolate(f123, x4.size()[2:], mode='bilinear', align_corners=False)
        f4 = self.x41_conv(x4)
        f1234 = self.x1x2x3x4_fushion(f123, f4)
        f1234 = self.x42_conv(f1234)
        # f1234 = F.interpolate(f1234, scale_factor=0.5, mode='bilinear', align_corners=False)

        feat1 = self.avg(f1)
        feat2 = self.avg(f12)
        feat3 = self.avg(f123)
        feat4 = self.avg(f1234)
        feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        feat = self.ca(feat)
        return feat

# if __name__ == "__main__":
#     x1 = torch.randn(1, 64, 56, 56)
#     x2 = torch.randn(1, 128, 28, 28)
#     x3 = torch.randn(1, 256, 14, 14)
#     x4 = torch.randn(1, 512, 7, 7)
#     net = PMTL(64, 128, 256, 512)
#     print(net(x1, x2, x3, x4).shape)