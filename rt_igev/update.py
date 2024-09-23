import torch
import torch.nn as nn
import torch.nn.functional as F

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, c, x):
        # x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + c[0])
        r = torch.sigmoid(self.convr(hx) + c[1])
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + c[2])
        h = (1-z) * h + z * q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels=2, corr_radius=4):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1) * 8
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 96-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels=2, corr_radius=4, n_downsample=2, hidden_dim=96):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)

        self.gru = ConvGRU(hidden_dim, hidden_dim)
        self.disp_head = DispHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2**n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None):

        motion_features = self.encoder(disp, corr)
        net = self.gru(net, inp, motion_features)

        delta_disp = self.disp_head(net)
        mask_feat_4 = self.mask_feat_4(net)
        return net, mask_feat_4, delta_disp
