import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RSU(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU, self).__init__()
        self.conv_in = REBNCONV(in_ch, out_ch)
        self.conv_mid = REBNCONV(out_ch, mid_ch)
        self.conv_out = REBNCONV(mid_ch, out_ch)

    def forward(self, x):
        hx = self.conv_in(x)        
        x = self.conv_mid(hx)       
        x = self.conv_out(x)        
        return x + hx               

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU(in_ch, 16, 64)
        self.stage2 = RSU(64, 16, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.final(x)
        return torch.sigmoid(x)
