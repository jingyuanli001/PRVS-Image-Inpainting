from modules.partialconv2d import PartialConv2d
import torch.nn as nn
import torch.nn.functional as F
PartialConv = PartialConv2d

class PConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False, deconv = False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias, multi_channel = True)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias, multi_channel = True)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias, multi_channel = True)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias, multi_channel = True)
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, bias = conv_bias)
        else:
            self.deconv = None
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if self.deconv is not None:
            h = self.deconv(h)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        h_mask = F.interpolate(h_mask, size = h.size()[2:])
        return h, h_mask