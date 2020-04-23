import math
from modules.partialconv2d import PartialConv2d
from modules.PConvLayer import PConvLayer
from modules.VSRLayer import VSRLayer
import modules.Attention as Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
PartialConv = PartialConv2d

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out

class PRVSNet(BaseNetwork):
    def __init__(self, layer_size=8, input_channels=3, att = False):
        super().__init__()
        self.layer_size = layer_size
        self.enc_1 = VSRLayer(3, 64, kernel_size = 7)
        self.enc_2 = VSRLayer(64, 128, kernel_size = 5)
        self.enc_3 = PConvLayer(128, 256, sample='down-5')
        self.enc_4 = PConvLayer(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PConvLayer(512, 512, sample='down-3'))
        self.deconv = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PConvLayer(512 + 512, 512, activ='leaky', deconv = True))
        self.dec_4 = PConvLayer(512 + 256, 256, activ='leaky', deconv = True)
        if att:
            self.att = Attention.AttentionModule()
        else:
            self.att = lambda x:x
        self.dec_3 = PConvLayer(256 + 128, 128, activ='leaky', deconv = True)
        self.dec_2 = VSRLayer(128 + 64, 64, stride = 1, activation='leaky', deconv = True)
        self.dec_1 = VSRLayer(64 + input_channels, 64, stride = 1, activation = None, batch_norm = False)
        self.resolver = Bottleneck(64,16)
        self.output = nn.Conv2d(128, 3, 1)
        
    def forward(self, input, input_mask, input_edge):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_edge_list = []
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        edge = input_edge

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if i not in [1, 2]:
                h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            else:
                h_dict[h_key], h_mask_dict[h_key], edge = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev], edge)
                h_edge_list.append(edge)
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        h = self.deconv(h)
        h_mask = F.interpolate(h_mask, scale_factor = 2)
        
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim = 1)
            h = torch.cat([h, h_dict[enc_h_key]], dim = 1)
            if i not in [2, 1]:
                h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            else:
                edge = h_edge_list[i-1]
                h, h_mask, edge = getattr(self, dec_l_key)(h, h_mask, edge)
                h_edge_list.append(edge)
            if i == 4:
                h = self.att(h)
        h_out = self.resolver(h)
        h_out = torch.cat([h_out, h], dim = 1)
        h_out = self.output(h_out)
        return h_out, h_mask, h_edge_list[-2], h_edge_list[-1]

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()
