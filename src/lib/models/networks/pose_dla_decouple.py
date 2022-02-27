from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
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


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
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


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


# Reid分支的USA层
class SAUp(nn.Module):
    def __init__(self, in_channel, out_channel, up_radio):
        super(SAUp, self).__init__()
        self.proj = DeformConv(in_channel, out_channel)
        self.up = nn.ConvTranspose2d(out_channel, out_channel, up_radio*2, stride=up_radio, padding=up_radio//2, output_padding=0,
                                groups=out_channel, bias=False)
        self.Conv = nn.Conv2d(out_channel, 1,
                          kernel_size=1, stride=1, bias=False)
        self.att = nn.Sigmoid()

        fill_up_weights(self.up)

    def forward(self,x):
        x_up = self.Conv(self.up(self.proj(x)))
        return self.att(x_up)



# CrossFeature non-local
class CrossNonLocal(nn.Module):
    def __init__(self, channel):
        super(CrossNonLocal,self).__init__()
        self.inter_channel = channel //2
        # q, k 对应高层次的特征图的embedding
        self.conv_q = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_k = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False)
        # v对应较低层的
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(self.inter_channel)
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        


    def forward(self, low_layer, high_layer):
        # high_layer [N, C, H, W]
        # low_layer [N, C/2, 2H, 2W]
        batch_size = high_layer.size(0)
        x_q = self.conv_q(high_layer).view(batch_size, self.inter_channel, -1) #[N,C/2,HW]
        x_q = x_q.permute(0,2,1) #[N, HW ,C/2]
        
        x_k = self.conv_k(high_layer).view(batch_size, self.inter_channel, -1) #[N,C/2,HW]
        f = torch.matmul(x_q,x_k)
        f_div_C = F.softmax(f, dim=-1)

        x_v = self.conv_v(low_layer).view(batch_size, self.inter_channel, -1) 
        x_v = x_v.permute(0,2,1) #[N,HW, C/2]

        y = torch.matmul(f_div_C, x_v)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channel, *high_layer.size()[2:])
        W_y = self.W(y) #[N,C/2,H,W]
        W_y = self.up(W_y) #[N,C/2,2H,2W]
        z = W_y + low_layer
        return z   


# 单特征层局部注意力
class SelfLocalATT(nn.Module):
    def __init__(self, channel, R=3, H=0,W=0):
        super(SelfLocalATT,self).__init__()
        self.inter_channel = channel //2
        self.R = R # 局部注意范围
        self.H = H #特征图的高
        self.W = W #特征图的宽
        # q, k 对应高层次的特征图的embedding
        self.conv_q = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_k = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False)
        # v
        self.conv_v = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1,stride=1,padding=0,bias=False)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(R*R*H*W)
        # ffn
        self.linear1 = nn.Linear(R*R*H*W, channel)
    
        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(channel, self.inter_channel*H*W)
        self.dropout3 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(self.inter_channel*H*W)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(channel)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        self.unfold = nn.Unfold(kernel_size=(self.R,self.R),stride=1,padding=(self.R-1)//2) 

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, x):
        # x [N, C, H, W]
        N, C, H, W = x.size()
        x_q = self.conv_q(x).view(N, self.inter_channel, -1) #[N,C/2,HW]
        x_q = x_q.permute(0,2,1) #[N, HW ,C/2]
        x_q = x_q.unsqueeze(2) #[N,HW,1,C/2]
        
        x_k = self.conv_k(x) #[N,C/2,H,W]
        x_k_unfold = self.unfold(x_k) #[N,C/2*R*R,HW]
        x_k_unfold =x_k_unfold.permute(0,2,1) #[N,HW,C/2*R*R]
        x_k_unfold = x_k_unfold.view(N,H*W,self.inter_channel,-1) #[N,HW,C/2,R*R]

        f = torch.matmul(x_q,x_k_unfold) #[N,HW,1,R*R]
        f = f.squeeze()
        f = f.permute(0,2,1).contiguous() #[N,R*R,HW]
        f = f.view(N,-1) #[N,R*R*H*W]
        f = f + self.dropout1(f)
        f = self.norm1(f)
        f = self.forward_ffn(f)
        f = f.view(N,-1,H,W)
        
        W_y = self.W(f) #[N,C,H,W]
        
        z = W_y + x
        return z      

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out



class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DeCoupleDLA(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DeCoupleDLA, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        # Reid 分支
        self.SA_3 = SAUp(channels[-1],channels[-2],2)
        self.SA_2 = SAUp(channels[-2],channels[-3],2)
        self.SA_1 = SAUp(channels[-3],channels[-4],2)
    

        self.heads = heads
        self.det_heads = dict([(key, heads[key]) for key in ['hm','wh','reg']])
        self.reid_heads = dict([(key, heads[key]) for key in ['id']])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        # x[0] (1,64,152,272)
        # x[1] (1,128,76,136)
        # x[2] (1,256,38,68)
        # x[3] (1,512,19,34)


        # det分支
        D = []
        for i in range(self.last_level - self.first_level):
            D.append(x[i].clone())
        self.ida_up(D, 0, len(D))

        # reid分支
        att_3 = self.SA_3(x[3])
        att_2 = self.SA_2(x[2]*att_3)
        att_1 = self.SA_1(x[1]*att_2)
        R = x[0]*att_1
        


        z = {}
        for head in self.det_heads:
            z[head] = self.__getattr__(head)(D[-1])
        
        for head in self.reid_heads:
            z[head] = self.__getattr__(head)(R)

        return [z]
    


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):

  model = DeCoupleDLA('dla{}'.format(num_layers), heads,
                pretrained=True,
                down_ratio=down_ratio,
                final_kernel=1,
                last_level=5,
                head_conv=head_conv)
  
  return model

