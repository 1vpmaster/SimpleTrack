from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
from collections import deque


import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ..decode import mot_decode
from ..utils import to_image_list

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


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
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
        
        self.heads = heads
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

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z], y[-1]


# 特征attention
class AttentionExtractor(nn.Module):
    def __init__(self, in_channels):
        super(AttentionExtractor, self).__init__()
    
    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        '''
        bbox [N,4]
        '''
        bbox = bbox[:,:4]
        ref_bbox = ref_bbox[:,:4]
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0,1) # 广播， [N,1] - [1,M] -> [N,M]
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0,1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0,1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0,1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2) #[N,M,4]

        return position_matrix
    
    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range,),), wave_length, device = device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1,-1,-1,dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [N, M, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [N, M, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2]*embedding.shape[3])

        return embedding
    
    def cal_position_embedding(self, bbox, bbox_ref):
        # [N, M, 4]
        position_matrix = self.extract_position_matrix(bbox, bbox_ref)
        # [N, M, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, N, M]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, N, M]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding
    
    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding, feat_dim=1024, dim=(1024, 1024, 1024),
                                    group=16, index=0):
        """
        :param roi_feat: [N, feat_dim]
        :param ref_feat: [M, feat_dim]
        :param position_embedding: [1, emb_dim, N , M]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)

        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        # position_embedding, [1, emb_dim, N, M]
        # -> position_feat_1, [1, group, N, M]
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [N, group, M, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, N, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, M, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [M, feat_dim]
        v_data = ref_feat

        # aff, [group, N, M]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1,2))
        aff_scale = (1.0/ math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [N, group, M]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [N, group, M]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.shape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [N*group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [N, group*feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group*feat_dim, 1, 1)
        # linear_out, [N, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."
    
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups
    return group_gn

def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1
    num_groups = 32
    eps = 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def make_fc(dim_in, hidden_dim, use_gn = False):
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class MultiFeatureExtractor(AttentionExtractor):
    def __init__(self, in_channels=1024):
        super(MultiFeatureExtractor, self).__init__(in_channels)
        # 测试用参数，测试完添加到opt中
        REDUCE_CHANNEL = True #是否减少通道数
        representation_size = 1024 #
        use_gn = False #是否使用group norm
        ATTENTION_ENABLE = True #是否进行注意力
        self.all_frame_interval = 10 # 10
        self.memory_enable = True #测试使用memory记忆
        if REDUCE_CHANNEL:
            # 减少通道数
            new_conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = 256
        else:
            new_conv = None
            output_channel = in_channels
        
        if ATTENTION_ENABLE:
            self.embed_dim = 64
            self.groups = 16
            self.feat_dim = representation_size

            self.stage = 3 # 注意力循环次数

            self.base_num = 75
            self.advanced_num = 15

            fcs, Wgs, Wqs, Wks, Wvs, us = [], [], [], [], [], []

            for i in range(self.stage):
                if i == 0:
                    r_size = 64 # 每个roi的特征展平
                else:
                    r_size = self.feat_dim
                fcs.append(make_fc(r_size, representation_size, use_gn))
                Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                Wks.append(make_fc(self.feat_dim, self.feat_dim))
                Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups)) # 分组卷积
                us.append(nn.Parameter(torch.Tensor(self.groups, 1, self.embed_dim))) #(16, 1, 64)
                for l in [Wgs[i], Wvs[i]]:
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                for weight in [us[i]]:
                    torch.nn.init.normal_(weight, std=0.01)
                
                self.l_fcs = nn.ModuleList(fcs)
                self.l_Wgs = nn.ModuleList(Wgs)
                self.l_Wqs = nn.ModuleList(Wqs)
                self.l_Wks = nn.ModuleList(Wks)
                self.l_Wvs = nn.ModuleList(Wvs)
                self.l_us = nn.ParameterList(us)
        
        self.out_channels = representation_size
    
    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0, ver="local"):
        if ver in ("local", "memory"):
            Wgs, Wqs, Wks, Wvs, us = self.l_Wgs, self.l_Wqs, self.l_Wks, self.l_Wvs, self.l_us
        
        dim_group = (dim[0]/ group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim. N, M]
        # ->position_feat_1, [1, group, N, nongt_dim]
        if position_embedding is not None:
            # 如果有位置编码的话
            position_feat_1 = F.relu(Wgs[index](position_embedding))
            # aff_weight, [num_rois, group, num_nongt_rois, 1]
            aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, group, num_nongt_rois]
            aff_weight = aff_weight.squeeze(3) #位置权重
        # multi head
        assert dim[0] == dim[1]

        q_data = Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff_a, [group, num_rois, num_nongt_rois]
        aff_a = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))

        # aff_c, [group, 1, num_nongt_rois]
        aff_c = torch.bmm(us[index], k_data_batch.transpose(1, 2)) # 可学习权重

        # aff = aff_a + aff_b + aff_c + aff_d
        aff = aff_a + aff_c # 广播机制

        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        if position_embedding is not None:
            weighted_aff = (aff_weight + 1e-6).log() + aff_scale + 1e-12 # 最终权重，只有邻近帧才有位置权重
        else:
            weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output
    
    def init_memory(self):
        self.mem_queue_list = []
        self.mem = []
        for i in range(self.stage):
            queue = {"rois": deque(maxlen=self.all_frame_interval),
                    "feats": deque(maxlen=self.all_frame_interval)}
            self.mem_queue_list.append(queue)
            self.mem.append(dict())
    
    def update_memory(self, i ,cache):
        number_to_push = self.base_num if i ==0 else self.advanced_num

        rois = cache["rois_ref"][:number_to_push]
        feats = cache["feats_ref"][:number_to_push]

        self.mem_queue_list[i]["rois"].append(rois)
        self.mem_queue_list[i]["feats"].append(feats)

        self.mem[i] = {"rois": torch.cat(list(self.mem_queue_list[i]["rois"]),dim = 0),
                        "feats": torch.cat(list(self.mem_queue_list[i]["feats"]), dim=0)}
    
    def select_feats(self, feats, proposals):
        feats = feats.permute(0,2,3,1).contiguous()
        feats = feats.view(feats.size(0), -1, feats.size(3))
        dim = feats.size(2)
        inds = torch.cat([proposals[i][1] for i in range(len(proposals))],dim=0) 
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)

        return feats
    
    def select_single_feats(self, feats, proposals):
        feats = feats.permute(0,2,3,1).contiguous()
        feats = feats.view(feats.size(0), -1, feats.size(3))
        dim = feats.size(2)
        inds = proposals[1]
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)

        return feats
    
    def generate_feats(self, x, proposals, proposals_key=None, ver="local"):

        if proposals_key is not None:
            assert ver == "local"

            x_key = self.select_single_feats(x[0:1, ...], proposals_key) # (N, task_feat_dim) 
            x_key = x_key.reshape(-1, x_key.size(-1))
        
        if proposals:
            if ver == "memory":
                x = self.select_feats(x, proposals)
                x = x.reshape(-1, x.size(-1))
            else:
                x = self.select_feats(x[1:,...], proposals[1:])
                x = x.reshape(-1, x.size(-1))
        
        if ver == "local":
            x_key = F.relu(self.l_fcs[0](x_key))
        x = F.relu(self.l_fcs[0](x))

        # distillation
        if ver in ("local", "memory"):
            x_dis = torch.cat([x[:self.advanced_num] for x in torch.split(x, self.base_num, dim=0)], dim=0)
            if ver=="memory":
                rois_dis = torch.cat([x[0][:,:self.advanced_num,:] for x in proposals], dim=1).squeeze() #丢弃index,只保留bbox
                rois_ref = torch.cat([x[0] for x in proposals], dim=1).squeeze()
            if ver == "local":
                rois_dis = torch.cat([x[0][:,:self.advanced_num,:] for x in proposals[1:]], dim=1).squeeze() #丢弃index,只保留bbox
                rois_ref = torch.cat([x[0] for x in proposals[1:]], dim=1).squeeze()
        
        if ver == "memory":
            self.memory_cache.append({"rois_cur": rois_dis,
                                      "rois_ref": rois_ref,
                                      "feats_cur": x_dis,
                                      "feats_ref": x})
            for _ in range(self.stage - 1):
                self.memory_cache.append({"rois_cur": rois_dis,
                                        "rois_ref": rois_dis})
        elif ver == "local":
            self.local_cache.append({"rois_cur": torch.cat([proposals_key[0].squeeze(), rois_dis], dim=0),
                                    "rois_ref": rois_ref,
                                    "feats_cur": torch.cat([x_key, x_dis], dim=0),
                                    "feats_ref": x})
            for _ in range(self.stage - 2):
                self.local_cache.append({"rois_cur": torch.cat([proposals_key[0].squeeze(), rois_dis], dim=0),
                                        "rois_ref": rois_dis})
            self.local_cache.append({"rois_cur": proposals_key[0].squeeze(),
                                    "rois_ref": rois_dis})

    def generate_feats_test(self, x, proposals):
        proposals, proposals_ref, proposals_ref_dis, x_ref, x_ref_dis = proposals

        rois_key = proposals[0].squeeze()
        
        rois = torch.cat([x[0] for x in proposals_ref], dim=0).squeeze()
        rois_dis = torch.cat([x[0] for x in proposals_ref_dis], dim=0).squeeze()

        self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                "rois_ref": rois,
                                "feats_cur": torch.cat([x, x_ref_dis], dim=0),
                                "feats_ref": x_ref})
        for _ in range(self.stage-2):
            self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                    "rois_ref": rois_dis})
        self.local_cache.append({"rois_cur": rois_key,
                                "rois_ref": rois_dis})
        

        

    def _forward_train_single(self, i, cache, memory=None, ver="memory"):

        rois_cur = cache.pop("rois_cur") #[45,6]
        rois_ref = cache.pop("rois_ref") #[225,6]
        feats_cur = cache.pop("feats_cur") #[45, 1024]
        feats_ref = cache.pop("feats_ref") #[225, 1024]

        if memory is not None:
            rois_ref = torch.cat([rois_ref, memory["rois"]], dim=0)
            feats_ref = torch.cat([feats_ref, memory["feats"]], dim=0)
        
        if ver == "memory":
            self.mem.append({"rois": rois_ref, "feats": feats_ref})
            if i == self.stage - 1:
                return
        
        if rois_cur is not None:
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        else:
            position_embedding = None
        
        attention = self.attention_module_multi_head(feats_cur, feats_ref, position_embedding,
                                                    feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                    index=i, ver=ver) #[45, 1024]
        
        feats_cur = feats_cur + attention

        if i != self.stage - 1:
            feats_cur = F.relu(self.l_fcs[i+1](feats_cur))
        
        return feats_cur

    def _forward_test_single(self, i, cache, memory):
        rois_cur = cache.pop("rois_cur")
        rois_ref = cache.pop("rois_ref")
        feats_cur = cache.pop("feats_cur")
        feats_ref = cache.pop("feats_ref")

        if memory is not None:
            rois_ref = torch.cat([rois_ref, memory["rois"]], dim=0)
            feats_ref = torch.cat([feats_ref, memory["feats"]], dim=0)
        
        # if rois_cur is not None:
        #     position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        # else:
        #     position_embedding = None
        # 无位置嵌入
        position_embedding = None

        attention = self.attention_module_multi_head(feats_cur, feats_ref, position_embedding,
                                                    feat_dim=1024, group=16, dim=(1024, 1024, 1024), index=i)
        
        feats_cur = feats_cur + attention

        if i != self.stage - 1:
            feats_cur = F.relu(self.l_fcs[i + 1](feats_cur))
        return feats_cur

    def _forward_train(self, x, proposals):
        proposals_l, proposals_m = proposals
        x_l, x_m = x # x_l 的第一个元素为当前帧的特征，x_l[0]=feat_cur

        self.memory_cache = []
        self.local_cache = []

        if proposals_m:
            with torch.no_grad():
                self.generate_feats(x_m, proposals_m, ver="memory")
        
        self.generate_feats(x_l, proposals_l, proposals_l[0], ver="local")

        # 1. 生成长程记忆
        with torch.no_grad():
            if self.memory_cache:
                self.mem = []
                for i in range(self.stage):
                    feats = self._forward_train_single(i, self.memory_cache[i], None, ver="memory")

                    if i == self.stage - 1:
                        break

                    self.memory_cache[i+1]["feats_cur"] = feats
                    self.memory_cache[i+1]["feats_ref"] = feats
            else:
                self.mem = None
        
        # 2. 更新当前特征
        for i in range(self.stage):
            if self.mem is not None:
                memory = self.mem[i]
            else:
                memory = None
            
            feats = self._forward_train_single(i, self.local_cache[i], memory, ver="local")

            if i == self.stage - 1:
                x = feats
            elif i == self.stage - 2:
                self.local_cache[i+1]["feats_cur"] = feats[:proposals[0][0][0].size(1)] # 应该需要修改
                self.local_cache[i+1]["feats_ref"] = feats[proposals[0][0][0].size(1):]
            else:
                self.local_cache[i+1]["feats_cur"] = feats
                self.local_cache[i+1]["feats_ref"] = feats[proposals[0][0][0].size(1):]

        return x
    
    def _forward_ref(self, x, proposals):
        x = self.select_single_feats(x, proposals)
        x = x.reshape(-1, x.size(-1))

        x = F.relu(self.l_fcs[0](x))

        return x

    def _forward_test(self, x, proposals):
        # proposals, proposals_ref, x_refs = proposals
        x = self.select_single_feats(x, proposals[0])
        x = x.reshape(-1, x.size(-1))

        x = F.relu(self.l_fcs[0](x))

        self.local_cache = []

        self.generate_feats_test(x, proposals)

        for i in range(self.stage):
            memory = self.mem[i] if self.mem[i] else None

            if self.memory_enable:
                self.update_memory(i, self.local_cache[i])
            
            feat_cur = self._forward_test_single(i, self.local_cache[i], memory)

            if i == self.stage - 1:
                x = feat_cur
            elif i == self.stage - 2:
                self.local_cache[i+1]["feats_cur"] = feat_cur[:proposals[0][0].size(1)] # 应该需要修改
                self.local_cache[i+1]["feats_ref"] = feat_cur[proposals[0][0].size(1):]
            else:
                self.local_cache[i+1]["feats_cur"] = feat_cur
                self.local_cache[i+1]["feats_ref"] = feat_cur[proposals[0][0].size(1):]
        
        return x

        

    def forward(self, x, proposals, pre_calculate=False):
        if pre_calculate:
            return self._forward_ref(x, proposals)

        if self.training:
            return self._forward_train(x, proposals)
        else:
            return self._forward_test(x, proposals)



# 视频帧特征聚合
class MultiDeCoupleDLA_CrossLocal(nn.Module):
    def __init__(self, istraining, num_layers, heads, head_conv=256, down_ratio=4):
        super(MultiDeCoupleDLA_CrossLocal, self).__init__()
        # 提取单帧的特征，得到包含检测三个特征图以及reid的一个特征图,返回一个字典
        # hm,wh,reg,id
        self.FeatureExtraNet = DLASeg(num_layers, heads,
        pretrained=True, down_ratio=down_ratio, final_kernel=1, last_level=5, head_conv=head_conv) 

        # 测试完毕后，需要集成到opt中
        self.base_num = 75
        self.advanced_ratio = 0.2
        final_kernel = 3 # 卷积核
        self.advanced_num = int(self.advanced_ratio * self.base_num)
        self.training = istraining
        self.aggerate_func = MultiFeatureExtractor(in_channels=1024)
        self.D_linear = make_fc(1024, 64, use_gn=False)
        self.all_frame_interval = 10
        

        # aggerate_head
        self.heads = heads
    

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
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
              fc = nn.Conv2d(64, classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
    
    def forward(self, batch):
        
        if self.training:
            # batch["input"] = to_image_list(batch["input"])
            # batch["ref_l"] = [to_image_list(image) for image in batch["ref_l"]]
            # batch["ref_m"] = [to_image_list(image) for image in batch["ref_m"]]

            return self._forward_train(batch["input"], batch["ref_l"], batch["ref_m"])
        else:
            # batch["input"] = to_image_list(images["input"])
            # batch["ref_l"] = [to_image_list(image) for image in batch["ref_l"]]
            # batch["ref_m"] = [to_image_list(image) for image in batch["ref_m"]]

            infos = batch.copy()
            infos.pop("input")
            return self._forward_test(batch["input"], infos)

    def _forward_train(self, img_cur, imgs_l, imgs_m):
        # img_cur [1,3,608,1280]
        # imgs_l [1,3,3,608,1280]
        # 1. 记忆帧
        proposals_m_list = []
        if imgs_m is not None:
            concat_imgs_m = imgs_m.squeeze()
            concat_feats_m, D_m= self.FeatureExtraNet(concat_imgs_m)
            feats_m_list = [{key: torch.chunk(value, value.shape[0],dim=0)[i] for key, value in concat_feats_m[0].items()} for i in range(concat_imgs_m.shape[0])]

            for i in range(concat_imgs_m.shape[0]):
                proposals_ref = self.bbox_select(feats_m_list[i], self.training, version='ref')
                proposals_m_list.append(proposals_ref)
        else:
            feats_m_list = []
        
        # 2. 邻近帧
        concat_imgs_l = imgs_l.squeeze() #[3,3,608,1088]
        concat_imgs_l = torch.cat((img_cur,concat_imgs_l), dim=0) #包括当前帧了
        concat_feats_l, D_l= self.FeatureExtraNet(concat_imgs_l) # D_l[4, 64, 152, 272] R_l [4, 64, 152, 272]
        # {'hm'(B), 'wh'(B), 'reg'(B), 'id'(B)} --> (B){'hm','wh','reg','id'}
        feats_l_list = [{key: torch.chunk(value, value.shape[0],dim=0)[i] for key, value in concat_feats_l[0].items()} for i in range(concat_imgs_l.shape[0])] 

        proposals_l_list = [] # 包含当前帧？
        proposals_cur = self.bbox_select(feats_l_list[0], self.training, version='key')
        proposals_l_list.append(proposals_cur)
        # proposals 中的每个proposal包含[0](x1,y1,x2,y2,score,cls) [1](index-->特征图中的位置)
        for i in range(imgs_l.shape[1]):
            proposals_ref = self.bbox_select(feats_l_list[i+1], self.training, version='ref')
            proposals_l_list.append(proposals_ref)
        
        feats_list_D = [D_l, D_m]

        proposals_list = [proposals_l_list, proposals_m_list]

        aggerate_D = self.feature_aggerate(feats_list_D, proposals_list) #[300, 1024]
    

        aggerate_D = self.D_linear(aggerate_D) #[300, 64] 线性层压缩至64，后与原始特征进行相加，这里采用的是scatter替换
   
        
        aggerate_D = self.fuse_point(aggerate_D, D_l[0,:,:,:], proposals_cur[1]) #[1, 64, 152, 272] 得到融合后的特征图
  

        aggerate_z = {}
        for head in self.heads:
            aggerate_z[head] = self.__getattr__(head)(aggerate_D)

        return [aggerate_z, feats_l_list[0]] # 返回值是聚合特征后的head和当前自己的head

    def _forward_test(self, imgs, infos):
        """
        forward for the test phase
        imgs: cur
        infos: 除去cur的batch
        """
        def update_feature(img=None, feats=None, proposals=None, proposals_feat_D=None, img_path=None):
            assert (img is not None) or (feats is not None and proposals is not None and proposals_feat_D is not None)

            if img is not None:
                feats = self.FeatureExtraNet(img) # feats (head, D_f, R_f)
                proposals = self.bbox_select(feats[0][0], self.training, version='ref')
                proposals_feat_D = self.feature_aggerate(feats[1], proposals, pre_calculate=True)
        
            
            self.feats.append(feats)
            self.proposals.append(proposals[0]) # (det, ind)
            self.proposals_dis.append(proposals[0][:,:self.advanced_num,:])
            self.proposals_feat.append([proposals_feat_D])
            self.proposals_feat_dis.append([proposals_feat_D[:self.advanced_num]])

            self.img_path.append(img_path)

        feats_cur = self.FeatureExtraNet(imgs)
        # 如果是第一帧，infos不知道有没有帧的编号, 用第一帧填充
        if infos["frame_category"] == 0:
            self.seg_len = infos["seg_len"]
            self.end_id = 0

            # 突然发现local帧要和memory相等， all_frame_interval为10
            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)

            # for debug
            self.img_path = deque(maxlen=self.all_frame_interval)

            self.aggerate_func.init_memory()

            
            proposals_cur = self.bbox_select(feats_cur[0][0], self.training, version="ref")
            proposals_feat_cur_D = self.feature_aggerate(feats_cur[1], proposals_cur, pre_calculate=True)
      

            while len(self.feats) < self.all_frame_interval:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur_D, infos["img_ref_path"])
            
        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, self.seg_len - 1)
            end_image = infos["ref_l"][0]

            update_feature(end_image, img_path=infos["img_ref_path"])
        
        proposals = self.bbox_select(feats_cur[0][0], self.training, version="key")
        
        vis_plot = False # debug用，查看roi分布
        if vis_plot == True:
            inds = proposals[1].cpu()[:,:50]
            h,w = feats_cur[1].size(2), feats_cur[1].size(3)
            points = torch.ones(inds.size(0), inds.size(1))
            feat_map = torch.zeros(feats_cur[1].size(2), feats_cur[1].size(3))
            # inds = inds.expand(feats_map.size(0),inds.size(-1))
            feat_map = feat_map.reshape(1,-1)
            feat_map = torch.scatter(feat_map,1,inds,points)
            feat_map = feat_map.reshape(feat_map.size(0),h,w).squeeze()
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(feat_map)
            plt.axis('off')
            plt.savefig('test.jpg')

        proposals_ref = list(self.proposals)
        proposals_ref_dis = list(self.proposals_dis)
        proposals_feat_ref_D = torch.cat([x[0] for x in list(self.proposals_feat)], dim=0)
        proposals_feat_ref_dis_D = torch.cat([x[0] for x in list(self.proposals_feat_dis)], dim=0)

        proposals_list_D = [proposals, proposals_ref, proposals_ref_dis, proposals_feat_ref_D, proposals_feat_ref_dis_D]

    

        # 应该一个D，一个R
        aggerate_D = self.feature_aggerate(feats_cur[1], proposals_list_D)
    
        aggerate_D = self.D_linear(aggerate_D) #[300, 64] 线性层压缩至64，后与原始特征进行相加，这里采用的是scatter替换

        
        aggerate_D = self.fuse_point(aggerate_D, feats_cur[1][0,:,:,:], proposals[1]) #[1, 64, 152, 272] 得到融合后的特征图


        # D_dis = aggerate_D - feats_cur[1]
        # R_dis = aggerate_R - feats_cur[2]

        aggerate_z = {}
        for head in self.heads:
            aggerate_z[head] = self.__getattr__(head)(aggerate_D)

        return aggerate_z


    def fuse_point(self, feats_point, feats_map, inds):
        h,w = feats_map.size(1), feats_map.size(2)
        feats_point = feats_point.permute(1,0)
        inds = inds.expand(feats_map.size(0),inds.size(-1))
        feats_map = feats_map.reshape(feats_map.size(0),-1)
        feats_map = torch.scatter(feats_map,1,inds,feats_point)
        feats_map = feats_map.reshape(feats_map.size(0),h,w).unsqueeze(0)
        return feats_map

    def bbox_select(self, features, is_train, version='ref'):
        # 提取检测特征和重识别特征
        # 测试用，测试完需要合并到opt中
        # 未实现去除小面积边界框的topk
        ltrb = True
        reg_offset = True
        '''
        features 是一张图像得到的四个head的字典
        return 返回值返回包括边界框的尺寸，为了编码特征之间的位置矩阵
        '''
        if version == "ref":
            post_top_n = 75
        else:
            post_top_n = 300
            if not is_train:
                post_top_n = 300 # orign 300

        with torch.no_grad():
            hm = features['hm'].sigmoid()
            wh = features['wh']
            reg = features['reg'] if reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=ltrb, K=post_top_n)
        
        return [dets, inds]


    def feature_aggerate(self, feats_list, proposals_list, pre_calculate=False):
        # 聚合特征，生成新的头
        return self.aggerate_func(feats_list, proposals_list, pre_calculate=pre_calculate)


        

def get_pose_net(istraining, num_layers, heads, head_conv=256, down_ratio=4):

    model = MultiDeCoupleDLA_CrossLocal(istraining, 'dla{}'.format(num_layers), heads,
                down_ratio=down_ratio,
                head_conv=head_conv)
    return model