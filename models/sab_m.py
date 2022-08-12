import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.utils import *
use_gpu = True

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 1)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.conv_qk1_a1 = nn.Conv2d(24, 8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_a1 = nn.BatchNorm2d(8)
        self.conv_qk1_a2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_a2 = nn.BatchNorm2d(8)
        self.bn_max1 = nn.BatchNorm2d(8)
        self.conv_qk1_b1 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_b1 = nn.BatchNorm2d(16)
        self.conv_qk1_b2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_b2 = nn.BatchNorm2d(16)
        self.bn_max2 = nn.BatchNorm2d(16)
        self.conv_qk1_c1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_c1 = nn.BatchNorm2d(32)
        self.conv_qk1_c2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_c2 = nn.BatchNorm2d(32)
        self.bn_max3 = nn.BatchNorm2d(32)
        self.conv_qk1_d1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_d1 = nn.BatchNorm2d(64)
        self.conv_qk1_d2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk1_d2 = nn.BatchNorm2d(64)

        self.up_conv_qk1c = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv_qk1c = nn.BatchNorm2d(16)
        self.decoder1_1_qk1c = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_1_qk1c = nn.BatchNorm2d(8)
        self.decoder1_2_qk1c = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_2_qk1c = nn.BatchNorm2d(8)

        self.conv_qk2_a1 = nn.Conv2d(24, 8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk2_a1 = nn.BatchNorm2d(8)
        self.conv_qk2_a2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk2_a2 = nn.BatchNorm2d(8)
        self.bn_max1 = nn.BatchNorm2d(8)
        self.conv_qk2_b1 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk2_b1 = nn.BatchNorm2d(16)
        self.conv_qk2_b2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk2_b2 = nn.BatchNorm2d(16)
        self.up_conv_qk2c = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv_qk2c = nn.BatchNorm2d(16)
        self.decoder1_1_qk2c = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_1_qk2c = nn.BatchNorm2d(8)
        self.decoder1_2_qk2c = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_2_qk2c = nn.BatchNorm2d(8)

        self.conv_qk12 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn_qk12 = nn.BatchNorm2d(8)


        self.relu = nn.ReLU(inplace=True)
        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        # print('x',x.shape)
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # print('x',x.shape)
        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        # print('qkv',qkv.shape)
        # reshape=qkv.reshape(N * W, self.groups, self.group_planes * 2, H)
        # print('reshape',reshape.shape)
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # print('q',q.shape)
        # print('k',k.shape)
        # print('v',v.shape)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        # print('all_embeddings',all_embeddings.shape)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # print('qk',qk.shape)
        # print('q_embedding',q_embedding.shape)
        # print('k_embedding',k_embedding.shape)
        # print('v_embedding',v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        # print('qr',qr.shape)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        # print('kr',kr.shape)

        qk = torch.cat([qk, qr, kr], dim=1)
        # print('qk_cat',qk.shape)
        qk_1,qk_2 = torch.split(qk,[N * W// 2, N * W// 2], dim=0)
        # print('qk_1',qk_1.shape)
        # print('qk_2',qk_2.shape)
        # kr_1,kr_2 = torch.split(kr,[N * W// 2, N * W// 2], dim=0)
        ##############################
        # multiply by factors

        conv_qk1_a1 = self.conv_qk1_a1(qk_1)
        conv_qk1_a1 = self.bn_qk1_a1(conv_qk1_a1)
        conv_qk1_a1 = self.relu(conv_qk1_a1)
        conv_qk1_a2 = self.conv_qk1_a2(conv_qk1_a1)
        conv_qk1_a2 = self.bn_qk1_a2(conv_qk1_a2)
        conv_qk1_a2 = self.relu(conv_qk1_a2)
        max_pool_qk1 = F.max_pool2d(conv_qk1_a2, 2, 2)
        max_pool_qk1 = self.bn_max1(max_pool_qk1)

        conv_qk1_b1 = self.conv_qk1_b1(max_pool_qk1)
        conv_qk1_b1 = self.bn_qk1_b1(conv_qk1_b1)
        conv_qk1_b1 = self.relu(conv_qk1_b1)
        conv_qk1_b2 = self.conv_qk1_b2(conv_qk1_b1)
        conv_qk1_b2 = self.bn_qk1_b2(conv_qk1_b2)
        conv_qk1_b2 = self.relu(conv_qk1_b2)

        up_conv_qk1c = self.up_conv_qk1c(conv_qk1_b2)
        up_conv_qk1c = self.bn_up_conv_qk1c(up_conv_qk1c)
        up_conv_qk1c = self.relu(up_conv_qk1c)
        decoder1_1_qk1c = self.decoder1_1_qk1c(up_conv_qk1c)
        decoder1_1_qk1c = self.bn_decoder1_1_qk1c(decoder1_1_qk1c)
        decoder1_1_qk1c = self.relu(decoder1_1_qk1c)
        decoder1_2_qk1c = torch.cat([conv_qk1_a2, decoder1_1_qk1c], dim=1)
        decoder1_2_qk1c = self.decoder1_2_qk1c(decoder1_2_qk1c)
        decoder1_2_qk1c = self.bn_decoder1_2_qk1c(decoder1_2_qk1c)
        decoder1_2_qk1c = self.relu(decoder1_2_qk1c)
############################################################################
############################################################################
        conv_qk2_a1 = self.conv_qk2_a1(qk_2)
        conv_qk2_a1 = self.bn_qk2_a1(conv_qk2_a1)
        conv_qk2_a1 = self.relu(conv_qk2_a1)
        conv_qk2_a2 = self.conv_qk2_a2(conv_qk2_a1)
        conv_qk2_a2 = self.bn_qk2_a2(conv_qk2_a2)
        conv_qk2_a2 = self.relu(conv_qk2_a2)
        max_pool_qk2 = F.max_pool2d(conv_qk2_a2, 2, 2)
        max_pool_qk2 = self.bn_max1(max_pool_qk2)

        conv_qk2_b1 = self.conv_qk2_b1(max_pool_qk2)
        conv_qk2_b1 = self.bn_qk2_b1(conv_qk2_b1)
        conv_qk2_b1 = self.relu(conv_qk2_b1)
        conv_qk2_b2 = self.conv_qk2_b2(conv_qk2_b1)
        conv_qk2_b2 = self.bn_qk2_b2(conv_qk2_b2)
        conv_qk2_b2 = self.relu(conv_qk2_b2)

        up_conv_qk2c = self.up_conv_qk2c(conv_qk2_b2)
        up_conv_qk2c = self.bn_up_conv_qk2c(up_conv_qk2c)
        up_conv_qk2c = self.relu(up_conv_qk2c)
        decoder1_1_qk2c = self.decoder1_1_qk2c(up_conv_qk2c)
        decoder1_1_qk2c = self.bn_decoder1_1_qk2c(decoder1_1_qk2c)
        decoder1_1_qk2c = self.relu(decoder1_1_qk2c)
        decoder1_2_qk2c = torch.cat([conv_qk2_a2, decoder1_1_qk2c], dim=1)
        decoder1_2_qk2c = self.decoder1_2_qk2c(decoder1_2_qk2c)
        decoder1_2_qk2c = self.bn_decoder1_2_qk2c(decoder1_2_qk2c)
        decoder1_2_qk2c = self.relu(decoder1_2_qk2c)


        conv_qk12 = torch.cat([decoder1_2_qk1c, decoder1_2_qk2c], dim=0)
        conv_qk12 = self.conv_qk12(conv_qk12)
        conv_qk12 = self.bn_qk12(conv_qk12)
        conv_qk12 = self.relu(conv_qk12)


        stacked_similarity = conv_qk12
        # stacked_similarity = self.adjust2(stacked_similarity)
        # print(kr.shape)

        # print('conv_qk12',conv_qk12.shape)
        # 在这里加unet
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 1, self.groups, H, H).sum(dim=1)
        # print('stacked_similarity',stacked_similarity.shape)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        # print('similarity',similarity.shape)

        # print('v',v.shape)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        # print('sv1',sv.shape)
        # sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        # print('sv2',sv.shape)
        # sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sv], dim=-1)
        # print('stacked_output1',stacked_output.shape)
        stacked_output = stacked_output.view(N * W, self.out_planes * 2, H)
        # print('stacked_output2',stacked_output.shape)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)
        # print('output',output.shape)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

###############################



##################################
class sab_net(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=3):
        super(sab_net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(256 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_1 = nn.Conv2d(imgchan, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = norm_layer(16, momentum=0.8)
        self.bn_max1_1 = norm_layer(16, momentum=0.8)

        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = norm_layer(16, momentum=0.8)
        self.bn_max1_2 = norm_layer(16, momentum=0.8)

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = norm_layer(32, momentum=0.8)
        self.bn_max2_1 = norm_layer(32, momentum=0.8)

        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = norm_layer(32, momentum=0.8)
        self.bn_max2_2 = norm_layer(32, momentum=0.8)

        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = norm_layer(64, momentum=0.8)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = norm_layer(64, momentum=0.8)
        self.bn_max3 = norm_layer(64, momentum=0.8)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = norm_layer(128, momentum=0.8)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = norm_layer(128, momentum=0.8)
        self.bn_max4 = norm_layer(128, momentum=0.8)

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1 = norm_layer(256, momentum=0.8)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2 = norm_layer(256, momentum=0.8)

        self.layer1 = self._make_layer(block, 32, 16, layers[0], stride=1, kernel_size=(img_size // 2))
        self.layer1_d = self._make_layer(block, 32, 16, layers[1], stride=1, kernel_size=(img_size // 2))
        # self.layer2 = self._make_layer(block_2, 64, 32, layers[2], stride=1, kernel_size=(img_size // 4))
        # self.layer2_d = self._make_layer(block_2, 64, 32, layers[3], stride=1, kernel_size=(img_size // 4))
        # Decoder
        self.up_conv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv1 = norm_layer(256, momentum=0.8)

        self.decoder1_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_1 = norm_layer(128, momentum=0.8)
        self.decoder1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder1_2 = norm_layer(128, momentum=0.8)

        self.up_conv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv2 = norm_layer(128, momentum=0.8)

        self.decoder2_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder2_1 = norm_layer(64, momentum=0.8)
        self.decoder2_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder2_2 = norm_layer(64, momentum=0.8)

        self.up_conv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv3 = norm_layer(64, momentum=0.8)

        self.decoder3_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder3_1 = norm_layer(32, momentum=0.8)
        self.decoder3_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder3_2 = norm_layer(32, momentum=0.8)

        self.up_conv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up_conv4 = norm_layer(32, momentum=0.8)

        self.decoder4_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder4_1 = norm_layer(16, momentum=0.8)
        self.decoder4_2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_decoder4_2 = norm_layer(16, momentum=0.8)

        self.relu = nn.GELU()
        self.adjust = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)
        # self.decoderf = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        # self.Sigmoid = nn.Sigmoid()

    def _make_layer(self, block, inplanes, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin = x.clone()
        # print('xin',xin.shape)
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.bn1_1(conv1_1)
        conv1_1 = self.relu(conv1_1)
        max_pool1_1 = F.max_pool2d(conv1_1, 2, 2)
        max_pool1_1 = self.bn_max1_1(max_pool1_1)

        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.bn1_2(conv1_2)
        conv1_2 = self.relu(conv1_2)
        max_pool1_2 = F.max_pool2d(conv1_2, 2, 2)
        max_pool1_2 = self.bn_max1_2(max_pool1_2)

        conv2_1 = torch.cat([max_pool1_1, max_pool1_2], dim=1)
        conv2_1 = self.conv2_1(conv2_1)
        conv2_1 = self.bn2_1(conv2_1)
        conv2_1 = self.relu(conv2_1)
        max_pool2_1 = F.max_pool2d(conv2_1, 2, 2)
        max_pool2_1 = self.bn_max2_1(max_pool2_1)

        layer1 = self.layer1(conv2_1)
        layer1 = torch.add(layer1, conv2_1)
        conv2_2 = self.conv2_2(layer1)
        conv2_2 = self.bn2_2(conv2_2)
        conv2_2 = self.relu(conv2_2)
        max_pool2_2 = F.max_pool2d(conv2_2, 2, 2)
        max_pool2_2 = self.bn_max2_2(max_pool2_2)

        conv3_1 = torch.cat([max_pool2_1, max_pool2_2], dim=1)
        conv3_1 = self.conv3_1(conv3_1)
        conv3_1 = self.bn3_1(conv3_1)
        conv3_1 = self.relu(conv3_1)
        # layer2 = self.layer2(conv3_1)
        # layer2 = torch.add(layer2, conv3_1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.bn3_2(conv3_2)
        conv3_2 = self.relu(conv3_2)
        max_pool3 = F.max_pool2d(conv3_2, 2, 2)
        max_pool3 = self.bn_max3(max_pool3)

        conv4_1 = self.conv4_1(max_pool3)
        conv4_1 = self.bn4_1(conv4_1)
        conv4_1 = self.relu(conv4_1)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.bn4_2(conv4_2)
        conv4_2 = self.relu(conv4_2)
        max_pool4 = F.max_pool2d(conv4_2, 2, 2)
        max_pool4 = self.bn_max4(max_pool4)

        conv5_1 = self.conv5_1(max_pool4)
        conv5_1 = self.bn5_1(conv5_1)
        conv5_1 = self.relu(conv5_1)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_2 = self.bn5_2(conv5_2)
        conv5_2 = self.relu(conv5_2)


        up_conv1 = self.up_conv1(conv5_2)
        up_conv1 = self.bn_up_conv1(up_conv1)
        up_conv1 = self.relu(up_conv1)

        decoder1_1 = self.decoder1_1(up_conv1)
        decoder1_1 = self.bn_decoder1_1(decoder1_1)
        decoder1_1 = self.relu(decoder1_1)
        decoder1_1 = torch.cat([conv4_2, decoder1_1], dim=1)
        decoder1_2 = self.decoder1_2(decoder1_1)
        decoder1_2 = self.bn_decoder1_2(decoder1_2)
        decoder1_2 = self.relu(decoder1_2)


        up_conv2 = self.up_conv2(decoder1_2)
        up_conv2 = self.bn_up_conv2(up_conv2)
        up_conv2 = self.relu(up_conv2)

        decoder2_1 = self.decoder2_1(up_conv2)
        decoder2_1 = self.bn_decoder2_1(decoder2_1)
        decoder2_1 = self.relu(decoder2_1)
        # layer2_d = self.layer2_d(conv3_2)
        # layer2_d = torch.add(layer2_d, conv3_2)
        decoder2_1 = torch.cat([conv3_2, decoder2_1], dim=1)
        decoder2_2 = self.decoder2_2(decoder2_1)
        decoder2_2 = self.bn_decoder2_2(decoder2_2)
        decoder2_2 = self.relu(decoder2_2)


        up_conv3 = self.up_conv3(decoder2_2)
        up_conv3 = self.bn_up_conv3(up_conv3)
        up_conv3 = self.relu(up_conv3)

        decoder3_1 = self.decoder3_1(up_conv3)
        decoder3_1 = self.bn_decoder3_1(decoder3_1)
        decoder3_1 = self.relu(decoder3_1)
        layer1_d = self.layer1_d(conv2_2)
        layer1_d = torch.add(layer1_d, conv2_2)
        decoder3_1 = torch.cat([layer1_d, decoder3_1], dim=1)
        decoder3_2 = self.decoder3_2(decoder3_1)
        decoder3_2 = self.bn_decoder3_2(decoder3_2)
        decoder3_2 = self.relu(decoder3_2)


        up_conv4 = self.up_conv4(decoder3_2)
        up_conv4 = self.bn_up_conv4(up_conv4)
        up_conv4 = self.relu(up_conv4)

        decoder4_1 = self.decoder4_1(up_conv4)
        decoder4_1 = self.bn_decoder4_1(decoder4_1)
        decoder4_1 = self.relu(decoder4_1)
        decoder4_1 = torch.cat([conv1_2, decoder4_1], dim=1)
        decoder4_2 = self.decoder4_2(decoder4_1)
        decoder4_2 = self.bn_decoder4_2(decoder4_2)
        decoder4_2 = self.relu(decoder4_2)

        x = self.adjust(decoder4_2)

        # x = self.soft(x)
        # pdb.set_trace()
        return x

    def forward(self, x):
        return self._forward_impl(x)


def SAB(pretrained=False, **kwargs):
    model = sab_net(AxialBlock, AxialBlock, [1, 1], s=0.125, **kwargs)
    return model
