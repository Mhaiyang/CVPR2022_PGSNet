"""
 @Time    : 2021/11/1 21:32
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : CVPR2022_PGSNet
 @File    : pgsnet.py
 @Function:
 
"""
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from torch.cuda.amp import autocast as autocast

from backbone.resnet.resnet import resnet18


###################################################################
# ########################## Conformer ############################
###################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), out_features=None,
                 skip=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer,
                       drop=drop)
        self.skip = skip

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.skip:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2).contiguous()
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        H = int(H)
        W = int(W)
        H_T = int(self.up_stride * H)
        W_T = int(self.up_stride * W)
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).contiguous().reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H_T, W_T))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        conv_features = []
        tran_features = []
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, h, w] -> [N, 64, h / 4, w / 4]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2).contiguous()
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        conv_features.append(x)
        tran_features.append(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

            conv_features.append(x)
            tran_features.append(x_t)

        # conv classification
        # x_p = self.pooling(x).flatten(1)
        # conv_cls = self.conv_cls_head(x_p)

        # trans classification
        # x_t = self.trans_norm(x_t)
        # tran_cls = self.trans_cls_head(x_t[:, 0])

        # return [conv_cls, tran_cls]
        return conv_features, tran_features


###################################################################
# ################# Early Dynamic Attention (EDA) #################
###################################################################
class Early_Fusion(nn.Module):
    def __init__(self, backbone_path):
        super(Early_Fusion, self).__init__()
        net = resnet18(backbone_path=backbone_path)
        self.get_value = nn.Sequential(net.conv1, net.bn1, net.relu,
                                       net.maxpool, net.layer1,
                                       net.layer2,
                                       net.layer3,
                                       net.layer4,
                                       net.avgpool,
                                       nn.Conv2d(512, 1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        r = x[:, 0, :, :][:, None, :, :]
        g = x[:, 1, :, :][:, None, :, :]
        b = x[:, 2, :, :][:, None, :, :]

        v_r = self.get_value(r.repeat(1, 3, 1, 1))
        v_g = self.get_value(g.repeat(1, 3, 1, 1))
        v_b = self.get_value(b.repeat(1, 3, 1, 1))

        concat = torch.cat([v_r, v_g, v_b], 1)
        weight = self.softmax(concat)

        w_r = weight[:, 0, :, :][:, None, :, :]
        w_g = weight[:, 1, :, :][:, None, :, :]
        w_b = weight[:, 2, :, :][:, None, :, :]

        out = torch.cat([w_r * r, w_g * g, w_b * b], 1)

        return out


###################################################################
# ############## Global Context Generation (GCG) ##################
###################################################################
class Cross(nn.Module):
    def __init__(self, dim=576, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Cross, self).__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # [B, 26*26+1, 576]
        B, N, C = x.shape
        x = self.norm_x(x)
        y = self.norm_y(y)

        q = self.q(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + x
        infer_fea = self.norm_z(infer_fea)

        return infer_fea


class Fusion_T(nn.Module):
    def __init__(self, dim=576):
        super(Fusion_T, self).__init__()
        self.cross_image_aolp = Cross(dim)
        self.cross_image_dolp = Cross(dim)
        self.cross_aolp_image = Cross(dim)
        self.cross_aolp_dolp = Cross(dim)
        self.cross_dolp_image = Cross(dim)
        self.cross_dolp_aolp = Cross(dim)
        self.fusion_t = nn.Sequential(nn.Linear(int(dim * 6), dim), nn.GELU())

    def forward(self, image_t, aolp_t, dolp_t):
        cross_image_aolp = self.cross_image_aolp(image_t, aolp_t)
        cross_image_dolp = self.cross_image_dolp(image_t, dolp_t)
        cross_aolp_image = self.cross_aolp_image(aolp_t, image_t)
        cross_aolp_dolp = self.cross_aolp_dolp(aolp_t, dolp_t)
        cross_dolp_image = self.cross_dolp_image(dolp_t, image_t)
        cross_dolp_aolp = self.cross_dolp_aolp(dolp_t, aolp_t)
        concat = torch.cat([cross_image_aolp, cross_image_dolp, cross_aolp_image, cross_aolp_dolp,
                            cross_dolp_image, cross_dolp_aolp], 2)
        fusion_t = self.fusion_t(concat)

        return fusion_t


###################################################################
# ######## Dynamic Multimodal Feature Integration (DMFI) ##########
###################################################################
class MSDP(nn.Module):
    def __init__(self, channel):
        super(MSDP, self).__init__()
        channel_half = channel // 2
        channel_triple = channel * 3

        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel_half, 3, 1, 1), nn.BatchNorm2d(channel_half), nn.PReLU())
        self.scale1 = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.query_conv1 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.key_conv1 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.value_conv1 = nn.Conv2d(channel_half, channel_half, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel_half, 3, 1, 1), nn.BatchNorm2d(channel_half), nn.PReLU())
        self.scale2 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.query_conv2 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.key_conv2 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.value_conv2 = nn.Conv2d(channel_half, channel_half, 1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel_half, 3, 1, 1), nn.BatchNorm2d(channel_half), nn.PReLU())
        self.scale3 = nn.AdaptiveAvgPool2d(output_size=(9, 9))
        self.query_conv3 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.key_conv3 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.value_conv3 = nn.Conv2d(channel_half, channel_half, 1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(nn.Conv2d(channel, channel_half, 3, 1, 1), nn.BatchNorm2d(channel_half), nn.PReLU())
        self.scale4 = nn.AdaptiveAvgPool2d(output_size=(11, 11))
        self.query_conv4 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.key_conv4 = nn.Conv2d(channel_half, channel_half // 8, 1)
        self.value_conv4 = nn.Conv2d(channel_half, channel_half, 1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.fusion = nn.Sequential(nn.Conv2d(channel_triple, channel, 1), nn.BatchNorm2d(channel), nn.PReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv1 = self.conv1(x)
        scale1 = self.scale1(conv1)
        B, C, H, W = scale1.size()
        proj_query1 = self.query_conv1(scale1).view(B, -1, H * W).permute(0, 2, 1)
        proj_key1 = self.key_conv1(scale1).view(B, -1, H * W)
        energy1 = torch.bmm(proj_query1, proj_key1)
        attention1 = self.softmax(energy1)
        proj_value1 = self.value_conv1(scale1).view(B, -1, H * W)
        out1 = torch.bmm(proj_value1, attention1.permute(0, 2, 1))
        out1 = out1.view(B, C, H, W)
        out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)
        out1 = self.gamma1 * out1 + conv1

        conv2 = self.conv2(x)
        scale2 = self.scale2(conv2)
        B, C, H, W = scale2.size()
        proj_query2 = self.query_conv2(scale2).view(B, -1, H * W).permute(0, 2, 1)
        proj_key2 = self.key_conv2(scale2).view(B, -1, H * W)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(scale2).view(B, -1, H * W)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(B, C, H, W)
        out2 = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)
        out2 = self.gamma2 * out2 + conv2

        conv3 = self.conv3(x)
        scale3 = self.scale3(conv3)
        B, C, H, W = scale3.size()
        proj_query3 = self.query_conv3(scale3).view(B, -1, H * W).permute(0, 2, 1)
        proj_key3 = self.key_conv3(scale3).view(B, -1, H * W)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(scale3).view(B, -1, H * W)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(B, C, H, W)
        out3 = F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)
        out3 = self.gamma3 * out3 + conv3

        conv4 = self.conv4(x)
        scale4 = self.scale4(conv4)
        B, C, H, W = scale4.size()
        proj_query4 = self.query_conv4(scale4).view(B, -1, H * W).permute(0, 2, 1)
        proj_key4 = self.key_conv4(scale4).view(B, -1, H * W)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(scale4).view(B, -1, H * W)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(B, C, H, W)
        out4 = F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)
        out4 = self.gamma4 * out4 + conv4

        concat = torch.cat([x, out1, out2, out3, out4], 1)
        fusion = self.fusion(concat)

        return fusion


class Dynamic_Weight(nn.Module):
    def __init__(self, scale, dim=576):
        super(Dynamic_Weight, self).__init__()
        self.scale = scale
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.map1 = nn.Linear(dim, 1)
        self.map2 = nn.Linear(dim, 1)
        self.map3 = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_t, aolp_t, dolp_t):
        B, N, _ = image_t.shape

        t1 = self.norm1(image_t)[:, 1:, :]
        t2 = self.norm2(aolp_t)[:, 1:, :]
        t3 = self.norm3(dolp_t)[:, 1:, :]

        map1 = self.map1(t1).transpose(1, 2).contiguous().reshape(B, 1, int(np.sqrt(N - 1)), int(np.sqrt(N - 1)))
        map2 = self.map2(t2).transpose(1, 2).contiguous().reshape(B, 1, int(np.sqrt(N - 1)), int(np.sqrt(N - 1)))
        map3 = self.map3(t3).transpose(1, 2).contiguous().reshape(B, 1, int(np.sqrt(N - 1)), int(np.sqrt(N - 1)))

        map1 = F.interpolate(map1, size=(self.scale, self.scale), mode='bilinear', align_corners=True)
        map2 = F.interpolate(map2, size=(self.scale, self.scale), mode='bilinear', align_corners=True)
        map3 = F.interpolate(map3, size=(self.scale, self.scale), mode='bilinear', align_corners=True)

        map = torch.cat([map1, map2, map3], 1)
        weight_map = self.softmax(map)

        image_weight_map = weight_map[:, 0, :, :][:, None, :, :]
        aolp_weight_map = weight_map[:, 1, :, :][:, None, :, :]
        dolp_weight_map = weight_map[:, 2, :, :][:, None, :, :]

        return image_weight_map, aolp_weight_map, dolp_weight_map


class Fusion_C(nn.Module):
    def __init__(self, scale, dim=576):
        super(Fusion_C, self).__init__()
        self.dynamic_weight = Dynamic_Weight(scale, dim)

    def forward(self, image, aolp, dolp, image_t, aolp_t, dolp_t):
        image_weight_map, aolp_weight_map, dolp_weight_map = self.dynamic_weight(image_t, aolp_t, dolp_t)
        fusion_c = image_weight_map * image + \
                   aolp_weight_map * aolp + \
                   dolp_weight_map * dolp

        return fusion_c


###################################################################
# ################ Attention Enhancement (AE) #####################
###################################################################
class segmentation_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(segmentation_token_inference, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        t_all = self.norm1(fea)
        t_seg, t = t_all[:, 0, :][:, None, :], t_all[:, 1:, :]
        # t_seg [B, 1, 576]  t [B, 26*26, 576]

        q = self.q(t).reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(t_seg).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(t_seg).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).contiguous().reshape(B, N - 1, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:, :]
        infer_fea = self.norm2(infer_fea)

        return infer_fea


class Global_Attention(nn.Module):
    def __init__(self, channel, embed_dim=576, hidden_dim=128):
        super(Global_Attention, self).__init__()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        self.sigmoid4 = nn.Sigmoid()

        self.vector_c = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Conv2d(channel, hidden_dim, 1, 1, 0), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                                      nn.Conv2d(hidden_dim, channel, 1, 1, 0))
        self.vector_t = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                      nn.Conv2d(embed_dim, hidden_dim, 1, 1, 0), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                                      nn.Conv2d(hidden_dim, channel, 1, 1, 0))

        self.map_c = nn.Conv2d(channel, 1, 7, 1, 3)
        self.segmentation_token_inference = segmentation_token_inference(embed_dim)
        self.map_t = nn.Linear(embed_dim, 1)

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())

    def forward(self, x, y):
        # x: conv    y: global guidance
        B, _, h, w = x.shape
        _, N, C = y.shape

        vector_c = self.vector_c(x)
        t_features_for_vector = y[:, 1:, :][:, :, :, None].transpose(1, 2).contiguous()
        vector_t = self.vector_t(t_features_for_vector)
        vector = self.sigmoid1(vector_c) * self.sigmoid2(vector_t)
        channel_attention = x * vector
        add1 = x + channel_attention

        map_c = self.map_c(add1)
        t_features_for_map = self.segmentation_token_inference(y)
        map_t = self.map_t(t_features_for_map).transpose(1, 2).contiguous().reshape(B, 1, int(np.sqrt(N - 1)), int(np.sqrt(N - 1)))
        map_t = F.interpolate(map_t, size=(h, w), mode='bilinear', align_corners=True)
        map = self.sigmoid3(map_c) * self.sigmoid4(map_t)
        spatial_attention = add1 * map
        add2 = add1 + spatial_attention

        conv = self.conv(add2)

        return conv


###################################################################
# ################### Basic Decoder (BD) ##########################
###################################################################
class Decoder(nn.Module):
    def __init__(self, conv_channel):
        super(Decoder, self).__init__()
        self.conv_channel = conv_channel
        self.conv_output_channel = int(self.conv_channel / 2)

        self.conv_up = nn.Sequential(nn.Conv2d(self.conv_channel, self.conv_output_channel, 3, 1, 1),
                                     nn.BatchNorm2d(self.conv_output_channel), nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv_fusion = nn.Sequential(nn.Conv2d(self.conv_output_channel, self.conv_output_channel, 3, 1, 1),
                                         nn.BatchNorm2d(self.conv_output_channel), nn.ReLU())

    def forward(self, x, sc):
        conv_add = self.conv_up(x) + sc
        conv_fusion = self.conv_fusion(conv_add)

        return conv_fusion


###################################################################
# ###################### Fusion_t Predict #########################
###################################################################
class T_Predict(nn.Module):
    def __init__(self, embed_dim=576):
        super(T_Predict, self).__init__()
        self.segmentation_token_inference = segmentation_token_inference(embed_dim)
        self.map = nn.Linear(embed_dim, 1)

    def forward(self, t):
        B, N, _ = t.shape

        t_features_for_predict = self.segmentation_token_inference(t)
        map = self.map(t_features_for_predict).transpose(1, 2).contiguous().reshape(B, 1, int(np.sqrt(N - 1)), int(np.sqrt(N - 1)))

        return map


###################################################################
# ########################## NETWORK ##############################
###################################################################
class PGSNet(nn.Module):
    def __init__(self, backbone_path1=None, backbone_path2=None, backbone_path3=None, backbone_path4=None):
        super(PGSNet, self).__init__()
        # early fusion
        self.early_fusion_aolp = Early_Fusion(backbone_path4)
        self.early_fusion_dolp = Early_Fusion(backbone_path4)

        # backbone
        self.conformer1 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                                    num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.conformer2 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                                    num_heads=9, mlp_ratio=4, qkv_bias=True)
        self.conformer3 = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                                    num_heads=9, mlp_ratio=4, qkv_bias=True)

        if backbone_path1 is not None:
            self.conformer1.load_state_dict(torch.load(backbone_path1, map_location=torch.device('cpu')))
            print("From {} Load Weights Succeed!".format(backbone_path1))
        if backbone_path2 is not None:
            self.conformer2.load_state_dict(torch.load(backbone_path2, map_location=torch.device('cpu')))
            print("From {} Load Weights Succeed!".format(backbone_path2))
        if backbone_path3 is not None:
            self.conformer3.load_state_dict(torch.load(backbone_path3, map_location=torch.device('cpu')))
            print("From {} Load Weights Succeed!".format(backbone_path3))

        # channel reduction
        self.image_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.image_cr3 = nn.Sequential(nn.Conv2d(1536, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.image_cr2 = nn.Sequential(nn.Conv2d(768, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.image_cr1 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())

        self.aolp_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())

        self.dolp_cr4 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())

        # msdp
        self.msdp = MSDP(256)

        # fusion
        self.fusion_t = Fusion_T(576)
        self.fusion_c = Fusion_C(13, 576)

        # decoder
        self.decoder43 = Decoder(256)
        self.decoder32 = Decoder(128)
        self.decoder21 = Decoder(64)

        # global attention
        self.ga4 = Global_Attention(256, 576)
        self.ga3 = Global_Attention(128, 576)
        self.ga2 = Global_Attention(64, 576)
        self.ga1 = Global_Attention(32, 576)

        # predict
        self.predict_c1 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t1 = T_Predict(576)
        self.predict_c2 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t2 = T_Predict(576)
        self.predict_c3 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict_t3 = T_Predict(576)
        self.predict_t = T_Predict(576)
        self.predict4 = nn.Conv2d(256, 1, 7, 1, 3)
        self.predict3 = nn.Conv2d(128, 1, 7, 1, 3)
        self.predict2 = nn.Conv2d(64, 1, 7, 1, 3)
        self.predict1 = nn.Conv2d(32, 1, 7, 1, 3)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    @autocast()
    def forward(self, image, aolp, dolp):
        # image, aolp, dolp: [batch_size, channel=3, h, w]
        image_conv_features, image_tran_features = self.conformer1(image)
        image_layer0 = image_conv_features[0]  # [-1, 256, h/4, w/4]
        image_layer1 = image_conv_features[3]  # [-1, 256, h/4, w/4]
        image_layer2 = image_conv_features[7]  # [-1, 512, h/8, w/8]
        image_layer3 = image_conv_features[10]  # [-1, 1024, h/16, w/16]
        image_layer4 = image_conv_features[11]  # [-1, 1024, h/32, w/32]
        image_t0 = image_tran_features[0]  # [-1, (h/16)^2+1, 384] 384-->577
        image_t1 = image_tran_features[3]  # [-1, (h/16)^2+1, 384]
        image_t2 = image_tran_features[7]  # [-1, (h/16)^2+1, 384]
        image_t3 = image_tran_features[10]  # [-1, (h/16)^2+1, 384]
        image_t4 = image_tran_features[11]  # [-1, (h/16)^2+1, 384]

        early_fusion_aolp = self.early_fusion_aolp(aolp)
        aolp_conv_features, aolp_tran_features = self.conformer2(early_fusion_aolp)
        aolp_layer0 = aolp_conv_features[0]  # [-1, 256, h/4, w/4]
        aolp_layer1 = aolp_conv_features[3]  # [-1, 256, h/4, w/4]
        aolp_layer2 = aolp_conv_features[7]  # [-1, 512, h/8, w/8]
        aolp_layer3 = aolp_conv_features[10]  # [-1, 1024, h/16, w/16]
        aolp_layer4 = aolp_conv_features[11]  # [-1, 1024, h/32, w/32]
        aolp_t0 = aolp_tran_features[0]  # [-1, (h/16)^2+1, 384] 384-->577
        aolp_t1 = aolp_tran_features[3]  # [-1, (h/16)^2+1, 384]
        aolp_t2 = aolp_tran_features[7]  # [-1, (h/16)^2+1, 384]
        aolp_t3 = aolp_tran_features[10]  # [-1, (h/16)^2+1, 384]
        aolp_t4 = aolp_tran_features[11]  # [-1, (h/16)^2+1, 384]

        early_fusion_dolp = self.early_fusion_dolp(dolp)
        dolp_conv_features, dolp_tran_features = self.conformer3(early_fusion_dolp)
        dolp_layer0 = dolp_conv_features[0]  # [-1, 256, h/4, w/4]
        dolp_layer1 = dolp_conv_features[3]  # [-1, 256, h/4, w/4]
        dolp_layer2 = dolp_conv_features[7]  # [-1, 512, h/8, w/8]
        dolp_layer3 = dolp_conv_features[10]  # [-1, 1024, h/16, w/16]
        dolp_layer4 = dolp_conv_features[11]  # [-1, 1024, h/32, w/32]
        dolp_t0 = dolp_tran_features[0]  # [-1, (h/16)^2+1, 384] 384-->577
        dolp_t1 = dolp_tran_features[3]  # [-1, (h/16)^2+1, 384]
        dolp_t2 = dolp_tran_features[7]  # [-1, (h/16)^2+1, 384]
        dolp_t3 = dolp_tran_features[10]  # [-1, (h/16)^2+1, 384]
        dolp_t4 = dolp_tran_features[11]  # [-1, (h/16)^2+1, 384]

        # channel reduction
        image_cr4 = self.image_cr4(image_layer4)
        image_cr3 = self.image_cr3(image_layer3)
        image_cr2 = self.image_cr2(image_layer2)
        image_cr1 = self.image_cr1(image_layer1)

        aolp_cr4 = self.aolp_cr4(aolp_layer4)

        dolp_cr4 = self.dolp_cr4(dolp_layer4)

        # fusion t
        fusion_t = self.fusion_t(image_t4, aolp_t4, dolp_t4)

        # fusion c
        fusion_c = self.fusion_c(image_cr4, aolp_cr4, dolp_cr4, image_t4, aolp_t4, dolp_t4)

        # msdp
        msdp = self.msdp(fusion_c)

        # 4
        ga4 = self.ga4(msdp, fusion_t)

        # 3
        decoder43 = self.decoder43(ga4, image_cr3)
        ga3 = self.ga3(decoder43, fusion_t)

        # 2
        decoder32 = self.decoder32(ga3, image_cr2)
        ga2 = self.ga2(decoder32, fusion_t)

        # 1
        decoder21 = self.decoder21(ga2, image_cr1)
        ga1 = self.ga1(decoder21, fusion_t)

        # predict
        predict_c1 = self.predict_c1(image_cr4)
        predict_t1 = self.predict_t1(image_t4)
        predict_c2 = self.predict_c2(aolp_cr4)
        predict_t2 = self.predict_t2(aolp_t4)
        predict_c3 = self.predict_c3(dolp_cr4)
        predict_t3 = self.predict_t3(dolp_t4)
        predict_t = self.predict_t(fusion_t)
        predict4 = self.predict4(ga4)
        predict3 = self.predict3(ga3)
        predict2 = self.predict2(ga2)
        predict1 = self.predict1(ga1)

        # rescale
        predict_c1 = F.interpolate(predict_c1, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t1 = F.interpolate(predict_t1, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_c2 = F.interpolate(predict_c2, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t2 = F.interpolate(predict_t2, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_c3 = F.interpolate(predict_c3, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t3 = F.interpolate(predict_t3, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict_t = F.interpolate(predict_t, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=image.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=image.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict_c1, predict_t1, predict_c2, predict_t2, predict_c3, predict_t3, \
                   predict_t, predict4, predict3, predict2, predict1

        return torch.sigmoid(predict1)
