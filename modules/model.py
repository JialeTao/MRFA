from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
# import timm
import math

from .generator import  OcclusionAwareGenerator
from .util import AntiAliasInterpolation2d, make_coordinate_grid
from .raft import RaftFlow
from .kp_detector import KPDetector, TPSKPDetector
from .dense_motion import DenseMotionNetwork, TPSDenseMotionNetwork
from .bg_motion_predictor import BGMotionPredictor
from .transformer.pose_tokenpose_b import get_pose_net
from torch.autograd import grad


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        # N * (HW) * 2 * 1
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            # 1 * (HW) * 1 * 2 - 1 * 1 * 25 *2
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian



class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # X = X.clamp(-1, 1)
        # X = X / 2 + 0.5
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels=3):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict



class MRFA(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(MRFA, self).__init__()

        self.cfg = cfg
        train_params = cfg.train_params
        self.train_params = train_params
        self.scales = train_params['scales']
        self.loss_weights = train_params['loss_weights']
        self.pyramid = ImagePyramide(self.scales).cuda()
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19().cuda()

        ##################################################
        if train_params['prior_model'] == 'fomm':
            self.encoder = KPDetector(**cfg.fomm_kp_detector)
            self.dense_motion = DenseMotionNetwork(**cfg.dense_motion)
        elif train_params['prior_model'] == 'tpsm':
            self.encoder = TPSKPDetector(**cfg.tpsm_kp_detector)
            self.dense_motion = TPSDenseMotionNetwork(**cfg.tpsm_dense_motion)
            self.dropout_epoch=train_params['dropout_epoch']
            self.dropout_maxp=train_params['dropout_maxp']
            self.dropout_startp=train_params['dropout_startp']
            self.dropout_inc_epoch=train_params['dropout_inc_epoch']
        elif train_params['prior_model'] == 'mtia':
            self.encoder = get_pose_net(cfg.mtia_kp_detector, is_train=True)
            self.dense_motion = DenseMotionNetwork(**cfg.dense_motion)
        else:
            print("please specify the prior motion model")

        if train_params['bg_start'] < train_params['num_epochs']:
            self.bg_predictor = BGMotionPredictor()
        self.bg_start = train_params['bg_start']

        self.decoder = RaftFlow(**cfg.raft_flow)
        self.down =  AntiAliasInterpolation2d(3, 0.25)

    def forward(self, x, epoch=100, is_train=True):

        kp_s = self.encoder(x['source'])
        kp_d = self.encoder(x['driving'])

        img_down = self.down(x['source'])
        if(epoch>=self.bg_start):
            bg_param = self.bg_predictor(x['source'], x['driving'])
        else:
            bg_param = None

        if self.train_params['prior_model'] == 'tpsm' and epoch < self.dropout_epoch:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        else:
            dropout_flag = False
            dropout_p = 0
        dense_motion = self.dense_motion(x['source'], kp_d, kp_s, bg_param=bg_param, dropout_flag=dropout_flag, dropout_p=dropout_p)

        if self.train_params['prior_model'] == 'tpsm':
            kp_s_value = kp_s['kp'].view(x['source'].shape[0], -1, 5, 2).mean(2)
            kp_d_value = kp_d['kp'].view(x['driving'].shape[0], -1, 5, 2).mean(2)
        else:
            kp_s_value = kp_s['kp']
            kp_d_value = kp_d['kp']

        gen, warp_img, occlusion = self.decoder(kp_s_value, kp_d_value, dense_motion, img=img_down, img_full=x['source'])
        warp_img = torch.cat([warp_img, occlusion.repeat(1,3,1,1)], dim=3)

        loss_values = {}

        if not is_train:
            return gen, warp_img, loss_values, kp_s['kp'], kp_d['kp']


        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(gen)
        value_total = 0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

            for i, weight in enumerate(self.loss_weights['perceptual']):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['equivariance']  != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.encoder(transformed_frame)
            value = torch.abs(kp_d['kp'] - transform.warp_coordinates(transformed_kp['kp']))
            value = value.mean()
            loss_values['equivariance'] = self.loss_weights['equivariance'] * value

            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['kp']),
                                                    transformed_kp['jacobian'])
                normed_driving = torch.inverse(kp_d['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
                value = torch.abs(eye - value)
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if epoch >= self.bg_start:
            bg_param_reverse = self.bg_predictor(x['driving'], x['source'])
            value = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = 10 * value


        return gen, warp_img, loss_values, kp_s['kp'], kp_d['kp']

