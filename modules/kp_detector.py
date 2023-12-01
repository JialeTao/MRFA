from torch import nn
import torch
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter

from collections import OrderedDict
from torch.jit.annotations import Dict

import torch.nn.functional as F
from .util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d, DownBlock2d
from torch.nn import BatchNorm2d

from modules.util import make_coordinate_grid, coords_grid



class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion=32, num_kp=15, num_channels=3, max_features=1024, num_blocks=5, 
                 temperature=0.1, scale_factor=0.25, estimate_jacobian=False,estimate_occlusion=False):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=0)
        self.num_kp = num_kp

        self.estimate_jacobian = estimate_jacobian
        if estimate_jacobian:
            self.num_jacobian_maps = 1
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=0)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))

        self.estimate_occlusion = estimate_occlusion
        if estimate_occlusion:
            kp_occlusion = [DownBlock2d(self.predictor.out_filters,block_expansion,kernel_size=3, padding=1)]
            kp_occlusion.append(DownBlock2d(block_expansion,block_expansion*2,kernel_size=3, padding=1))
            kp_occlusion.append(DownBlock2d(block_expansion*2,block_expansion*3,kernel_size=3, padding=1))
            kp_occlusion.append(DownBlock2d(block_expansion*3,block_expansion*4,kernel_size=3, padding=1))
            kp_occlusion.append(nn.Conv2d(block_expansion*4,num_kp,kernel_size=(4,4),padding=0,stride=4))
            self.kp_occlusion = nn.Sequential(*kp_occlusion)

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_part_flows(self, source_image, kp_driving, kp_source, bg_params=None):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        num_kp = kp_driving['kp'].shape[1]
        identity_grid = make_coordinate_grid((h, w), type=kp_source['kp'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['kp'].view(bs, num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
            # n * 10 * h * w * 2
        driving_to_source = coordinate_grid + kp_source['kp'].view(bs, num_kp, 1, 1, 2)
        flow = driving_to_source - identity_grid 
        flow = (h-1)*flow / 2.0
        return flow.view(bs,num_kp*2,h,w)

        # # #adding background feature
        # # identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # # adding background feature
        # if bg_params is None:
        #     bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        # else:
        #     bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        #     bg_grid = to_homogeneous(bg_grid)
        #     bg_grid = torch.matmul(bg_params.view(bs, 1, 1, 1, 3, 3), bg_grid.unsqueeze(-1)).squeeze(-1)
        #     bg_grid = from_homogeneous(bg_grid)
        # sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)
        # return sparse_motions

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        kp = (heatmap * grid).sum(dim=(2, 3)) # N * 10 * 2
        # kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        out = self.gaussian2kp(heatmap)
        if self.estimate_jacobian:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            # N * 10 *2 *2
            out = {'kp': out, 'jacobian': jacobian}
        else:
            out = {'kp': out}
        
        if self.estimate_occlusion:
            # N*10*1*1
            kp_occlusion = self.kp_occlusion(feature_map)
            kp_occlusion = torch.sigmoid(kp_occlusion)
            out['kp_occlusion'] = kp_occlusion
        # out['heatmap'] = heatmap
        # out['feature_map'] = feature_map
        return out


class TPSKPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_tps, **kwargs):
        super(TPSKPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps*5*2)

        
    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {'kp': fg_kp.view(bs, self.num_tps*5, -1)}

        return out
