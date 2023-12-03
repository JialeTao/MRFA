import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math

from .util import AntiAliasInterpolation2d, make_coordinate_grid, coords_grid, bilinear_sampler, batch_bilinear_sampler, Hourglass, kp2gaussian
from .generator import OcclusionAwareGenerator


class CorrBlock:
    def __init__(self, corr, num_levels=2, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            if batch >1 and h1 >= 128:
                corr = batch_bilinear_sampler(corr, coords_lvl, h=h1, w=w1, mini_batch=1)
            else:
                corr = bilinear_sampler(corr, coords_lvl)
            # corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

class BasicMotionEncoder(nn.Module):
    def __init__(self, num_levels=2, radius=3):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = num_levels * (2*radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+96, 128-2, 3, padding=1)
        
    def forward(self, delta_flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(delta_flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, delta_flow], dim=1)

class RefineFlow(nn.Module):
    def __init__(self):
        super(RefineFlow, self).__init__()
        self.convc1 = nn.Conv2d(192, 128, 3, padding=1)
        # self.convc2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 2, 3, padding=1)
        self.convo1 = nn.Conv2d(256, 128, 3, padding=1)
        self.convo2 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, m_f, warp_f):
        c = F.relu(self.convc1(warp_f))
        ## c = F.relu(self.convc2(c))
        inp = torch.cat([m_f,c],dim=1)
        flow = self.conv2(F.relu(self.conv1(inp)))
        occ = self.convo2(F.relu(self.convo1(inp)))
        out = torch.cat([flow,occ],dim=1)
        return out, inp
        # return flow, inp


class RaftFlow(nn.Module):
    def __init__(self, prior_only=False, num_kp=10, dim=256, size=256, generator=None, driving_encoder=None, source_encoder=None):
        super(RaftFlow, self).__init__()
       
        self.scale = dim ** -0.5
        self.size = size
        ## basic flow resolution of the prior motion model
        self.h = size // 4
        self.w = size // 4
        self.prior_only = prior_only
        self.generator = OcclusionAwareGenerator(**generator)
        # self.down =  AntiAliasInterpolation2d(3, 0.25)

        ## start iteration from resolution of size//2**5
        channels = {
            size//32: 512,
            size//16: 512,
            size//8: 512,
            size//4: 256,
            size//2: 128,
            size: 64,
            # 512: 32,
        }
        self.total_iter = int(math.log(2**5, 2)) + 1
        self.num_iter = int(math.log(2**5, 2)) + 1
        self.basic_res_index = int(math.log(self.h//(size//32), 2))
        
        if not self.prior_only:
            
            ### driving and source structure encoder
            self.kp = Hourglass(**driving_encoder)
            self.kp_img = Hourglass(**source_encoder)
            self.kp_head = nn.Conv2d(in_channels=self.kp.out_filters, out_channels=dim, kernel_size=1,padding=0)
            self.kp_img_head = nn.Conv2d(in_channels=self.kp_img.out_filters, out_channels=dim, kernel_size=1,padding=0)
            # self._make_position_embedding(64, 64, num_kp, 'sine-full')
            # self.pos_embedding = rearrange(self.pos_embedding, 'b (h w) c -> b c h w', h=self.h, w=self.w)
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_kp, self.h, self.w))
            trunc_normal_(self.pos_embedding, std=.02)
            ### driving and source structure encoder

            ### flow updater
            self.corr_enc = BasicMotionEncoder()
            self.refine = RefineFlow()
            ## project different-scale warped features to the same dimension
            self.to_context = nn.ModuleList()
            for i in range(self.num_iter):
                f_channel = channels[(size//32) * (2**i)]
                self.to_context.append(nn.Conv2d(f_channel, 192, 1, padding=0))
            ### flow updater
        
    def forward(self, kp_s, kp_d, dense_motion, img, img_full):
        
        feature = self.generator.encode(img_full)
        if img is None:
            img = self.down(img_full)
        b,_,h,w = img.shape

        out_warp_f = []
        out_occlusion = []
        out_flow = []
        
        out_warp_f_c = []
        out_occlusion_c = []

        ### prior only
        if self.prior_only:
            flow = dense_motion['deformation']
            occlusion = dense_motion['occlusion']
            for i in range(self.total_iter):
                if flow.shape[2] != feature[i].shape[2]:
                    flow_res = F.interpolate(flow.permute(0,3,1,2), size=feature[i].shape[2:], mode='bilinear',align_corners=True)
                    occlusion_res = F.interpolate(occlusion, size=feature[i].shape[2:], mode='bilinear',align_corners=True)
                else:
                    flow_res = flow.permute(0,3,1,2)
                    occlusion_res = occlusion
                out_warp_f.append(F.grid_sample(feature[i], flow_res.permute(0,2,3,1)))
                out_occlusion.append(F.sigmoid(occlusion_res))
            warp_img = F.grid_sample(img_full, flow_res.permute(0,2,3,1))
            out = self.generator.decode(out_warp_f, warp_img, out_occlusion)
            for i in range(len(out_occlusion)):
                out_occlusion[i] = F.interpolate(out_occlusion[i], size=self.size, mode='bilinear',align_corners=True) 
            occlusion = torch.cat(out_occlusion,dim=3)
            return out, warp_img, occlusion
        ### prior only

        ### compute correlation volume at basic resolution
        kp_s = kp2gaussian(kp_s, (h,w), 0.1) + self.pos_embedding
        kp_d = kp2gaussian(kp_d, (h,w), 0.1) + self.pos_embedding
        fe_s = self.kp_img(torch.cat([kp_s,img],dim=1))
        fe_d = self.kp(kp_d)
        k_s = self.kp_img_head(fe_s)
        q_d = self.kp_head(fe_d)
        f_s = rearrange(k_s, 'b c h w -> b (h w) c', h=self.h, w=self.w)
        f_d = rearrange(q_d, 'b c h w -> b (h w) c', h=self.h, w=self.w)
        corr_volume = torch.einsum('bic,bjc->bij',f_d,f_s) * self.scale
        ### compute correlation volume at basic resolution
        
        ### prior motion initialization
        id_grid = coords_grid(b, self.h, self.w, corr_volume.device)
        init_flow = (self.h-1)*(dense_motion['deformation'].permute(0,3,1,2)+1) / 2.0 - id_grid
        init_occlusion = dense_motion['occlusion']
        ### prior motion initialization

        # ### NPMR only initialization
        # id_grid = coords_grid(b, self.h, self.w, corr_volume.device)
        # id_grid = rearrange(id_grid, 'b c h w -> b (h w) c', h=self.h, w=self.w)
        # init_flow = torch.einsum('bij,bjc->bic', corr_volume.softmax(-1), id_grid)
        # init_flow = rearrange(init_flow, 'b (h w) c-> b c h w', h=self.h, w=self.w)
        # id_grid = rearrange(id_grid, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        # init_flow = init_flow - id_grid
        # init_occlusion = None
        # ### NPMR only initialization

        ## start iter resolution: size//32 = (size//4) // 8, where size//4 defines the basic flow resolution
        flow = F.interpolate(init_flow, scale_factor=1.0/8.0,mode='bilinear',align_corners=True) / 8.0
        occlusion = F.interpolate(init_occlusion, scale_factor=1.0/8.0,mode='bilinear',align_corners=True) if init_occlusion is not None else init_occlusion
        ## reshape for pooling the driving image dimension, fitting different resolutions 
        corr_volume = rearrange(corr_volume, 'b (h w) n -> (b n) h w', h=self.h, w=self.w).unsqueeze(1)

        
        for i in range(self.total_iter):
            ## identity grid with current-iter resolution
            id_grid = coords_grid(b, self.size//32*(2**i), self.size//32*(2**i), corr_volume.device)
            flow_sample = flow
            id_grid_sample = id_grid

            ### computing upsampled or downsampled structure correlation volume
            if i < self.basic_res_index:
                corr_volume_res = F.avg_pool2d(corr_volume, 2**(self.basic_res_index-i), stride=2**(self.basic_res_index-i))
                scale = 2**(self.basic_res_index-i)
            elif i == self.basic_res_index:
                corr_volume_res = corr_volume
                scale = 1
            elif i > self.basic_res_index:
                ## for resolution larger than basic flow resolution, sampling with the basic flow resolution
                corr_volume_res = corr_volume
                scale = 0.5**(i-self.basic_res_index)
                flow_sample = F.interpolate(flow, size=self.h, mode='bilinear',align_corners=True) * scale
                id_grid_sample = coords_grid(b, self.h,self.w, corr_volume.device)
                scale = 1
            ### computing upsampled or downsampled structure correlation volume

            if i < self.num_iter:
                ## reshape for pooling the source image dimension, thus to obtain correlation pyramids
                corr_volume_res = rearrange(corr_volume_res, '(b n) c h w -> (b h w) c n', n=self.h*self.w)
                corr_volume_res = rearrange(corr_volume_res, 'b c (p q) -> b c p q', p=self.h,q=self.w)
                ## building correlation pyramids
                corr_fn = CorrBlock(corr_volume_res)
                ## sampling correlation features from the pyramids
                corr = corr_fn((flow_sample+id_grid_sample)*scale)
                if i > self.basic_res_index:
                    ## reshape from basic resolution to current resolution
                    corr = F.interpolate(corr, size=flow.shape[2], mode='bilinear',align_corners=True)
                ## non-prior-based motion feature
                m_f = self.corr_enc(flow, corr)
                
                warp_f = bilinear_sampler(feature[i], (flow+id_grid).permute(0,2,3,1))
                warp_f = F.relu(self.to_context[i](warp_f))

                d_flow, _= self.refine(m_f,warp_f)
                flow_w = flow + d_flow[:,0:2,:,:]
                d_occ = d_flow[:,2:,:,:]
                if occlusion is not None:
                    occlusion = occlusion + d_occ
                else:
                    occlusion = d_occ
            else:
                flow_w = flow
            
            out = bilinear_sampler(feature[i], (flow_w+id_grid).permute(0,2,3,1))
            out_occlusion.append(F.sigmoid(occlusion))
            out_warp_f.append(out)

            ## coarse warping (prior motion flow)
            if i!= self.basic_res_index:
                flow_res = F.interpolate(dense_motion['deformation'].permute(0,3,1,2), size=feature[i].shape[2:], mode='bilinear',align_corners=True)
                occlusion_res = F.interpolate(dense_motion['occlusion'], size=feature[i].shape[2:], mode='bilinear',align_corners=True)
            else:
                flow_res = dense_motion['deformation'].permute(0,3,1,2)
                occlusion_res = dense_motion['occlusion']
            out_warp_f_c.append(F.grid_sample(feature[i], flow_res.permute(0,2,3,1)))
            out_occlusion_c.append(F.sigmoid(occlusion_res))
            ## coarse warping (prior motion flow)

            ### updating flow by directly scaling initial flow
            if i < self.num_iter - 1:
                ## scale between basic resolution and updated resolution
                scale = 2 ** (self.basic_res_index-i) / 2.0
                d_f = F.interpolate(d_flow[:,0:2,:,:], scale_factor=2, mode='bilinear',align_corners=True) * 2
                ## flow refinement of current iteration
                flow = d_f + F.interpolate(init_flow, size=self.size//32 * (2 ** (i+1)), mode='bilinear',align_corners=True) / scale
                ## flow refinement of previous iterations
                if i ==0:
                    d_f_pre = d_f
                else:
                    flow = flow + F.interpolate(d_f_pre, scale_factor=2, mode='bilinear',align_corners=True) * 2
                    d_f_pre = d_f + F.interpolate(d_f_pre, scale_factor=2, mode='bilinear',align_corners=True) * 2
                ## occlusion refinement
                d_occ = F.interpolate(d_occ, scale_factor=2, mode='bilinear',align_corners=True) 
                occlusion = d_occ + F.interpolate(init_occlusion, size=self.size//32 * (2 ** (i+1)), mode='bilinear',align_corners=True)
                if i ==0:
                    d_occ_pre = d_occ
                else:
                    occlusion = occlusion + F.interpolate(d_occ_pre, scale_factor=2, mode='bilinear',align_corners=True)
                    d_occ_pre = d_occ + F.interpolate(d_occ_pre, scale_factor=2, mode='bilinear',align_corners=True)
            ## for num_iter < total_iter
            # elif i < self.total_iter-1:
            #     ## without refinement
            #     flow = F.interpolate(flow, scale_factor=2, mode='bilinear',align_corners=True) * 2
            #     occlusion = F.interpolate(occlusion, scale_factor=2, mode='bilinear',align_corners=True) 
        
        warp_img = bilinear_sampler(img_full, (flow+id_grid).permute(0,2,3,1))
        out = self.generator.decode(out_warp_f, warp_img, out_occlusion, out_warp_f_c, out_occlusion_c)

        out_occlusion.append(F.sigmoid(init_occlusion))
        out_occlusion_vis = []
        for i in range(len(out_occlusion)):
            out_occlusion_vis.append(F.interpolate(out_occlusion[i], size=self.size, mode='bilinear',align_corners=True))
        occlusion = torch.cat(out_occlusion_vis,dim=3)
        
        return out, warp_img, occlusion
