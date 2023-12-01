import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math

from .util import make_coordinate_grid, coords_grid, bilinear_sampler
from .styledecoder import StyledConv, ModulatedConv2d, Decoder, ConstantInput, ConvLayer, ToRGB

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

class Residual(nn.Module):
    def __init__(self, fn, num_keypoints=10):
        super().__init__()
        self.fn = fn
        self.num_keypoints = num_keypoints
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class QKVPreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)





class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False, num_img_tokens=None):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    # @get_local('dots')
    # @get_local('attn')
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # print(attn[0,0,-1,-1])
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class QKVAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    # @get_local('dots')
    # @get_local('attn')
    def forward(self, x, k=None, v=None, mask = None):
        b, n, _, h = *x.shape, self.heads
        # if k is None:
        #     k=x
        # if v is None:
        #     v=x
        q = self.to_q(x)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        # print(attn[0,0,-1,-1])
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        # # linear attention
        # out = k.softmax(dim=-2).permute(0,1,3,2)
        # out = torch.einsum('bhcn,bhnd->bhcd', out, v)
        # out = torch.einsum('bhnc,bhcd->bhnd', q.softmax(dim=-1), out)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)
        # warp_k = torch.einsum('bhij,bhjd->bhid', attn, k)
        # warp_k = rearrange(warp_k, 'b h n d -> b n (h d)')
        return out

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False, num_patches=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head, num_img_tokens=num_patches)), num_keypoints=self.num_keypoints),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None,pos=None):
        img_token = x[:,self.num_keypoints:]
        for idx,(attn, ff) in enumerate(self.layers):
            if (idx>0 and self.all_attn):
                x[:,self.num_keypoints:] += pos
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False, num_patches=256, v_pos=False):
        super().__init__()
        self.encoder= nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        self.v_pos = v_pos
        for d in range(depth):
            self.encoder.append(nn.ModuleList([
                Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head)), num_keypoints=self.num_keypoints),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None, pos=None):
        for idx,(attn, ff) in enumerate(self.encoder):
            if self.all_attn:
                if self.v_pos:
                    x[:,self.num_keypoints:] += pos
                    q = x
                    k = x
                    v = x
                else:
                    v = x.clone()
                    x[:,self.num_keypoints:] += pos
                    q = x
                    k = x
            x = attn(q, k=k, v=v, mask = mask)
            x = ff(x)
        return x

class TransformerDecoder(nn.Module):
    # def __init__(self, dim, depth, heads, mlp_dim, dropout, seq_len, scale_with_head=False):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, scale_with_head=False):
        super().__init__()
        self.decoder= nn.ModuleList([])
        for d in range(depth):
            self.decoder.append(nn.ModuleList([
                # Residual(QKVPreNorm(dim, LinearQKVAttention(dim,seq_len,k=128,heads=heads))),
                # Residual(QKVPreNorm(dim, LinearQKVAttention(dim,seq_len,k=128,heads=heads))),
                # Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head))),
                Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, k, v, mask = None, pos=None):
        for idx,(cross_attn, ff) in enumerate(self.decoder):
        # for idx,(self_attn, cross_attn, ff) in enumerate(self.decoder):
            # x = self_attn(x,k=x,v=x)
            x = cross_attn(x,k=k,v=v)
            x = ff(x)
        return x

class SparseTransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, scale_with_head=False):
        super().__init__()
        self.decoder= nn.ModuleList([])
        for d in range(depth):
            self.decoder.append(nn.ModuleList([
                Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, k, v, mask = None, pos=None):
        for idx,(self_attn, cross_attn, ff) in enumerate(self.decoder):
            x = cross_attn(x, k=k, v=v, mask=mask)
            x = ff(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class QKPreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, q, k, v=None, img=None, **kwargs):
        return self.fn(self.norm(q), self.norm(k), v, img, **kwargs)
class QKAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_k = nn.Linear(dim, dim, bias = False)
        # self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_q = nn.Linear(dim, dim, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )
    # @get_local('dots')
    # @get_local('attn')
    def forward(self, x, k=None, v=None, img=None, mask = None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(k)
        # v = self.to_v(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        if v is not None:
            v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        # attn = F.softmax(dots, dim=-1)
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')

        # dots = dots[:,:,:,0:4096]
        # attn = F.softmax(dots.sum(1), dim=-1)
        # warped_src = torch.einsum('bhij,bhjd->bhid', dots.softmax(dim=-1), k[:,:,0:4096,:])
        # warped_src = rearrange(warped_src, 'b h n d -> b n (h d)')
        # img = rearrange(img, 'b c h w -> b (h w) c')
        # warped_img = torch.einsum('bij,bjd->bid', dots.sum(1).softmax(-1), img)
        # warped_img = rearrange(warped_img, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)

        # # linear attention
        # out = k.softmax(dim=-2).permute(0,1,3,2)
        # out = torch.einsum('bhcn,bhnd->bhcd', out, v)
        # out = torch.einsum('bhnc,bhcd->bhnd', q.softmax(dim=-1), out)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return dots
        # return out



class PositionTransformer(nn.Module):
    def __init__(self, *, feature_size, patch_size, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, dropout = 0., emb_dropout = 0., pos_embedding_type="learn"):
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        # patch_dim = channels * patch_size[0] * patch_size[1]
        # assert pos_embedding_type in ['sine','learnable','sine-full']

        self.h = feature_size[0] // patch_size[0]
        self.w = feature_size[1] // patch_size[1]

        self.h = 32
        self.w = 32
        self.heads = heads

        self.inplanes = 64
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        ## multi resolution
        self._make_position_embedding(32, 32, dim, 'sine-full')
        ## multi resolution

        # self._make_position_embedding(self.w, self.h, dim, 'sine-full')
        # self.pos_embedding_oppo = (1-self.pos_embedding ** 2) ** 0.5
        self.pos_emb_src = self.pos_embedding
        self.pos_emb_dri = self.pos_embedding
        # self.pos_emb_src = nn.Parameter(torch.zeros(1, num_patches, dim))
        # self.pos_emb_dri = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # self.dropout = nn.Dropout(0.1)

        # self.position_transformer = TransformerDecoder(dim, depth, heads, mlp_dim, dropout, scale_with_head=True)
        # self.warp_transformer = QKPreNorm(dim,QKAttention(dim, heads, dropout, scale_with_head=True))
        self.add_keys = nn.Parameter(torch.zeros(1, 100, dim))
        self.add_values = nn.Parameter(torch.zeros(1, 100, dim))

        channels = {
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        depths = {
            32: depth,
            64: 1,
            128: 1,
            256: 1,
            512: 1,
        }
        self.resolution = [32,64,128,256]
        self.warp_transformer = nn.ModuleList()
        self.upconv_dri = nn.ModuleList()
        self.position_transformer = nn.ModuleList()

        self.position_transformer.append(TransformerDecoder(dim, depth, heads, mlp_dim, dropout, scale_with_head=True))
        self.warp_transformer.append(QKAttention(dim, heads, dropout, scale_with_head=True))

        # self.position_transformer.append(TransformerDecoder(dim, depth, heads, mlp_dim, dropout, scale_with_head=True))
        # self.warp_transformer.append(QKAttention(dim, heads, dropout, scale_with_head=True))
        # self.add_keys_low = nn.Parameter(torch.zeros(1, 25, dim))
        # self.add_values_low = nn.Parameter(torch.zeros(1, 25, dim))
        # self.id_grid = nn.Parameter(make_coordinate_grid((32, 32), type=self.pos_embedding.type()), requires_grad=False).unsqueeze(0)
        # self.id_grid = rearrange(self.id_grid, 'b h w c -> b (h w) c', h=32, w=32)
        

        # for i in range(len(self.resolution)):
        #     res = self.resolution[i]
        #     self.position_transformer.append(TransformerDecoder(dim, depths[self.resolution[i]], heads, mlp_dim, dropout, scale_with_head=True))
        #     self.warp_transformer.append(QKAttention(dim, heads, dropout, scale_with_head=True))
        #     # self.warp_transformer.append(QKPreNorm(dim, LinearQKVAttention(dim, seq_len=res*res, k = 128, heads=heads,to_v=False)))
        #     # if i != len(self.resolution) - 1:
        #     #     self.upconv_dri.append(StyledConv(dim, dim, 3, dim, upsample=True, blur_kernel=[1, 3, 3, 1], demodulate=False))
    
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # self.occlusion = nn.Conv2d(dim,1,3,padding=1)
        # self.occlusion = nn.ModuleList([Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, scale_with_head=True))), nn.Linear(dim,1)])
        # self.occlusion = QKAttention(dim, 1, dropout, scale_with_head=True)
        # self.occlusion_token = nn.Parameter(torch.zeros(1, 1, dim))
        # self.skip_conv = StyledConv(dim, dim//2, 3, dim, upsample=True, blur_kernel=[1, 3, 3, 1], demodulate=False)

        # self.upconv_src = StyledConv(dim, dim, 3, dim, upsample=True, blur_kernel=[1, 3, 3, 1], demodulate=False)
        if apply_init:
            self.apply(self._init_weights)


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos



    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # Batched index_select
    def batched_index_select(self, t, dim, index):
        # selcet to dim 1 of indx
        dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), t.size(2))
        out = t.gather(dim, dummy) # b x e x f
        return out

    def forward(self, motion, feature, mask = None):
        motion = rearrange(motion, 'b c h w -> b (h w) c')
        # sine_motion = torch.sin(F.sigmoid(motion) * (2 * math.pi))
        # cosi_motion = torch.cos(F.sigmoid(motion) * (2 * math.pi))
        # motion_w = F.sigmoid(motion_w)
        p = self.patch_size
        b, _, _, _ = feature[0].shape


        pos_emb_src = self.pos_emb_src
        # # pos_emb_dri = self.dropout(self.pos_emb_dri)
        pos_emb_dri = self.pos_emb_dri
        # # pos_emb_src = pos_emb_src + motion.sum(1,keepdim=True)

        # pos_emb_src = rearrange(self.pos_emb_src, 'b (h w) c -> b c h w',  h=256, w=256)
        # pos_emb_src = F.interpolate(pos_emb_src, size=(self.resolution[0],self.resolution[0]), mode='bilinear')
        # pos_emb_src = rearrange(pos_emb_src, 'b c h w -> b (h w) c',  h=self.resolution[0], w=self.resolution[0])
        # pos_emb_dri = pos_emb_src

        pos_emb_dri = pos_emb_dri.repeat(b,1,1)
        pos_emb_src = pos_emb_src.repeat(b,1,1)
        # pos_embedding_oppo = (1-pos_emb_src ** 2) ** 0.5
        # x = self.position_transformer(pos_emb_dri, pos_emb_src*sine_motion + pos_embedding_oppo*cosi_motion, pos_emb_src*sine_motion + pos_embedding_oppo*cosi_motion)
        x = self.position_transformer[0](pos_emb_dri, pos_emb_src+motion, pos_emb_src+motion)

        out = []
        f = feature[0] 
        h, w=f.shape[2:]
        f = rearrange(f, 'b c h w -> b (h w) c', h=h, w=w)
        add_keys = self.add_keys.repeat(b,1,1)
        add_values = self.add_values.repeat(b,1,1)
        keys = torch.cat([pos_emb_src+motion, add_keys], dim=1)
        values = torch.cat([f,add_values], dim=1)
        # keys = pos_emb_src+motion
        # values = f
        warp_f, dots= self.warp_transformer[0](x, keys, values)
        warp_f = rearrange(warp_f, 'b (h w) c -> b c h w', h=h, w=w)
        out.append(warp_f)

        ## res16
        # pos_emb_src = rearrange(pos_emb_src, 'b (h w) c -> b c h w',  h=32, w=32)
        # pos_emb_src = F.interpolate(pos_emb_src, size=(16,16), mode='bilinear')
        # pos_emb_src = rearrange(pos_emb_src, 'b c h w -> b (h w) c',  h=16, w=16)
        # x = self.position_transformer[1](pos_emb_src, pos_emb_src+motion, pos_emb_src+motion)
        # add_keys = self.add_keys_low.repeat(b,1,1)
        # add_values = self.add_values_low.repeat(b,1,1)
        # keys = torch.cat([pos_emb_src+motion, add_keys], dim=1)
        # f = rearrange(feature[0], 'b c h w -> b (h w) c', h=16, w=16)
        # values = torch.cat([f,add_values], dim=1)
        # warp_f_low, dots_low = self.warp_transformer[1](x, keys, values)
        # warp_f_low = rearrange(warp_f_low, 'b (h w) c -> b c h w', h=16, w=16)
        # out.append(warp_f_low)
        # out.append(warp_f)
        ## res16

        # # attn b * h * 1024 * 1124
        # dots = rearrange(dots[:,:,:,0:1024].sum(1), 'b (h w) n-> b h w n', h=32, w=32)
        # occlusion = F.softmax(dots.sum(1), dim=-1)
        # occlusion = occlusion[:,:,0:1024].sum(-1,keepdim=True)
        # occlusion = rearrange(occlusion, 'b (h w) c -> b c h w', h=32, w=32)
        dots = dots[:,:,:,0:1024].sum(1).softmax(dim=-1)

        id_grid = make_coordinate_grid((32, 32), type=dots.type()).unsqueeze(0)
        id_grid = rearrange(id_grid, 'b h w c -> b (h w) c', h=32, w=32)
        flow = torch.einsum('bij,bjc->bic', dots, id_grid.repeat(b,1,1))
        flow = rearrange(flow, 'b (h w) c-> b c h w', h=32, w=32)

        # flow_res = F.interpolate(flow, size=(16,16), mode='bilinear')
        # warp_f_res = F.grid_sample(feature[0], flow_res.permute(0,2,3,1))
        # out.append(warp_f_res)
        # out.append(warp_f)
        
        for i in range(len(self.resolution)-1):
            res = self.resolution[i+1]
            flow_res = F.interpolate(flow, size=(res,res), mode='bilinear')
            warp_f = F.grid_sample(feature[i+1], flow_res.permute(0,2,3,1))
            # occlusion_res = F.interpolate(occlusion, size=(res,res), mode='bilinear')
            # warp_f = warp_f * occlusion_res
            out.append(warp_f)
        # warp_img = F.grid_sample(feature[-1], flow_res.permute(0,2,3,1))
        # warp_img = warp_img * occlusion_res
        # out.append(warp_img)


        # for i in range(len(self.resolution)-1):

        #     res_current = self.resolution[i]
        #     res = self.resolution[i+1]
        #     # print(i, res)

        #     # attn_res = F.interpolate(dots, size=(res,res), mode='bilinear')
        #     # attn_res = rearrange(attn_res, 'b (i j) h w -> b (h w) i j',  i=32, j=32, h=res, w=res)
        #     # attn_res = F.interpolate(attn_res, size=(res,res), mode='bilinear')
        #     # attn_res = rearrange(attn_res, '(b e) (h w) i j-> b e (i j) (h w)', e=self.heads, i=res, j=res, h=res, w=res)
        #     # _, ind = torch.sort(attn_res, descending=True, dim=-1)
        #     # sample_ind = ind[:,:,0:100]
        #     # sample_ind = rearrange(sample_ind, 'b n k -> (b n) k')

        #     pos_emb_src = rearrange(self.pos_emb_src, 'b (h w) c -> b c h w',  h=128, w=128)
        #     pos_emb_src = F.interpolate(pos_emb_src, size=(res,res), mode='bilinear')
        #     pos_emb_src = rearrange(pos_emb_src, 'b c h w -> b (h w) c',  h=res, w=res)

        #     f = feature[i+1]
        #     h, w=f.shape[2:]
        #     f = rearrange(f, 'b c h w -> b (h w) c', h=h, w=w)
        #     x = rearrange(x, 'b (h w) c -> b c h w',  h=res_current, w=res_current)
        #     x = self.upconv_dri[i](x.contiguous())
        #     x = rearrange(x, 'b c h w -> b (h w) c',  h=res, w=res)
        #     # x = x.unsqueeze(1)

        #     # pos_emb_src = pos_emb_src.unsqueeze(1).repeat(1,res*res,1,1) # b (h w) (h w) c
        #     # pos_emb_src = rearrange(pos_emb_src, 'b (h w) n c -> (b h w) n c', h=res, w=res)
        #     # pos_emb_src = self.batched_index_select(pos_emb_src, 1, sample_ind)

        #     # f = f.unsqueeze(1).repeat(1,res*res,1,1) # b (h w) (h w) c
        #     # f = rearrange(f, 'b (h w) n c -> (b h w) n c', h=res, w=res)
        #     # f = self.batched_index_select(f, 1, sample_ind)
        
        #     x = self.position_transformer[i+1](x, pos_emb_src+motion, pos_emb_src+motion)
        #     warp_f = self.warp_transformer[i+1](x, pos_emb_src+motion, f)
        #     warp_f = rearrange(f, 'b (h w) c -> b c h w', h=res, w=res)
        #     # attn_res = F.softmax(attn_res, dim=-1)
        #     # warp_f = torch.einsum('bhij,bhjd->bhid', attn_res, f)
        #     # warp_f = rearrange(warp_f, 'b e (h w) d -> b (e d) h w', h=h,w=w)
        #     out.append(warp_f)


        # # x = rearrange(x, 'b (h w) c -> b c h w',  h=f.shape[2]//2, w=f.shape[3]//2)
        # # x = self.upconv_dri(x.contiguous())
        # # x = rearrange(x, 'b c h w -> b (h w) c',  h=f.shape[2], w=f.shape[3])
        # # pos_emb_src = rearrange(pos_emb_src, 'b (h w) c -> b c h w',  h=self.h, w=self.w)
        # # pos_emb_src =  F.interpolate(pos_emb_src, size=f.shape[2:], mode='bilinear')
        # # pos_emb_src = rearrange(pos_emb_src, 'b c h w -> b (h w) c',  h=f.shape[2], w=f.shape[3])
        # # # pos_embedding_oppo = (1-pos_emb_src ** 2) ** 0.5

        # f = rearrange(f, 'b c h w -> b (h w) c', h=h, w=w)
        # add_keys = self.add_keys.repeat(b,1,1)
        # add_values = self.add_values.repeat(b,1,1)
        # # # keys = pos_emb_src+motion
        # # # values = f
        # # # keys = torch.cat([pos_emb_src*sine_motion + pos_embedding_oppo*cosi_motion, add_keys], dim=1)
        # keys = torch.cat([pos_emb_src+motion, add_keys], dim=1)
        # values = torch.cat([f,add_values], dim=1)
        # # print(x.device, self.warp_transformer[0].to_q.weight.device)
        # out, attn = self.warp_transformer[]0(x, keys, values)
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        # # skip = self.skip_conv(skip.contiguous())
        # # out.append(skip)

        return out
        # return warp_f, flow
        # return out, occlusion

class Refine(nn.Module):
    def __init__(self,in_channels=3, dim=512, style_dim=512, blur_kernel=[1,3,3,1]):
        super(Refine, self).__init__()

        # self.convs = nn.ModuleList()
        # self.convs.append(StyledConv(in_channels, dim//4, 3, style_dim, blur_kernel=blur_kernel))
        # self.convs.append(StyledConv(dim//4, dim, 3, style_dim, blur_kernel=blur_kernel))
        # self.convs.append(StyledConv(dim, dim, 3, style_dim, blur_kernel=blur_kernel))
        # self.convs.append(StyledConv(dim, dim+1, 7, style_dim, blur_kernel=blur_kernel))
        self.conv1 = StyledConv(in_channels, in_channels, 3, style_dim, blur_kernel=blur_kernel)
        self.conv2 = StyledConv(in_channels, in_channels, 3, style_dim, blur_kernel=blur_kernel)
        # self.pred = StyledConv(in_channels, dim, 3, style_dim, blur_kernel=blur_kernel)
        # self.occ = StyledConv(in_channels, 1, 3, style_dim, blur_kernel=blur_kernel)
        self.pred = ModulatedConv2d(in_channels, dim, 1, style_dim, demodulate=False)
        # self.bias1 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.occ = ModulatedConv2d(in_channels, 1, 1, style_dim, demodulate=False)
        # self.bias2 = nn.Parameter(torch.zeros(1, 1, 1, 1))
    def forward(self, x, motion):
        # for id, conv in enumerate(self.convs):
        #     x = conv(x, motion)
        x = self.conv1(x, motion)
        x = self.conv2(x, motion)
        m = self.pred(x, motion)
        occ = self.occ(x, motion)
        # return m
        return m, occ

class Occlusion(nn.Module):
    def __init__(self,in_channels=3, style_dim=512, blur_kernel=[1,3,3,1]):
        super(Occlusion, self).__init__()

        self.conv1 = StyledConv(in_channels, in_channels, 3, style_dim, blur_kernel=blur_kernel)
        self.conv2 = StyledConv(in_channels, in_channels, 3, style_dim, blur_kernel=blur_kernel)
        self.occ = ModulatedConv2d(in_channels, 1, 1, style_dim, demodulate=False)
        # self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
    def forward(self, x, motion):
        x = self.conv1(x, motion)
        x = self.conv2(x, motion)
        x = self.occ(x, motion)
        # occ = self.occ(x, motion)
        # return m
        return x

class RefineTransformer(nn.Module):
    # def __init__(self, dim, depth, heads, mlp_dim, dropout, seq_len, scale_with_head=False):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, scale_with_head=False):
        super().__init__()
        self.decoder= nn.ModuleList([])
        self.refiner = nn.ModuleList([])
        self.occs = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.input = ConstantInput(256, size=64)
        for d in range(depth):
            self.decoder.append(nn.ModuleList([
                # Residual(QKVPreNorm(dim, LinearQKVAttention(dim,seq_len,k=128,heads=heads))),
                # Residual(QKVPreNorm(dim, LinearQKVAttention(dim,seq_len,k=128,heads=heads))),
                # Residual(QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head))),
                QKVPreNorm(dim, QKVAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head)),
                QKPreNorm(dim, QKAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
            # self.decoder.append(QKPreNorm(dim, QKAttention(dim, heads = heads, dropout = dropout, scale_with_head=scale_with_head)))
            if d==0:
                self.refiner.append(Refine(in_channels=dim, dim=dim))
            else:
                self.refiner.append(Refine(in_channels=dim+3, dim=dim))
            # self.refiner.append(nn.Linear(512,dim))
            self.occs.append(Occlusion(in_channels=256, style_dim=512))
            self.dec.append(Decoder(in_channel=256, out_channel=3))
            self.convs.append(ConvLayer(in_channel=256, out_channel=256, kernel_size=3))
        # self.refiner.append(Refine(in_channels=256, dim=dim))
        # id_grid = make_coordinate_grid((64, 64), type=torch.Tensor).unsqueeze(0)
        # id_grid = rearrange(id_grid, 'b h w c -> b (h w) c', h=64, w=64)
        # self.id_grid = nn.Parameter(id_grid, requires_grad=False)
    # def forward(self, x, key, add_keys, add_values, motion1,motion2, fea=None, mask = None, pos=None):
    def forward(self, x, key, motion, fea=None, img=None, mask = None, pos=None):
        attn = None
        # occlusion = torch.zeros(fea.shape[0],1,fea.shape[2],fea.shape[3]).to(fea.device)+5
        out = torch.zeros(fea.shape[0],3,256,256).to(fea.device)
        # add_values = rearrange(add_values, 'b n (e c) -> b e n c', e=1)
        warp_img = img
        x0 = x.clone()
        # warp_f = rearrange(x, 'b (h w) c-> b c h w', h=64, w=64)
        for i, ((cross_attn,cross_attn2, ff), refine, dec, occs, conv) in enumerate(zip(self.decoder, self.refiner, self.dec, self.occs, self.convs)):
        # for idx,(self_attn, cross_attn, ff) in enumerate(self.decoder):
            # x = self_attn(x,k=x,v=x)
            # add_key = add_keys[:,i*50:(i+1)*50,:]
            # add_key = add_keys[:,i*100:(i+1)*100,:]
            # add_value = add_values[:,:,i*100:(i+1)*100,:]
            
            warp_f = rearrange(x-x0, 'b (h w) c-> b c h w', h=64, w=64)
            if i!=0:
                warp_f = torch.cat([warp_f,warp_img],dim=1)
            # # m = refine(warp_f.contiguous(), motion)
            m, occlusion = refine(warp_f.contiguous(), motion)
            # # occlusion += occ
            m = rearrange(m, 'b c h w-> b (h w) c', h=64, w=64)
            # m = refine(motion)
            # k = key+m
            # v = key+m
            # key = key+m.unsqueeze(1)
            x = x+m
            xr, dots = cross_attn(x,k=key, v=key)
            x = x+xr
            x = ff(x)
            
            dots = cross_attn2(x,k=key)
            # dots = cross_attn2(x,k=torch.cat([key, add_key],dim=1))
            attn = dots.softmax(dim=-1)

            # add_value = add_values[:,:,(i-1)*50:(i*50),:]
            f = rearrange(fea, 'b (e c) h w-> b e (h w) c', e=1, h=64, w=64)
            im = rearrange(img, 'b (e c) h w-> b e (h w) c', e=1, h=64, w=64)
            f = torch.cat([im,f], dim=3)
            # f = torch.cat([f,add_value], dim=2)
            warp_f_c = torch.einsum('bhij,bhjc->bhic', attn, f)
            warp_f_c = rearrange(warp_f_c, 'b e (h w) c-> b (e c) h w', e=1, h=64, w=64)
            warp_img = warp_f_c[:,0:3,:,:]
            warp_f_c = warp_f_c[:,3:,:,:]
            # occlusion = occs(warp_f_c.contiguous(), m)
            # if i == 0:
            #     occlusion = occ
            # else:
            #     occlusion += occ
            # warp_f = conv(warp_f_c*F.sigmoid(occlusion))
            # warp_f = warp_f_c * F.sigmoid(occlusion) + warp_f * (1-F.sigmoid(occlusion))
            if i==0:
                rec = dec(warp_f_c)
            else:
                rec = dec(warp_f_c*F.sigmoid(occlusion))
            # img = dec(warp_f_c)
            out += rec
            # flow = torch.einsum('bij,bjc->bic', attn, self.id_grid.repeat(x.shape[0],1,1))
            # flow = rearrange(flow, 'b (h w) c-> b h w c', h=64, w=64)
            # warp_f = F.grid_sample(f, flow)
        # # f = rearrange(fea, 'b c h w-> b (h w) c', h=64, w=64)
        # # warp_f = torch.einsum('bij,bjc->bic', attn, f)
        # # warp_f = rearrange(warp_f, 'b (h w) c-> b c h w', h=64, w=64)
        # # add_value = add_values[:,:,i*50:(i+1)*50,:]
        # f = rearrange(fea, 'b (e c) h w-> b e (h w) c', e=8, h=64, w=64)
        # # f = torch.cat([f,add_value], dim=2)
        # warp_f = torch.einsum('bhij,bhjc->bhic', attn, f)
        # warp_f = rearrange(warp_f, 'b e (h w) c-> b (e c) h w', e=8, h=64, w=64)
        # # m, occlusion, warp_f = refine(warp_f.contiguous(), motion)
        # # occlusion += occ
        # # m = rearrange(m, 'b c h w-> b (h w) c', h=64, w=64)
        return out, dots, F.sigmoid(occlusion)
        # return out, dots

class CorrBlock:
    def __init__(self, corr, num_levels=1, radius=3, h=64, w=64, p=64, q=64):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        # corr = CorrBlock.corr(fmap1, fmap2)
        # corr = rearrange(corr, 'b (h w) (p q) -> b h w p q',h=h,w=w,p=p,q=q).unsqueeze(3)
        self.unit = 2.0 / 63.0

        # batch, h1, w1, dim, h2, w2 = corr.shape
        # corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
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

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = 1 * (2*4 + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, delta_flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(delta_flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, delta_flow], dim=1)

class SmallMotionEncoder(nn.Module):
    def __init__(self):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = 1 * (2*3 + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 32, 1, padding=0)
        # self.convc2 = nn.Conv2d(64, 32, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(96, 62, 3, padding=1)

    def forward(self, delta_flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(delta_flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, delta_flow], dim=1)

class RefinFlow(nn.Module):
    def __init__(self):
        super(RefinFlow, self).__init__()
        self.convc1 = nn.Conv2d(192, 128, 3, padding=1)
        self.convc2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, m_f, warp_f):
        c = F.relu(self.convc1(warp_f))
        c = F.relu(self.convc2(c))
        inp = torch.cat([m_f,c],dim=1)
        out = self.conv2(F.relu(self.conv1(inp)))
        return out

class SinglePositionTransformer(nn.Module):
    def __init__(self, *, feature_size, patch_size, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, dropout = 0., emb_dropout = 0., pos_embedding_type="learn"):
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        # patch_dim = channels * patch_size[0] * patch_size[1]
        # assert pos_embedding_type in ['sine','learnable','sine-full']

        self.h = feature_size[0] // patch_size[0]
        self.w = feature_size[1] // patch_size[1]

        self.h = 64
        self.w = 64
        self.heads = heads

        self.inplanes = 64
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

    
        self._make_position_embedding(64, 64, dim, 'sine-full')
        self.pos_emb_src = self.pos_embedding
        self.pos_emb_dri = self.pos_embedding
        self.dropout = nn.Dropout(emb_dropout)
        # self.add_keys = nn.Parameter(torch.zeros(1, 400, dim))
        # self.add_values = nn.Parameter(torch.zeros(1, 400, 256))

        self.position_transformer = TransformerDecoder(dim, depth, heads, mlp_dim, dropout, scale_with_head=True)
        # self.position_transformer = RefineTransformer(dim, depth, heads, mlp_dim, dropout, scale_with_head=True)
        # self.warp_transformer = QKPreNorm(dim, QKAttention(dim, 1, dropout, scale_with_head=True))
        # self.linear = nn.Linear(dim*2+3, dim, bias=False)
        # self.linear2 = nn.Linear(dim*2, dim, bias=True)

        # # out 128,82
        self.corr_enc = SmallMotionEncoder()
        self.refine = RefinFlow()
        self.scale = 512 ** -0.5
        self.norm = nn.LayerNorm(dim)

        channels = {
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        self.num_iter = int(math.log(256, 2)) - 2
        self.to_imgs = nn.ModuleList()
        self.to_context = nn.ModuleList()
        # 345678
        for i in range(self.num_iter):
            f_channel = channels[2 ** (i+3)]
            self.to_imgs.append(ToRGB(f_channel))
            self.to_context.append(nn.Conv2d(f_channel, 192, 1, padding=0))


        if apply_init:
            self.apply(self._init_weights)


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos



    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # Batched index_select
    def batched_index_select(self, t, dim, index):
        # selcet to dim 1 of indx
        dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), t.size(2))
        out = t.gather(dim, dummy) # b x e x f
        return out

    # def forward(self, motion, feature, feature_g, img, mask = None):
    def forward(self, l_d, l_s, feature, img, img_full, mask = None):
        # motion = rearrange(motion, 'b c h w -> b (h w) c')
        p = self.patch_size
        b, _, _, _ = img.shape
        pos_emb_src = self.pos_emb_src
        pos_emb_dri = self.pos_emb_dri
    
        pos_emb_dri = pos_emb_dri.repeat(b,1,1)
        pos_emb_src = pos_emb_src.repeat(b,1,1)

        # add_keys = self.add_keys.repeat(b,1,1)
        # add_values = self.add_values.repeat(b,1,1)

        # img = rearrange(img, 'b c h w -> b (h w) c')
        # kv = self.linear(torch.cat([pos_emb_src+motion, img], dim=-1))
        # x = self.position_transformer(pos_emb_dri, kv, kv)
        # motion = motion.unsqueeze(1)
        l_d = l_d.unsqueeze(1)
        l_s = l_s.unsqueeze(1)
        x = self.position_transformer(pos_emb_dri+l_d, pos_emb_src+l_s, pos_emb_src+l_s)
        corr_volume = torch.einsum('bid,bjd->bij', self.norm(x),self.norm(pos_emb_src+l_s)) * self.scale
        # keys = torch.cat([pos_emb_src+l_s, add_keys], dim=1)
        # f = rearrange(feature, 'b c h w -> b (h w) c', h=64, w=64)
        # values = torch.cat([f,add_values], dim=1)
        # attn, out = self.warp_transformer(x, keys, values)
        # out = rearrange(out, 'b (h w) c -> b c h w', h=64, w=64)
        # sim = dots.sum(1).softmax(dim=-1)
        # sim = rearrange(sim, 'b (h w) n -> b h w n')
        # out, attn, occlusion = self.position_transformer(pos_emb_dri, pos_emb_src, motion, fea=feature, img=img)

        # out = []
        # f = feature
        # h, w=f.shape[2:]
        # f = rearrange(f, 'b c h w -> b (h w) c', h=h, w=w)
        # add_keys = self.add_keys.repeat(b,1,1)
        # add_values = self.add_values.repeat(b,1,1)
        # keys = torch.cat([kv, add_keys], dim=1)
        # keys = torch.cat([pos_emb_src+motion, add_keys], dim=1)
        # values = torch.cat([f,add_values], dim=1)
        # keys = pos_emb_src+motion
        # keys = motion
        # # values = f
        # attn  = self.warp_transformer(x, keys)
        id_grid = coords_grid(b, 64, 64, corr_volume.device)
        # id_grid = make_coordinate_grid((64, 64), type=attn.type()).unsqueeze(0).repeat(b,1,1,1)
        id_grid = rearrange(id_grid, 'b c h w -> b (h w) c', h=64, w=64)
        flow = torch.einsum('bij,bjc->bic', corr_volume.softmax(-1), id_grid)
        flow = rearrange(flow, 'b (h w) c-> b c h w', h=64, w=64)
        flow = F.interpolate(flow, size=(8,8), mode='bilinear',align_corners=True) / 8.0
        # id_grid = rearrange(id_grid, 'b (h w) c -> b c h w', h=64, w=64)
        id_grid = coords_grid(b, 8, 8, corr_volume.device)
        flow = flow - id_grid
        # flow = id_grid - id_grid
        skip = None
        occlusion = None
        corr_volume = rearrange(corr_volume, 'b (h w) n -> (b n) h w', h=64, w=64).unsqueeze(1)

        for i in range(self.num_iter):
            id_grid = coords_grid(b, 2**(i+3), 2**(i+3), corr_volume.device)
            if i < 3:
                corr_volume_res = F.avg_pool2d(corr_volume, 2**(3-i), stride=2**(3-i))
                scale = 2**(3-i)
            elif i == 3:
                corr_volume_res = corr_volume
                scale = 1
            elif i > 3:
                corr_volume_res = F.interpolate(corr_volume, scale_factor=2**(i-3), mode='bilinear',align_corners=True)
                scale = 0.5**(i-3)
            corr_volume_res = rearrange(corr_volume_res, '(b n) c h w -> (b h w) c n', n=4096)
            corr_volume_res = rearrange(corr_volume_res, 'b c (p q) -> b c p q', p=64,q=64)
            corr_fn = CorrBlock(corr_volume_res)
            # warp_f = F.grid_sample(feature, flow.permute(0,2,3,1))
            # delta_flow = flow - id_grid
            corr = corr_fn((flow.detach()+id_grid)*scale)
            m_f = self.corr_enc(flow.detach().clone(), corr)
            warp_f = bilinear_sampler(feature[i], (flow.detach()+id_grid).permute(0,2,3,1))
            warp_f = F.relu(self.to_context[i](warp_f))
            d_flow = self.refine(m_f, warp_f)
            flow = flow + d_flow[:,0:2,:,:]
            d_occlusion = d_flow[:,2:,:,:]
            if i != 0:
                occlusion = occlusion + d_occlusion
            else:
                occlusion = d_occlusion
            out = bilinear_sampler(feature[i], (flow+id_grid).permute(0,2,3,1))
            out = self.to_imgs[i](out*F.sigmoid(occlusion), skip)
            skip = out
            if i != self.num_iter-1:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear',align_corners=True) * 2
                occlusion = F.interpolate(occlusion, scale_factor=2, mode='bilinear',align_corners=True) 


        # warp_f_m = []
        # warp_f_g = []
        # for f,f_g in zip(feature, feature_g):
        #     h,w = f.shape[2:]
        #     if flow.shape[2]!=h or flow.shape[3]!=w:
        #         flow_res = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)
        #     else:
        #         flow_res = flow
        #     warp_f_m.append(F.grid_sample(f, flow_res.permute(0,2,3,1)))
        #     warp_f_g.append(F.grid_sample(f_g, flow_res.permute(0,2,3,1)))
        # flow_res = F.interpolate(flow, size=(256, 256), mode='bilinear', align_corners=True)*4
        warp_img = bilinear_sampler(img_full, (flow+id_grid).permute(0,2,3,1))
        return out, warp_img, F.sigmoid(occlusion)


        # warp_f, dots = self.position_transformer(pos_emb_dri, keys, values)
        # warp_f = rearrange(warp_f, 'b (h w) c -> b c h w', h=h, w=w)
        # warp_src = rearrange(warp_src, 'b (h w) c -> b c h w', h=h, w=w)
        # warp_img = rearrange(warp_img, 'b (h w) c -> b c h w', h=h, w=w)
        # q = rearrange(q, 'b h n c -> b n (h c)')
        # warp_img = torch.cat([warp_img, warp_src, q],dim=-1)
        # warp_img = self.linear(warp_img)
        # # warp_img = rearrange(warp_img, 'b (h w) c -> b c h w', h=h, w=w)
        # # warp_f = warp_f + warp_img
        # warp_f = self.linear2(torch.cat([warp_f,warp_img], dim=-1))
        # warp_f = rearrange(warp_f, 'b (h w) c -> b c h w', h=h, w=w)

        
        # return warp_f, warp_img
        # return warp_f, flow
        # return out, occlusion

class LatentMotionTransformer(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_motions, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="sine-full"):
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        # assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.num_motions = num_motions
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.motion_token = nn.Parameter(torch.randn(1, dim, self.num_motions))
        # self.motion_dicts = nn.Parameter(torch.randn(1, motion_dim, num_motion_codes))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,num_keypoints=num_motions,all_attn=self.all_attn, scale_with_head=True, num_patches=num_patches)

        self.to_motion_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        trunc_normal_(self.motion_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        # print(x.shape)
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        mq,mr = torch.qr(self.motion_token+1e-9)
        motion_tokens = mq.permute(0,2,1).repeat(b,1,1).contiguous()
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((motion_tokens.detach(), x), dim=1)
        elif self.pos_embedding_type == 'none':
            x = torch.cat((motion_tokens.detach(), x), dim=1)
        else:
            x = torch.cat((motion_tokens.detach(), x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_motions)]
        x = self.dropout(x)

        x = self.transformer(x, mask,self.pos_embedding)
        x = self.to_motion_token(x[:, 0:self.num_motions])
        weights = self.mlp_head(x)
        x = x * weights
        return x


class LinearQKVAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 128, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0., to_v=True):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = dim // heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            if to_v:
                self.to_v = nn.Linear(dim, kv_dim, bias = False)
            else:
                self.to_v = None
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        if self.to_v is not None:
            self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self,x,key,value):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if key is None else key.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if key is None else key

        keys = self.to_k(kv_input)
        if self.to_v is not None:
            values = self.to_v(value) if not self.share_kv else keys
        else:
            values = value

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        # merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        # keys, values = map(merge_key_values, (keys, values))

        keys = keys.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        values = values.reshape(b, k, -1, value.shape[-1]//h).transpose(1, 2).expand(-1, h, -1, -1)

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        if self.to_v is not None:
            out = self.to_out(out)
        return out


