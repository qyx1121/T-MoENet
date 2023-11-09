
from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from util.misc import avg_1d_pool


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
        return_all_features: bool = False, add_mask: bool = False
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self.add_mask = add_mask
        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def forward(self, q, k, v, mask):
        if not self.add_mask:
            mask = torch.ones_like(mask)

        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        
        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        #aff = aff.softmax(dim=-2)
        
        rmask = ~(mask.bool())
        aff = aff.masked_fill(rmask.unsqueeze(1).unsqueeze(-1).to(aff.device), float("-inf"))
        aff = aff.softmax(dim = -2)

        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        add_mask: bool = False
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim, add_mask=add_mask
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)
        self.norm3 = LayerNorm(in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)


    def forward(self, x, y, mask):
        y_norm = self.norm3(y)
        x = x + self.attn(self.norm1(x), y_norm, y_norm, mask)
        x = x + self.mlp(self.norm2(x))

        return x


class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = False,
        mlp_dropout: float = 0.5,
        add_vid_feat: bool = False,
        add_mask: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.add_vid_feat = add_vid_feat

        if add_vid_feat:
            self.decoder_layers = nn.ModuleList(
                [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout, add_mask=add_mask) for _ in range(num_layers)]
            )
            self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))
            self._initialize_weights()

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim) for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, in_features, video_mask):
        N, T, C = in_features.size()
        
        if self.add_vid_feat:
            x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        frame_features = in_features
        for i in range(self.num_layers):
            frame_features = in_features
            feat = in_features
            #feat = frame_features

            feat = feat.permute(0, 2, 1).contiguous() # N * L, C, T
            feat = self.temporal_conv[i](feat)
            feat = feat.view(N, C, T).permute(0, 2, 1,).contiguous() # N, T, C
            frame_features = frame_features + feat
        
            frame_features = frame_features + self.temporal_pos_embed[i].view(1, T, C)
            
            if self.add_vid_feat:
                x = self.decoder_layers[i](x, frame_features, video_mask)
        
        
        if self.add_vid_feat:
            return x
        
        return frame_features


class EVLTransformer(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        decoder_num_layers: int = 2,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 16,
        decoder_mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = False,
        decoder_mlp_dropout: float = 0.5,
        add_video_feat: bool = False,
        output_dim: int = 1536,
        add_mask: bool = False
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_feature_dim = 768
        backbone_spatial_size = (16, 16)

        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
            add_vid_feat = add_video_feat,
            add_mask=add_mask
        )
        self.add_vid_feat = add_video_feat
        if self.add_vid_feat:
            self.norm = nn.LayerNorm(backbone_feature_dim)
            #self.dropout = nn.Dropout(0.5)
            self.proj = nn.Linear(decoder_qkv_dim, output_dim)

    def forward(self, x, video_mask):

        features = x
        x = self.decoder(features, video_mask)
        if self.add_vid_feat:
            x = self.norm(x)
            #x = self.dropout(x)
            x = self.proj(x)

        return x

class TemporalAttention(nn.Module):
    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 8,
        max_frames: int = 40,
        stride: int = 4,
        kernel_size: int = 4,
        add_mask: bool = True,
    ):
        super().__init__()
        
        self.num_layers = 2
        self.kernel_size = kernel_size
        self.stride = stride
        max_frames = (max_frames - self.kernel_size) // self.stride + 1
        
        self.decoder_layers = nn.ModuleList(
                [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, 2.0, 0.5, add_mask=add_mask) for _ in range(self.num_layers)]
            )

        self.temporal_pos_embed = nn.Parameter(torch.zeros([max_frames, in_feature_dim]))
        self.norm = nn.LayerNorm(in_feature_dim)
    
    def forward(self, x, video_mask):
        
        x, video_mask = avg_1d_pool(x, self.kernel_size, self.stride, video_mask, return_mask=True)

        x = x + self.temporal_pos_embed.unsqueeze(0)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, x, video_mask)
        
        return x
    
def recursive_gumbel_softmax(sim, x, video_mask, topk):
    # sim: bs, T
    # x: bs, T, dim

    feats = []
    bs = x.shape[0]
    idxs = torch.zeros(bs, 10)
    v_masks = []

    rmask = ~(video_mask.bool())
    sim = sim.masked_fill(rmask.unsqueeze(1).to(sim.device), float("-inf"))

    for i in range(topk):
        choice = F.gumbel_softmax(sim/0.01, hard=True, dim = -1, tau=0.1).squeeze(1) # bs, T
        idxs[:, i] = torch.argsort(choice, descending=True)[:, 0]
        tmp = torch.sum(choice.unsqueeze(-1) * x, dim = 1, keepdim=True) # bs, dim
        feats.append(tmp)

        mask_tmp = video_mask[torch.arange(bs), idxs[:, i].to(torch.long)]
        v_masks.append(mask_tmp)
        sim = sim - choice.unsqueeze(1)
    
    rank = torch.argsort(idxs, dim = 1)
    
    feats = torch.cat(feats, dim= 1) # bs, 10, dim
    res = [feats[torch.arange(bs), rank[:, i]] for i in range(10)]
    res = torch.stack(res, dim=1)
    
    video_mask = torch.stack(v_masks, dim=1)
    video_mask = [video_mask[torch.arange(bs), rank[:, i]] for i in range(10)]
    video_mask = torch.stack(video_mask, dim = 1)

    return res, video_mask
 