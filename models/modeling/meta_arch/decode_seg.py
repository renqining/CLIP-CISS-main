#5
from ast import Gt
import numpy as np
# from mmcv.cnn import ConvModule
# from mmseg.ops import Upsample, resize

# from mmseg.models.builder import HEADS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
# from mmseg.models.losses import accuracy
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import configurable
# from .utils import positional_encoding

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        # attns = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            # attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn
    
class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # memory2 = self.self_attn(memory, memory, memory, attn_mask=None,
        #                       key_padding_mask=None)[0]
        # memory = memory + self.dropout1(memory2)
        # memory = self.norm1(memory)
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  #nn.Multiheadattention
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.proj_head = nn.Linear(num_heads, num_heads)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() 
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  #attn shape [bs, num_head, num_cls, num_patch]
        attn_save = attn.clone()
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # v0
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads
        # V1,2
        #return x.transpose(0, 1), attn_save

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@SEM_SEG_HEADS_REGISTRY.register()
class ATMSingleHeadSeg(nn.Module):
    @configurable
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            
            num_classes = 150,
            inc_list = [],
            step = 0,
                
    ):
        super(ATMSingleHeadSeg, self).__init__()

        self.image_size = img_size
        self.use_stages = use_stages
        self.in_channels = in_channels
        
        self.step = step
        self.num_classes = num_classes
        self.inc_list = inc_list
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.class_embed = nn.Linear(3*dim, 2, bias=False)
        # self.class_embed = nn. Sequential(
        #     nn.Linear(3*dim, dim // 4, bias=False),
        #     # nn.ReLU(inplace = True),
        #     nn.Linear(dim // 4, 2, bias=False),
        #     # nn.ReLU(inplace =True)
        # )   
        # self.class_embed = nn. Sequential(
        #     nn.Linear(dim, dim // 2),
        #     # nn.ReLU(inplace = True),
        #     nn.Linear(dim // 2, 2),
        #     # nn.ReLU(inplace =True)
        # )   
        self.q_proj = nn.Linear(dim * 2, dim)
        # self.conv = nn.Conv2d(768,512,1)
        # self.thresh_bg = nn.Parameter(torch.tensor(0.0))
        if self.step == 0:
            self.init_weights()  #whether to use???

    @classmethod
    def from_config(cls, cfg, input_size=None):
        ret = {
                "img_size": cfg.MODEL.SEM_SEG_HEAD.IMG_SIZE,
                "in_channels":cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS,
                
                "embed_dims":cfg.MODEL.SEM_SEG_HEAD.EMBED_DIMS,
                "num_layers":cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS,
                "num_heads":cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS,
                "use_stages":cfg.MODEL.SEM_SEG_HEAD.USE_STAGES,
                "use_proj":cfg.MODEL.SEM_SEG_HEAD.USE_PROJ,
                
        }
        if hasattr(cfg, "CONT"):
            ret["num_classes"] = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
            inc_list = [cfg.CONT.BASE_CLS] if cfg.CONT.TASK == 0 else [cfg.CONT.BASE_CLS]+[cfg.CONT.INC_CLS]*cfg.CONT.TASK
            ret["inc_list"] = inc_list
            ret["step"] = cfg.CONT.TASK
        return ret

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
                
    def forward(self, inputs_both):# v1
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1] #bs, dim
        text_token = inputs_both[1] #n,dim
        
        t0 = text_token
        bs = cls_token.shape[0]
        t0 = t0.expand(bs, -1, -1)
        #v1
        #qt = torch.einsum("bd,bcd->bcd", cls_token, t0) #b,c,d
        #v2
        # qt = torch.einsum("bd,bcd->bc", cls_token, t0) #b,c
        # qt = F.softmax(qt, dim=-1).unsqueeze(-1) #b,c,1
        
        q = self.get_qs(text_token, cls_token)
        
        q0 = q #b,c,d
        q = q.transpose(0,1)
        # q.shape  num_cls, bs, dim
        
        out = {}
        out["features"] = inputs
        # out["features_conv"] = []
        # for i in range(len(inputs)):
        #     out["features_conv"].append(self.conv(inputs[i]))
        out["txt_embedding"] = text_token
        out["cls_token"] = cls_token
        
        inputs = list(inputs)
        inputs.reverse()
        
        attns = []
        maps_size = []
       
        qs = []
        outputs_class = []
        for idx, (x_, proj_, norm_, decoder_) in enumerate(zip(inputs, self.input_proj, self.proj_norm, self.decoder)):
            x_ = self.d4_to_d3(x_)
            x_ = norm_(proj_(x_))

            q, attn = decoder_(q, x_.transpose(0, 1))  #attn [bs, num_head, num_cls, num_patch]
            #outputs_class.append(self.class_embed(q.transpose(0, 1) - q0))
            #outputs_class.append(self.class_embed(torch.cat((q.transpose(0, 1) - q0, q.transpose(0, 1)*q0),dim = -1)))
            outputs_class.append(self.class_embed(torch.cat((q.transpose(0, 1) - q0, q.transpose(0, 1)*q0, torch.einsum("bd,bcd->bcd", cls_token, t0)),dim = -1)))
            #outputs_class.append(self.class_embed(torch.cat((q.transpose(0, 1) - q0, torch.einsum("bd,bcd->bcd", cls_token, t0)),dim = -1)))
            
            
            attn = attn.transpose(-1, -2)
            attn = self.d3_to_d4(attn)
            qs.append(q.transpose(0,1))
            maps_size.append(attn.size()[-2:])
            
            attns.append(attn)
        qs = torch.stack(qs, dim=0) #3 b num_cls dim
        
        out["pred_logits"] = outputs_class[-1]
        outputs_class = torch.stack(outputs_class, dim=0)
        out["qs"] = qs #3 b 150 384
        size = maps_size[-1]
        
        outputs_seg_masks = []
        for i_attn, attn in enumerate(attns):
            if i_attn == 0:
                outputs_seg_masks.append(F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
            else:
                outputs_seg_masks.append(outputs_seg_masks[i_attn - 1] +
                                         F.interpolate(attn, size=size, mode='bilinear', align_corners=False))
        
        out["pred_masks"] = F.interpolate(outputs_seg_masks[-1],
                                          size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)  
        
        # for inference
        out["pred"] = self.semantic_inference(out["pred_logits"], out["pred_masks"])
        
        outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
        out["aux_outputs"] = self._set_aux_loss(
            outputs_seg_masks
        )
        
        return out
    
    
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)
        #mask_cls = F.softmax(mask_cls, dim=-1)[..., 0]# b,c
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bq,bqhw->bqhw", mask_cls[...,0], mask_pred)
        
        return semseg
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]
    
    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_seg_masks):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [
    #         {"pred_logits": a, "pred_masks": b}
    #         for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
    #     ]
        
        
    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        # q = [q.cls, q]
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        #return q
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.cat((q1, q), dim=-1)
        return self.q_proj(q_)
    
    