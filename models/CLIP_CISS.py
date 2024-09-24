# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from detectron2.config import configurable
# from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom


@META_ARCH_REGISTRY.register()
class CLIP_CISS(nn.Module):
    @configurable
    def __init__(self,
                 pixel_mean,
                 pixel_std,
                 
                 decode_head=None,
                 slide_stride = 341,
                 slide_crop = 512,
                 exclude_key=None,
                 load_text_embedding=None,
                 text_encoder =None,
                 backbone= None,
                 ori_clip = None,
                 num_classes = 150,
                 inc_list = [],
                 step = 0,
                 
                ):
        super(CLIP_CISS, self).__init__()
        self.step = step
        self.num_classes = num_classes
        self.inc_list = inc_list
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.decode_head = decode_head
        # self.pixel_mean = pixel_mean
        # self.pixel_std = pixel_std
        self.logit_scale = self.backbone.logit_scale
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        print('pixel_std:' + str(pixel_std[i] for i in range(3)))
        self.load_text_embedding = load_text_embedding
        self.num_classes = decode_head.num_classes 
        self.ori_clip = ori_clip 
        #self.conv = nn.Conv2d(768,512,1)
        assert self.load_text_embedding != None
        
        if self.ori_clip is not None:
            self._freeze_stages(self.ori_clip)
        
        if self.training:
            if self.text_encoder != None:
                self._freeze_stages(self.text_encoder)
            
           
        else:
            if self.text_encoder != None:
                self.text_encoder.eval()
            self.backbone.eval()
            self.decode_head.eval()
            

    @classmethod
    def from_config(cls, cfg, input_size=None):       
        backbone = build_backbone(cfg)
        decode_head = build_sem_seg_head(cfg, None)
        # text_encoder = CLIPTextEncoder( 
        #                                 context_length = cfg.TEXT_ENCODER.CONTEXT_LENGTH,
        #                                 vocab_size=49408,
        #                                 transformer_width=cfg.TEXT_ENCODER.TRANSFORMER_WIDTH,
        #                                 transformer_heads=cfg.TEXT_ENCODER.TRANSFORMER_HEADS,
        #                                 transformer_layers=cfg.TEXT_ENCODER.TRANSFORMER_LAYERS,
        #                                 embed_dim=1024,
        #                                 out_dim=cfg.TEXT_ENCODER.EMBED_DIM,
        #                                 pretrained=cfg.MODEL.PRETRAINED_TEXT
        #                                )
        ret = {
            "backbone": backbone,
            "decode_head": decode_head,
            # "text_encoder": text_encoder,
            "pixel_mean":cfg.MODEL.PIXEL_MEAN,
            "pixel_std":cfg.MODEL.PIXEL_STD,
            "exclude_key":cfg.MODEL.EXCLUDE_KEY,
            "load_text_embedding":cfg.MODEL.LOAD_TEXT_EMBEDDING,
        }
        if hasattr(cfg, "CONT"):
            ret["num_classes"] = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
            inc_list = [cfg.CONT.BASE_CLS] if cfg.CONT.TASK == 0 else [cfg.CONT.BASE_CLS]+[cfg.CONT.INC_CLS]*cfg.CONT.TASK
            ret["inc_list"] = inc_list
            ret["step"] = cfg.CONT.TASK
            if cfg.CONT.ORI_CLIP == True:
                ori_clip = build_backbone(cfg)
                ret["ori_clip"] = ori_clip
        return ret

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False
    
    def text_embedding(self, texts, img):  #not used
        text_embeddings = self.text_encoder(texts.to(img.device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings
    
    @property
    def device(self):
        return self.pixel_mean.device
      
    def forward(self, batched_inputs, old = False):
        if self.training or old == True:
            
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images)

            visual_feat = self.backbone(images.tensor)
            if self.ori_clip is not None:
                visual_feat_old = self.ori_clip(images.tensor)
               

            if self.load_text_embedding:
                text_feat = np.load(self.load_text_embedding).astype(np.float32)
                text_feat = torch.from_numpy(text_feat).to(images.tensor.device)
            else:
                if not self.multi_prompts:
                    text_feat = self.text_embedding(self.texts, images.tensor)
                else:
                    assert AttributeError("preparing the multi embeddings")
            text_feat = text_feat[:self.num_classes,:]
            feat = []
            feat.append(visual_feat)
            feat.append(text_feat)
            
            outputs = self.decode_head(feat) # test train the same
            # inputs = visual_feat[0]
            # aux_feat = []
            # for i in range(len(inputs)):
            #     aux_feat.append(self.conv(inputs[i]))
            # outputs["features_conv"] = aux_feat
            outputs['last_map'] = feat[0][2]
            if self.ori_clip is not None:
                # outputs['ori_clip'] = visual_feat_old[1]
                # outputs['ori_clip_features'] = visual_feat_old[0]
                outputs['ori_clip_map'] = visual_feat_old[2]
                outputs['ori_clip_cls'] = visual_feat_old[1]
            del outputs['features']
            return { "outputs": outputs, "shape": images.tensor.shape[-2:]}
        
        else:
            h_stride=w_stride =self.slide_stride
            h_crop=w_crop = self.slide_crop
            
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images)
            
            images = images.tensor
            
            _,_,h_img, w_img = images.shape
            # num_classes = self.decode_head.num_classes #no bg
            num_classes = self.decode_head.num_classes 
            #num_classes = self.decode_head.num_classes 
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            
            preds = images.new_zeros((1, num_classes, h_img, w_img))
            count_mat = images.new_zeros((1, 1, h_img, w_img))
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = images[:, :, y1:y2, x1:x2]
                    
                    visual_feat = self.backbone(crop_img)

                    if self.load_text_embedding:
                        text_feat = np.load(self.load_text_embedding).astype(np.float32)
                        text_feat = torch.from_numpy(text_feat).to(images.device)
                    else:
                        if not self.multi_prompts:
                            text_feat = self.text_embedding(self.texts, images)
                        else:
                            assert AttributeError("preparing the multi embeddings")
                    text_feat = text_feat[:self.num_classes,:]
                    feat = []
                    feat.append(visual_feat)
                    feat.append(text_feat)
                    
                    outputs = self.decode_head(feat) # test train the same
                    
                    crop_seg_logit = outputs['pred']
                    
                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            
            preds = preds / count_mat
            #preds = F.softmax(preds, dim=1)
            return [{'sem_seg': preds[0]}] 
            
      
    @staticmethod
    def prepare_targets(targets, shape):
        h_pad, w_pad = shape
        new_targets = []
        
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                        device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
            for ii in range(len(targets_per_image.gt_classes)):
                assert targets_per_image.gt_classes[ii] > 0
        
        return new_targets

    @staticmethod
    def prepare_semantic_train(outputs, targets, mask_bg=True):
        logits, mask = outputs["pred_logits"], outputs["pred_masks"]
        mask = mask.sigmoid()
        if mask_bg:
            semseg = torch.einsum("bqc,bqhw->bchw", logits, mask)
            semseg = semseg[:, :-1]  # Exclude no class since we have Bkg class
        else:
            raise NotImplementedError
        return semseg
