# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN



def add_CLIP_CISS_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    cfg.MODEL.WEIGHTS = " "
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part ofFthe crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.HEAD_MULTIPLIER = 1.0
    
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    
    
    cfg.MODEL.BACKBONE.NAME = "VPTCLIPVisionTransformer"
    cfg.MODEL.BACKBONE.INPUT_RESOLUTION=512
    cfg.MODEL.BACKBONE.PATCH_SIZE= 16
    cfg.MODEL.BACKBONE.WIDTH= 768
    cfg.MODEL.BACKBONE.OUTPUT_DIM= 512
    cfg.MODEL.BACKBONE.GET_EMBEDDINGS= True
    cfg.MODEL.BACKBONE.DROP_PATH_RATE= 0.1
    cfg.MODEL.BACKBONE.LAYERS= 12
    cfg.MODEL.BACKBONE.OUT_INDICES= [11]
    cfg.MODEL.BACKBONE.NUM_TOKENS= 10
    cfg.MODEL.BACKBONE.PROMPT_DIM= 768
    cfg.MODEL.BACKBONE.TOTAL_D_LAYER=11
    
    
    
    cfg.MODEL.PRETRAINED = 'ViT-B-16.pt'
    cfg.MODEL.PRETRAINED_TEXT = 'ViT-B-16.pt'
    cfg.MODEL.CONTEXT_LENGTH = 77
    cfg.MODEL.EXCLUDE_KEY = 'prompt'
    cfg.MODEL.LOAD_TEXT_EMBEDDING = 'configs/_base_/datasets/text_embedding/voc12_single.npy'
    
    
    
    cfg.MODEL.SEM_SEG_HEAD.NAME= 'ATMSingleHeadSeg'
    cfg.MODEL.SEM_SEG_HEAD.IMG_SIZE= 512
    cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS= 512
    cfg.MODEL.SEM_SEG_HEAD.CHANNELS= 512
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS= 3
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES= 150
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS= 8
    cfg.MODEL.SEM_SEG_HEAD.USE_STAGES= 1
    cfg.MODEL.SEM_SEG_HEAD.USE_PROJ=True
    cfg.MODEL.SEM_SEG_HEAD.EMBED_DIMS = 512
    
    
    
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT= 1.0
    cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT= 20.0
    cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT= 1.0
    cfg.MODEL.SEM_SEG_HEAD.DEC_LAYERS= 3
    
    
    cfg.MODEL.TEXT_ENCODER = CN()
    cfg.MODEL.TEXT_ENCODER.NAME = 'CLIPTextEncoder'
    cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
    cfg.MODEL.TEXT_ENCODER.EMBED_DIM = 512
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_WIDTH = 512
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_HEADS = 8
    cfg.MODEL.TEXT_ENCODER.TRANSFORMER_LAYERS = 12
