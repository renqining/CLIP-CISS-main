# Copyright (c) Facebook, Inc. and its affiliates.
# from .backbone.swin import D2SwinTransformer
# from .backbone.visiontransformer import visiontransformer
# from .pixel_decoder.fpn import BasePixelDecoder
#from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
# from .meta_arch.mask_former_head import MaskFormerHead
# from .meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead
# from .meta_arch.atm_head import ATMHead
from .backbone.img_encoder import VPTCLIPVisionTransformer, CLIPVisionTransformer
from .backbone.text_encoder import CLIPTextEncoder
from .meta_arch.decode_seg import ATMSingleHeadSeg