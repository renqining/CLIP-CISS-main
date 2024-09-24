# from .per_pixel import PerPixelDistillation
# from .maskformer import MaskFormerDistillation
# from .segkd import SegVitDistillation
from .CLIP_CISSKD import CLIP_CISSDistillation

def build_wrapper(cfg, model, model_old):
    # if cfg.MODEL.MASK_FORMER.PER_PIXEL:
    #     return PerPixelDistillation(cfg, model, model_old)
    # else:
    #     return MaskFormerDistillation(cfg, model, model_old)
    if cfg.MODEL.META_ARCHITECTURE == "CLIP_CISS":
        return CLIP_CISSDistillation(cfg, model, model_old)
    else:
        #return SegVitDistillation(cfg, model, model_old)
        return None