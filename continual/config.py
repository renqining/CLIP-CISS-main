from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.WANDB = True

    cfg.CONT = CN()
    cfg.CONT.BASE_CLS = 15
    cfg.CONT.INC_CLS = 5
    cfg.CONT.ORDER = list(range(1, 21))
    # cfg.CONT.ORDER_NAME = None
    cfg.CONT.TASK = 0
    cfg.CONT.WEIGHTS = None
    cfg.CONT.MODE = "overlap"  # Choices "overlap", "disjoint", "sequential"
    # cfg.CONT.INC_QUERY = False
    # cfg.CONT.COSINE = False
    # cfg.CONT.USE_BIAS = True
    cfg.CONT.WA_STEP = 0

    cfg.CONT.DIST = CN()
    
    cfg.CONT.DIST.PSEUDO_THRESHOLD = 0.
    cfg.CONT.DIST.KD_WEIGHT = 0.
    cfg.CONT.DIST.MASK_KD = 0.
    cfg.CONT.DIST.CLIP_KD = 0.
    cfg.CONT.DIST.GPD = 0.5
    cfg.CONT.DIST.Q_KD = 0.2
    
    
    cfg.CONT.ORI_CLIP = False #keep original clip model
