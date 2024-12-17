
from models.CLIP_CISS import CLIP_CISS

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from models.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
# import einops
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def unbiased_knowledge_distillation_loss(inputs, targets, reweight=False, gamma=2., temperature=1.):
    '''
    inputs b, c, N   targets b, c, n
    b,151,100  b,101,100
    '''
    targets = targets * temperature

    den = torch.logsumexp(inputs, dim=1)  #   b,100
    outputs_no_bgk = inputs[:, :targets.shape[1]-1] - den.unsqueeze(dim=1)  # b,100,100
    outputs_bkg = torch.logsumexp(inputs[:, targets.shape[1]-1:], dim=1) - den  # b,150
    labels = torch.softmax(targets, dim=1)  # b,101,100
    labels_soft = torch.log_softmax(targets, dim=1) #b,101,100

    loss = labels[:, -1] * (labels_soft[:, -1] - outputs_bkg) + \
           (labels[:, :-1] * (labels_soft[:, :-1] - outputs_no_bgk)).sum(dim=1)  # B, Q  b,100
    # Re-weight no-cls queries as in classificaton
    if reweight:
        loss = ((1-labels[:, -1]) ** gamma * loss).sum() / ((1-labels[:, -1]) ** gamma).sum()
    else:
        loss = loss.mean()
    return loss

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

class SetCriterion(nn.Module):
    
    def __init__(self, num_classes, weight_dict, losses, old_classes,new_classes,step,logit_scale=0,eos_coef=0.1):
        
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.kd_reweight = True
        self.alpha = 1.0
        self.new_classes = new_classes
        self.old_classes = old_classes
        self.step = step
        self.logit_scale = logit_scale
        
    
    
    def compute_class_features_mean(self, features, targets):
        # 初始化一个空字典来存储类别和它们的特征均值
        class_features_mean = {}
        
        batch_size, depth, height, width = features.shape
        _, H, W = targets[0]["masks"].shape
        features = F.interpolate(features ,(H,W) ,mode = 'bilinear',align_corners=False)
        
        # 将特征图扁平化
        features = features.view(batch_size, depth, -1)  # 形状变为 (b, d, h*w)
        
        
        for i in range(batch_size):
            # 获取当前图片的所有标签和掩码
            img_labels = targets[i]["labels"]
            img_masks = targets[i]["masks"]
            
            # 将掩码扩展为与特征图相同的形状，并扁平化
            img_masks = img_masks.view(-1, H * W)  # 形状变为 (n, h*w)
            # 计算掩码区域的数量
            mask_counts = img_masks.sum(dim=1, keepdim=True)  # 形状变为 (n, 1)
            mask_counts = torch.where(mask_counts == 0, torch.ones_like(mask_counts), mask_counts)
            img_masks = img_masks.float()/mask_counts
            # 使用逐元素乘法应用掩码
            # 构建掩码和特征图的形状为 (n, h*w) 和 (h*w, d)，以便进行逐元素乘法
            masked_features_sum = torch.einsum('ni,id->nd', img_masks, features[i].t())  # 形状变为 (n, d)
            
            # 计算掩码区域的特征均值
            class_feature_mean = masked_features_sum
            
            for j, label in enumerate(img_labels):
                label_item = label.item()
                if label_item not in class_features_mean:
                    class_features_mean[label_item] = []
                class_features_mean[label_item].append(class_feature_mean[j])
        
        # 计算最终的类别特征均值
        for key in class_features_mean:
            class_features_mean[key] = torch.mean(torch.stack(class_features_mean[key]), dim=0)
        
        return class_features_mean
    
    
    def loss_gpd(self, outputs, targets, indices, num_masks, outputs_old=None,bg=None,targets_with_pseudo=None):
        
        losses = {}
        if self.step == 0 and self.weight_dict['loss_gpd'] > 0.:
            txt_all = outputs["txt_embedding"] # c,d
            c,d = txt_all.shape
            # txt_all = txt_all/txt_all.norm(dim = -1, keepdim = True)
            #bs,dim,h,w = outputs["features"][0].shape  #feature shape
            _, H, W  = targets[0]['masks'].shape
            
            features = outputs["last_map"]#b,dim,h,w
            features_old = outputs["ori_clip_map"]
            
            protos = self.compute_class_features_mean(features, targets)
            protos_old = self.compute_class_features_mean(features_old, targets)
            del features
            del features_old
            protos = [proto for proto in protos.values()]
            protos_old = [proto for proto in protos_old.values()]
            if (len(protos) == 0) or (len(protos_old) == 0):
                losses['loss_gpd'] = torch.tensor(0.).to(torch.float16).to(targets[0]['masks'].device)
                return losses
                
            protos = torch.stack(protos) 
            protos_old = torch.stack(protos_old)
            protos = protos / protos.norm(dim=1 , keepdim=True)
            protos_old = protos_old / protos_old.norm(dim=1 , keepdim=True)
            src_logits = protos @ txt_all.t()  #n,c
            tar_logits = protos_old @ txt_all.t()
            
            
            # losses['loss_gpd'] = torch.sum((src_logits - tar_logits)*(src_logits - tar_logits))
            
            cls = outputs["cls_token"]
            cls_old = outputs["ori_clip_cls"]
            p = F.softmax(cls@txt_all.t(), dim=1)
            p_old = F.softmax(cls_old@txt_all.t(), dim=1)
            log_q = torch.log(p + 1e-8)
            l2 = F.kl_div(log_q, p_old, reduce=False).sum(dim=1).mean()
            
            losses['loss_gpd'] = torch.sum((src_logits - tar_logits)*(src_logits - tar_logits))+l2
            return losses
        
        if self.weight_dict['loss_gpd'] > 0.:
            assert targets_with_pseudo is not None
            targets = targets_with_pseudo
            # cls_tokens = outputs["cls_token"] #b,d
            txt_all = outputs["txt_embedding"] # c,d
            c,d = txt_all.shape
            # txt_all = txt_all/txt_all.norm(dim = -1, keepdim = True)
            #bs,dim,h,w = outputs["features"][0].shape  #feature shape
            _, H, W  = targets[0]['masks'].shape
            
            features = outputs["last_map"]
            features_old = outputs_old["last_map"]
            protos = self.compute_class_features_mean(features, targets)
            protos_old = self.compute_class_features_mean(features_old, targets)
            
            del features
            del features_old
            protos = [proto for proto in protos.values()]
            protos_old = [proto for proto in protos_old.values()]
            if (len(protos) == 0) or (len(protos_old) == 0):
                losses['loss_gpd'] = torch.tensor(0.).to(torch.float16).to(targets[0]['masks'].device)
                return losses
                
            protos = torch.stack(protos) 
            protos_old = torch.stack(protos_old)
            protos = protos / protos.norm(dim=1 , keepdim=True)
            protos_old = protos_old / protos_old.norm(dim=1 , keepdim=True)
            src_logits = protos @ txt_all.t()  #n,c
            tar_logits = protos_old @ txt_all.t()
            
            #losses['loss_gpd'] = torch.sum((src_logits - tar_logits)*(src_logits - tar_logits))
            cls = outputs["cls_token"]
            cls_old = outputs_old["cls_token"]
            p = F.softmax(cls@txt_all.t(), dim=1)
            p_old = F.softmax(cls_old@txt_all.t(), dim=1)
            log_q = torch.log(p + 1e-8)
            l2 = F.kl_div(log_q, p_old, reduce=False).sum(dim=1).mean()
            
            losses['loss_gpd'] = torch.sum((src_logits - tar_logits)*(src_logits - tar_logits))+l2
            return losses
        return losses
    
    
    def loss_q_kd(self, outputs, targets, indices, num_masks, outputs_old=None,bg=None,targets_with_pseudo=None):
        losses = {}
        if self.weight_dict['loss_q_kd'] > 0.:
            target = outputs_old["qs"]  #3,b,n,dim
            nl,bs,nq,dim = target.shape
            target = target.reshape([nl*bs*nq, -1])
            
            source = outputs["qs"][:,:,:nq,:].reshape([nl*bs*nq,-1])
            loss = torch.sum((source - target)**2, dim=-1)
            losses["loss_q_kd"] = loss.mean()
        return losses
    
    def loss_q_kd_v2(self, outputs, targets, indices, num_masks, outputs_old=None,bg=None,targets_with_pseudo=None):
        losses = {}
        if self.weight_dict['loss_q_kd'] > 0.:
            target = outputs_old["qs"]  #3,b,n,dim
            nl,bs,nq,dim = target.shape
            target = target.reshape([nl*bs*nq, -1])
            
            source = outputs["qs"][:,:,:nq,:].reshape([nl*bs*nq,-1])
            loss = torch.pow((source - target), 2).sum(dim=1)
            losses["loss_q_kd"] = loss.mean()
            
            target2 = target[0].reshape([bs*nq, -1]) * target[-1].reshape([bs*nq, -1])
            source2 = outputs["qs"][0,:,:nq,:].reshape([bs*nq, -1]) * outputs["qs"][-1,:,:nq,:].reshape([bs*nq, -1])
            loss2 = torch.pow((source2 - target2), 2).sum(dim=1).mean()
            losses["loss_q_kd"] += loss2*5.0
            
        return losses
    
    
    def loss_labels(self, outputs, targets, indices, num_masks, outputs_old=None,bg=None,targets_with_pseudo=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(
            src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device
        )  #b,c
        target_classes[idx] = 0
        src_logits_ = src_logits.transpose(1, 2)
        if self.old_classes != -1:
            src_logits_ = src_logits_[:,:,self.old_classes:]
            target_classes = target_classes[:,self.old_classes:]
        
        loss_ce = F.cross_entropy(src_logits_, #b,c+1,c  b,2,c
                                  target_classes, #b,c
                                  # self.empty_weight
                                  )
        losses = {"loss_ce": loss_ce}
        
        # if outputs_old is not None:
        #     tar_logits = outputs_old["pred_logits"].float()
        #     b,c,_ = tar_logits.shape
        #     tar_logits = tar_logits.reshape(b*c,-1)
        #     tar_logits = F.softmax(tar_logits, dim=-1)
        #     src_logits = src_logits[:,:c,:].reshape(b*c,-1)
        #     loss_kd = F.cross_entropy(src_logits, tar_logits.long() ,reduction = 'mean')
        #     # loss_kd = torch.pow((src_logits - tar_logits), 2).sum(dim=1)
        #     losses["loss_ce_kd"] = loss_kd
        
        return losses
    
    
    def loss_masks(self, outputs, targets, indices, num_masks, outputs_old,bg,targets_with_pseudo=None):
        # indices?
        assert "pred_masks" in outputs
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        if src_masks.dim() != 4:
            return {"no_loss": 0}
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        
        if self.weight_dict['loss_mask_kd'] > 0. and outputs_old is not None:
            old_masks = outputs_old["pred_masks"].detach()
            new_masks = outputs["pred_masks"][:,:old_masks.shape[1]]
            
            labels = old_masks.sigmoid()
            losses['loss_mask_kd'] = F.binary_cross_entropy_with_logits(new_masks, labels)

        
        del src_masks
        del target_masks
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, outputs_old,bg,targets_with_pseudo):
        # loss_map = {"masks": self.loss_masks}
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks,
                    
                    "gpd_kd":self.loss_gpd,
                    "q_kd": self.loss_q_kd,
                    
                    }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, outputs_old,bg,targets_with_pseudo)

    def forward(self, outputs, targets, outputs_old=None, bg = None, targets_with_pseudo = None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        labels = [x['labels'] for x in targets]
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t])
        indices = indices_new
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs["pred_masks"].device#next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:  #labels masks
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks,outputs_old,bg,None))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # use the indices as the last stage
                for loss in ["masks"]:
                #for loss in self.losses:
                    if outputs_old is not None:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, outputs_old["aux_outputs"][i],None,None)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, None,None,None)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if self.step == 0:
            loss = "gpd_kd"
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks,outputs_old,None,None))
        
        if self.step > 0:
            for loss in [
                         "gpd_kd",
                         "q_kd"]:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks,outputs_old,None,targets_with_pseudo))
        
        return losses

class CLIP_CISSDistillation:
    def __init__(self, cfg, model, model_old):
        
        self.cfg = cfg
        self.model = model
        self.model_old = model_old
        
        self.step = cfg.CONT.TASK
        try:
            self.logit_scale = self.model.logit_scale 
        except:
            self.logit_scale = self.model.module.logit_scale
        self.classes = [cfg.CONT.BASE_CLS] + cfg.CONT.TASK * [cfg.CONT.INC_CLS]
        self.old_classes = cfg.CONT.BASE_CLS + (cfg.CONT.TASK-1) * cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else -1
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else cfg.CONT.BASE_CLS
        self.num_classes = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        self.use_bg = False
        # deep_supervision = cfg.MODEL.DEEP_SUPERVISION
        deep_supervision = True
        # loss weights
        cx = cfg.MODEL.SEM_SEG_HEAD
        
        class_weight = cx.LOSS_WEIGHT
        dice_weight = cx.DICE_WEIGHT
        mask_weight = cx.MASK_WEIGHT

        self.kd_weight = cfg.CONT.DIST.KD_WEIGHT
        
        self.pseudo_thr = cfg.CONT.DIST.PSEUDO_THRESHOLD
        
        self.mask_kd_weight = cfg.CONT.DIST.MASK_KD
        
        self.gpd_weight = cfg.CONT.DIST.GPD
        self.q_kd_weight = cfg.CONT.DIST.Q_KD
        weight_dict = {"loss_ce": class_weight,
                       "loss_mask": mask_weight,
                       "loss_dice": dice_weight,
                       
                       
                       "loss_ce_kd": self.kd_weight,
                       "loss_mask_kd": self.mask_kd_weight,
                       }
        
        if deep_supervision:
            dec_layers = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        weight_kd_dict = {
                        "loss_gpd":self.gpd_weight,
                        "loss_q_kd": self.q_kd_weight,
                            }
        weight_dict.update(weight_kd_dict)
        
        
        losses = ["labels","masks"]
        #losses = ["masks"]
        self.criterion = SetCriterion(
                self.num_classes,
                weight_dict=weight_dict,
                losses = losses,
                old_classes = self.old_classes,
                new_classes = self.new_classes,
                step = self.step,
                logit_scale = self.logit_scale
            )
        
        self.criterion.to(self.device)
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self):
        return self.model.training

    @property
    def device(self):
        return self.model.device
    
    def make_pseudolabels2(self, out, data, targets):
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQx2, BxQxHxW
        n = logits.shape[1]
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )
        
        # mask_cls = F.softmax(logits, dim=-1)[..., 0]# b,c
        # mask = mask.sigmoid()
        # semseg = torch.einsum("bq,bqhw->bqhw", mask_cls, mask)
        
        for i in range(logits.shape[0]):  # iterate on batch size
            scores, labels = F.softmax(logits[i], dim=-1).max(-1)# q   0-100 100 for bg
            mask_pred = mask[i].sigmoid() #q h w
            #mask_pred = semseg[i]
            
            keep = labels.ne(1) & (scores > 0.5)
            cur_scores = scores[keep]
            cur_classes = torch.tensor(range(n))[keep].to(cur_scores.device)
            cur_masks = mask_pred[keep]
            cur_masks_bin = mask_pred[keep].clone()

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks #n,h,w
            cur_prob_masks = F.softmax(cur_prob_masks, dim=0) ## add
            tar = targets[i]
            gt_pixels = tar['masks'].sum(dim=0).bool()  # H,W
            keep2 = torch.zeros(len(cur_masks)).bool()

            if cur_masks.shape[0] > 0:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)  # REMOVE GT 
                cur_mask_ids[gt_pixels] = -1

                for k in range(cur_classes.shape[0]):
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= self.pseudo_thr).sum().item()
                    x_mask = (cur_mask_ids == k) & (cur_masks[k] >= self.pseudo_thr)

                    if mask_area > 0 and original_area > 0 and x_mask.sum().item() > 0:
                        if mask_area / original_area > 0.5:
                            keep2[k] = 1
                            cur_masks_bin[k] = x_mask

            if keep2.sum() > 0:
                pseudo_lab = cur_classes[keep2]
                pseudo_mask = cur_masks_bin[keep2].bool()

                tar['masks'] = torch.cat((tar['masks'], pseudo_mask), dim=0)
                tar['labels'] = torch.cat((tar['labels'], pseudo_lab), dim=0)

        return targets

    def make_pseudolabels3(self, out, data, targets):
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQx2, BxQxHxW
        n = logits.shape[1]
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )
        
        # mask_cls = F.softmax(logits, dim=-1)[..., 0]# b,c
        # mask = mask.sigmoid()
        # semseg = torch.einsum("bq,bqhw->bqhw", mask_cls, mask)
        
        for i in range(logits.shape[0]):  # iterate on batch size
            scores, labels = F.softmax(logits[i], dim=-1).max(-1)# q   0-100 100 for bg
            mask_pred = mask[i].sigmoid() #q h w
            #mask_pred = semseg[i]
            
            keep = labels.ne(1) & (scores > 0.5)
            cur_scores = scores[keep]
            cur_classes = torch.tensor(range(n))[keep].to(cur_scores.device)
            cur_masks = mask_pred[keep]
            cur_masks_bin = mask_pred[keep].clone()

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks #n,h,w
            cur_prob_masks = F.softmax(cur_prob_masks, dim=0) ## add
            tar = targets[i]
            gt_pixels = tar['masks'].sum(dim=0).bool()  # H,W
            keep2 = torch.zeros(len(cur_masks)).bool()

            if cur_masks.shape[0] > 0:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)  # REMOVE GT 
                cur_mask_ids[gt_pixels] = -1

                for k in range(cur_classes.shape[0]):
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_prob_masks[k] >= self.pseudo_thr).sum().item()
                    x_mask = (cur_mask_ids == k) & (cur_prob_masks[k] >= self.pseudo_thr)

                    if mask_area > 0 and original_area > 0 and x_mask.sum().item() > 0:
                        if mask_area / original_area > 0.5:
                            keep2[k] = 1
                            cur_masks_bin[k] = x_mask

            if keep2.sum() > 0:
                pseudo_lab = cur_classes[keep2]
                pseudo_mask = cur_masks_bin[keep2].bool()

                tar['masks'] = torch.cat((tar['masks'], pseudo_mask), dim=0)
                tar['labels'] = torch.cat((tar['labels'], pseudo_lab), dim=0)

        return targets
    
    def __call__(self, data):
        model_out = self.model(data)
        outputs = model_out['outputs']
        targets_with_pseudo = None
        if self.cfg.CONT.TASK == 0:
            model_out_old = None
            outputs_old = None
        else:
            model_out_old = self.model_old(data, old = True) 
            outputs_old = model_out_old['outputs'] 

        # prepare targets...
        if "instances" in data[0]:
            gt_instances = [x["instances"].to(self.device) for x in data]
            bg = [x["bg"].to(self.device) for x in data]
            targets = CLIP_CISS.prepare_targets(gt_instances, model_out['shape'])
            targets_ = CLIP_CISS.prepare_targets(gt_instances, model_out['shape'])
            # Labels assume that background is class 0, remove it.
            if not self.use_bg:
                for tar in targets:
                    tar['labels'] -= 1
                    for ii in range(len(tar['labels'])):
                        assert tar['labels'][ii] >= 0
                        
            # Pseudo-labeling algorithm
            if self.cfg.CONT.TASK > 0 and self.pseudo_thr > 0.:
                targets = self.make_pseudolabels2(model_out_old, data, targets)
            
            if self.cfg.CONT.TASK > 0:
                targets_with_pseudo = self.make_pseudolabels2(model_out_old, data, targets)
                
        else:
            targets = None

        losses = self.criterion(outputs, targets_, outputs_old, None, targets_with_pseudo)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                print('remove this loss if not specified in `weight_dict`')
                losses.pop(k)

        return losses
