VOC12_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'This is a photo of a {}.',
    'This is a photo of a small {}.',
    'This is a photo of a medium {}.',
    'This is a photo of a large {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'This is a {} in the scene.',
    'This is the {} in the scene.',
    'This is one {} in the scene.',
    ]


import torch
import numpy as np
import clip
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)

## single template
def single_templete(save_path, class_names, model):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).cuda()
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings





## VOC12:
save_path='/root/CLIP_CISS-main/configs/voc/voc_single.npy'
text_embeddings = single_templete(save_path, VOC12_classes, model)
