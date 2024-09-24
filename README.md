# CLIP-Guided Class Incremental Semantic Segmentation with Generalization-Preserving Knowledge Distillation
## Abstract
Deep neural networks achieve outstanding performance on specific tasks after training. However, directly tuning these models to learn new tasks often leads to the forgetting of previous knowledge, a phenomenon known as catastrophic forgetting. This paper focuses on the Class Incremental Semantic Segmentation (CISS) task, which aims to mitigate forgetting in segmentation models. Despite the significant progress of recent methods, effective knowledge transfer across sequential tasks remains underexplored. Moreover, these methods still struggle with the semantic shift issue. Based on these observations, we introduce a novel transformer-based framework for the CISS task, designed to acquire more task-general knowledge by leveraging the well-aligned text-image feature space of CLIP.Specifically, segmentation is performed by exploiting the matching process between patch-level image features and text features, which facilitates knowledge sharing and transfer across tasks. To address semantic shift, Class-Agnostic Confidence Prediction (CACP) head is proposed and integrated into the framework, which verifies the existence of different classes independently. This prevents the semantics of a foreground class from being interfered by the ever-changing `background' class. Additionally, to maintain the ability to segment previous classes while generalizing to future ones, we incorporate Generalization-Preserving Distillation (GPD) loss and Query-based Distillation (QD) loss into our framework. We evaluate the proposed framework's effectiveness using the VOC2012 and ADE20K datasets, demonstrating superior performance compared to previous state-of-the-art methods.  Specifically, our method achieves mIoU improvements of 1.0$\%$ and 1.6$\%$ in the most challenging ADE 100-5 (11 steps) and VOC 10-1 (11 steps) settings, respectively.

## Environment
- Build detectron2 from source 
See [installation instructions](INSTALL.md).
- Install pytorch

 `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`

- Other libraries
- numpy == 1.24.3
- wandb == 0.5.12
- timm == 0.9.7
- opencv-python == 4.8.1.78
- continuum == 1.2.7


## Getting Started

### Prepare the datasets
See [Preparing Datasets for Mask2Former](datasets/README.md).
Then set the 'DETECTRON2_DATASETS' to the dataset folder in the file 'train_inc_CLIP_CISS.py'.
### Preparing Pretrained CLIP model:
Download the pretrained model here: /pretrained/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

### How to run experiments:

- Use config file: `cfg_file=configs/ade20k/semantic-segmentation/configs/ade20k/semantic-segmentation/zegclip-Base-ADE20K-SemanticSegmentation.yaml`
- see examples in 'train.sh'

## Results
Results will be saved in a folder named `results/`. 
