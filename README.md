
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
### Preparing Pretrained CLIP model:
Download the pretrained model here: /pretrained/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

### How to run experiments:

- Use config file: `cfg_file=configs/ade20k/semantic-segmentation/configs/ade20k/semantic-segmentation/zegclip-Base-ADE20K-SemanticSegmentation.yaml`
- see examples in 'train.sh'

## Results
Results will be saved in a folder named `results/`. 
