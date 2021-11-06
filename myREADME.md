# AdaBins
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=adabins-depth-estimation-using-adaptive-bins) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=adabins-depth-estimation-using-adaptive-bins)

My readme file based on official readme to facilitate train/val/test.

Official implementation of [Adabins: Depth Estimation using adaptive bins](https://arxiv.org/abs/2011.14141). 

## Download links
* You can download the pretrained models "AdaBins_nyu.pt" and "AdaBins_kitti.pt" from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
* You can download the predicted depths in 16-bit format for NYU-Depth-v2 official test set and KITTI Eigen split test set [here](https://drive.google.com/drive/folders/1b3nfm8lqrvUjtYGmsqA5gptNQ8vPlzzS?usp=sharing)

## Installation
### Dependencies
- pytorch >= 1.6
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
    ```bash
    conda install -c conda-forge tqdm
    ```
- matplotlib
- scipy
- wandb (for logging, optional) 

### Wandb (weights and bias)
Web-based MLOps tool. Wandb needs an account before using: https://wandb.ai/site. Then run `wandb login` in shell 
to log in.
**P.S.** Do not stop code running in wandb web GUI, this may cause unreleased GPU memeory. 

### Data preparation
- The data preparation of NYUDv2 dataset is same as [BTS](https://github.com/cogaplex-bts/bts).  

## Train
```bash
# train model on nyudv2 dataset
CUDA_VISIBLE_DEVICES=0,1 python train.py args_train_nyu.txt 
```
Logging by wandb (project and current run) will be displayed in the wwandb weibsite.  

## Inference
Move the downloaded weights to a directory of your choice (we will use "./pretrained/" here). 
You can then use the pretrained models like so:

```python
from models import UnetAdaptiveBins
import model_io

# depth ranges in meters
MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256  # num of discrete depth bins 

# NYU
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
pretrained_path = "./pretrained/AdaBins_nyu.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

# TODO: how to get example_rgb_batch?
example_rgb_batch = ...

# predict depth bin boundaries and depth values 
bin_edges, predicted_depth = model(example_rgb_batch)


# KITTI
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "./pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)
```
Note that the model returns bin-edges (instead of bin-centers).

**Recommended way:** `InferenceHelper` class in [infer.py](./infer.py) provides an easy interface for inference and handles various types of inputs (with any prepocessing required). It uses Test-Time-Augmentation (H-Flips) and also calculates bin-centers for you:
```python
from infer import InferenceHelper
from PIL import Image

infer_helper = InferenceHelper(dataset='nyu')

# TODO: how to get example_rgb_batch?
example_rgb_batch = ...

# predict depth of a batched rgb tensor
bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py args_test_nyu.txt
```

## TODO:
* Add instructions for Evaluation and Training.
* Add Colab demo
* Add UI demo
* Remove unnecessary dependencies

