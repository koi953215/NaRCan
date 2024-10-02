## NaRCan: Natural Refined Canonical Image with Integration of Diffusion Prior for Video Editing
<!-- [![_](https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg)](https://arxiv.org/abs/2312.06640)  -->

**NeurIPS 2024**

[![_](https://img.shields.io/badge/💊-Project%20Page-orange)](https://koi953215.github.io/NaRCan_page/) 
[![_](https://img.shields.io/badge/arXiv-2406.06523-b31b1b.svg)](https://arxiv.org/abs/2406.06523) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Koi953215/NaRCan_demo) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=koi953215/NaRCan) 

**NaRCan** a video editing framework, integrates a hybrid deformation field network with diffusion priors to address the challenge of maintaining the canonical image as a natural image.

<p align="center">
    <img src="assets/teaser.png">
</p>

---
## Overview
<!-- ![overall_structure](assets/pipeline.png) -->

<p align="center">
    <img src="assets/pipeline.png">
</p>



## <a name="installation"></a>Installation

```shell
# clone this repo
git clone https://github.com/koi953215/NaRCan.git
cd NaRCan

# create environment
conda create -n narcan python=3.10
conda activate narcan
pip install -r requirements.txt
```

## <a name="Preprocessing"></a>Preprocessing (LoRA Fine-tuning)

Now, we need to use a technique similar to [RealFill](https://realfill.github.io/) to finetune the diffusion model, which is the diffusion prior in our pipeline. **(You can visit this [repo](https://github.com/thuanz123/realfill) for more details about the RealFill environment and operations.)**

<!-- | Model Name | Description | HuggingFace | BaiduNetdisk | OpenXLab |
| :---------: | :----------: | :----------: | :----------: | :----------: |
| v2.pth | IRControlNet trained on filtered laion2b-en  | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth) | [download](https://pan.baidu.com/s/1uTAFl13xgGAzrnznAApyng?pwd=xiu3)<br>(pwd: xiu3) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
| [GMFlow](https://github.com/haofeixu/gmflow) | IRControlNet trained on FFHQ | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth) | [download](https://pan.baidu.com/s/1kvM_SB1VbXjbipLxdzlI3Q?pwd=n7dx)<br>(pwd: n7dx) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
-->

First, switch to the RealFill folder.

```
cd realfill
```

Uniformly sample 5~10 frames from your dataset (scene) and place them in the `ref` folder. Next, put any single frame in the `target` folder and name it `target.png` (in practice, select the middle frame of your scene).

**Note: please organize your dataset using the following folder structure.**

```
data
└─── <your-scene-name>
    ├─── ref
    │    └─── [any number of images]
    └─── target
         ├─── target.png
         └─── mask.png
```

Open `scripts/train.sh` and make the following modifications.

```
export TRAIN_DIR="data/<your-scene-name>"
export OUTPUT_DIR="<your-scene-name>-model"
```

After completing the above steps, we can begin fine-tuning our model. **(Fine-tuning requires a large amount of GPU memory. If your GPU has limited memory, please refer to the [RealFill GitHub](https://github.com/thuanz123/realfill), which provides detailed instructions on how to train on a low-memory GPU.)**

```
cd scripts
bash train.sh
```


## <a name="Train"></a>Train a new model



## <a name="inference"></a>Inference

We provide some examples of inference commands; for additional arguments, please refer to [inference.py](inference.py).



### Blind Video Denoising

```shell
python -u inference.py \
--version v2 \
--task dn \
--upscale 1 \
--cfg_scale 4.0 \
--batch_size 10 \
--input inputs/noise_50/flamingo \
--output results/Denoise/flamingo  \
--config configs/inference/my_cldm.yaml \
--final_size "(480, 854)" \
--merge_ratio "(0.6, 0)" \
--better_start 
```

### Blind Video Super-Resolution

```shell
python -u inference.py \
--version v2 \
--task sr \
--upscale 4 \
--cfg_scale 4.0 \
--batch_size 10 \
--input inputs/BDx4/rhino \
--output results/SR/rhino  \
--config configs/inference/my_cldm.yaml \
--final_size "(480, 854)" \
--merge_ratio "(0.6, 0)" 
```

## Citation

Please cite us if our work is useful for your research.

```
@article{chen2024narcan,
    title={NaRCan: Natural Refined Canonical Image with Integration of Diffusion Prior for Video Editing},
    author={Ting-Hsuan Chen and Jiewen Chan and Hau-Shiang Shiu and Shih-Han Yen and Chang-Han Yeh and Yu-Lun Liu},
    journal={arXiv},
    year={2024}
}
```

<!-- ## License

This project is released under the [Apache 2.0 license](LICENSE). -->

## Acknowledgement

This research was funded by the [National Science and Technology Council](https://www.nstc.gov.tw/?l=en), Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2. The authors are grateful to [Google](https://about.google/), [NVIDIA](https://www.nvidia.com/en-us/), and [MediaTek Inc](https://www.mediatek.com/). for generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

<!-- ## Contact

If you have any questions, please feel free to contact with me at tchen783@usc.edu -->