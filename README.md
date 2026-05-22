<div align="center">

# EasyVFX:Frequency-DrivenDecouplingforResource-EfficientVFX Generation

Accepted by SIGGRAPH 2026

 [Yue Ma](https://mayuelala.github.io/)<sup>&#9993;</sup>, Xu Ye[(https://github.com/hello-it-bit)<sup></sup>, Qinghe Wang<sup>2</sup>, Yucheng Wang<sup>1</sup>, Hongyu Liu<sup>1</sup>, Yinhan Zhang<sup>1</sup>, Xinyu Wang<sup>3</sup>, Yuanpeng Chen<sup>4</sup>, Shanhui Mo<sup>4</sup>, Paul Liang<sup>5</sup>, Fangneng Zhan<sup>5</sup>, Qifeng Chen<sup>1</sup>, <br>
<sup>1</sup>Hong Kong University of Science and Technology <sup>2</sup>Dalian University of Technology <sup>3</sup>Tsinghua University <sup>4</sup>Independent <sup>5</sup>Massachusetts Institute of Technology (MIT) <br>
<sup>*</sup>Equal Contribution <sup>✉</sup>Corresponding Author



**Your star means a lot to us in developing this project!** ⭐⭐⭐
</div>



https://github.com/user-attachments/assets/5862d282-858d-4701-aede-e4872011676c







**📖 Table of Contents**


- [EasyVFX:Frequency-DrivenDecouplingforResource-EfficientVFX Generation]
  - [🛠️ Method Overview](#️-method-overview)
  - [🚀 Getting Started](#-getting-started)
  - [🏃🏼 Running Scripts](#-running-scripts)
  - [🤝🏼 Cite Us](#-cite-us)
  - [🙏 Acknowledgement](#-acknowledgement)


## 🛠️ Method Overview

We introduce **EasyVFX**, a resource-efficient framework that achieves realistic VFX synthesis under stringent constraints. Our core philosophy lies in frequency-domain decomposition: we observe that the complexity of VFX can be significantly mitigated by decoupling high-frequency components, which represent in tricate spatial appearances, from low-frequency components that encapsulate global motion dynamics.This spectral disentanglement transforms a high-dimensional learning problem into manageable sub-tasks, thereby lowering the optimization barrier and reducing data dependency. Building upon this insight,we proposea two-stage training paradigm. First, we design a Frequency-aware Mixture-of-Experts (Freq-MoE) architecture. By utilizing a soft routing mechanism, our model assigns specialize dexperts to distinct spectral bands, enabling them to cultivate robust priors for appearance and motion dynamics. This specialization allows the model to acquire foundational VFX knowledge with fewer GPU resources. Second, we introduce a Test-Time Training strategy powered by a novel Frequency-constraint Loss. This allows the pre-trained model to swiftly adapt to specific, unseen effects through localized optimizations, requiring only about 100 steps on a single GPU.
![Method](https://github.com/user-attachments/assets/ae50d17d-176e-4dcf-b564-87b76cedd7c2)


## 🚀 Getting Started

<details>
<summary><b>Environment Requirement 🔧</b></summary>

**Step 1:** Clone this repo

```
git clone https://github.com/TencentARC/FlexiAct.git
```

**Step 2:** Install required packages

```
bash env.sh
conda activate EasyVFX
```

</details>

<details>
<summary><b>Data Preparation ⏬</b></summary>


**Option 1: Prepare data**

You can download the data we used in our paper at [here](https://huggingface.co/datasets/BianYx/VAP-Data). 
```
cd EasyVFX
git clone https://huggingface.co/datasets/BianYx/VAP-Data
```
you need to organize your training dataset in the following structure:

```
|-- benchmark
    |-- captions
        |-- some VFX
            |-- VFX
                |-- crop.csv
                |-- val_image.csv
    |-- reference_videos
        |-- some VFX
            |-- VFX
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
        |-- extract_vid_and_crop.py
    |-- target_image
        |-- some VFX
            |-- 1.jpg
            |-- 2.webp
            |-- ...
    
```

**Step1: Prepare your reference video**
Execute:
```
python extract_vid_and_crop.py
```


</details>

<details>
<summary><b>Checkpoints 📊</b></summary>

You need to download the base model [CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V) to `{your_cogvideoi2v_path}` by:
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V {your_cogvideoi2v_path}
```

</details>

## 🏃🏼 Running Scripts

<details>
<summary><b>Training</b></summary>
 

**Training script:**
```bash

bash scripts/train/RefAdapter_train.sh -v CUDA_VISIBLE_DEVICES

bash scripts/train/FEI_train.sh -v CUDA_VISIBLE_DEVICES -a your vfx name
```
</details>


<details>
<summary><b>Inference</b></summary>


```
bash scripts/inference/Inference.sh
```


</details>


## 🤝🏼 Cite Us

```
pass
```


## 🙏 Acknowledgement
<span id="acknowledgement"></span>

This repository borrows heavily from [FlexiAct](https://github.com/shiyi-zh0408/FlexiAct) and [CogVideoX](https://github.com/THUDM/CogVideo), thanks to the authors for sharing their code and models.

