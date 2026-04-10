

<!-- # magic-edit.github.io -->

<p align="center">

  <h2 align="center">RectifiedHR: Enable Efficient High-Resolution Synthesis via Energy Rectification</h2>
  <p align="center">
    <a href="https://zhenyangcs.github.io/"><strong>Zhen Yang</strong></a><sup>1*</sup>
    ·
    <a href="https://scholar.google.com/citations?user=d8VVM4UAAAAJ&hl=en"><strong>Guibao Shen</strong></a><sup>1*</sup>
    ·  
    <a href="https://scholar.google.com/citations?user=iahnRcgAAAAJ&hl=zh-CN"><strong>Minyang Li</strong></a><sup>1*</sup>
    ·  
    <a href="https://liang-hou.github.io/"><strong>Liang Hou</strong></a><sup>2</sup>
    ·
    <a href="https://xiaobul.github.io/"><strong>Mushui Liu</strong></a><sup>4</sup>
    ·
    <a href="https://wileewang.github.io/"><strong>Luozhou Wang</strong></a><sup>1</sup>
    ·
    <a href="https://www.xtao.website/"><strong>Xin Tao</strong></a><sup>2</sup>
    ·
    <a href="https://www.yingcong.me/"><strong>Yingcong Chen</strong></a><sup>1,3&#9993;</sup>
    <br>
    <sup>1</sup>HKUST(GZ) · <sup>2</sup>KuaiShou Research · <sup>3</sup>HKUST · <sup>4</sup>Zhejiang University
    <br>
    <sup>*</sup>Both authors contributed equally.
    <sup>&#9993;</sup>Corresponding author.
    </br>
        <a href="https://arxiv.org/abs/2503.02537">
        <img src='https://img.shields.io/badge/Arxiv-RectifiedHR-blue' alt='Paper'></a>
        <a href="https://zhenyangcs.github.io/RectifiedHR-Diffusion/">
        <img src='https://img.shields.io/badge/Project-Website-orange' alt='Project website'></a>
        <!-- <a href="https://drive.google.com/file/d/1JX8w0S9PCD9Ipmo9IiICO8R7e1haTGdF/view?usp=sharing">
        <img src='https://img.shields.io/badge/Dataset-OIR--Bench-green' alt='OIR-Bench'></a>
        <a href="https://iclr.cc/virtual/2024/poster/18242">
        <img src='https://img.shields.io/badge/Video-ICLR-yellow' alt='Video'></a> -->
  </p>
</p>





## Getting Started
1. Create the environment and install the dependencies by running:
```
conda create -n RectifiedHR python=3.9
conda activate RectifiedHR
pip install diffusers==0.34.0
pip install transformers==4.46.2
pip install opencv-python>4.10.0
pip install torch==2.5.1
pip install accelerate==1.1.1
pip install einops==0.8.0
pip install ftfy==6.3.1
```
2. Run with SDXL
```
python run_sdxl.py
```

3. Run with WAN
```
python run_wan.py
```

4. Change the hyperparameters

Edit the parameters in configs_sdxl.py or configs_wan.py to generate higher-quality images.


## BibTeX
```BibTeX
@article{yang2025rectifiedhr,
  title={Rectifiedhr: Enable efficient high-resolution image generation via energy rectification},
  author={Yang, Zhen and Shen, Guibao and Hou, Liang and Liu, Mushui and Wang, Luozhou and Tao, Xin and Wan, Pengfei and Zhang, Di and Chen, Ying-Cong},
  journal={arXiv e-prints},
  pages={arXiv--2503},
  year={2025}
}
```

