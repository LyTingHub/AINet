<h1 align="center">Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration</h1>

Official implementation of **Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration**.

> Mengyuan Liu<sup>1</sup>, Yangting Lin<sup>1</sup>, Qiongjie Cui<sup>2</sup>
>
> <sup>1</sup>State Key Laboratory of General Artificial Intelligence, Peking University, Shenzhen Graduate School  
> <sup>2</sup>Singapore University of Technology and Design

**Published in IEEE Transactions on Image Processing (TIP), 2026.**

[![IEEE TIP](https://img.shields.io/badge/IEEE%20TIP-10.1109%2FTIP.2026.3718431-00629B)](https://doi.org/10.1109/TIP.2026.3718431)
[![arXiv](https://img.shields.io/badge/arXiv-2610.16624-b31b1b.svg)](https://arxiv.org/abs/2610.16624)

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11643466) | [arXiv](https://arxiv.org/abs/2610.16624) | [DOI](https://doi.org/10.1109/TIP.2026.3718431)

## 💡 Environment

The project is developed under the following environment:

- Python 3.9.x
- PyTorch 2.1.0
- CUDA 12.1

For installation of the project dependencies, please run:

```bash
pip install -r requirements.txt
```

## ✨ Training

You can train the model as follows:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name HARPER_result.txt --layer-norm-axis spatial --with-normalization
```

The configuration file is located at `config/harper_config.py`.

## Citation

If this repository is helpful, please cite the published AINet paper:

```bibtex
@article{liu2026adaptive,
  author  = {Liu, Mengyuan and Lin, Yangting and Cui, Qiongjie},
  title   = {Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  volume  = {35},
  pages   = {8484--8496},
  doi     = {10.1109/TIP.2026.3718431}
}
```
