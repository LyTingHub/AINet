# Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration

Official PyTorch implementation of **Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration**.

Mengyuan Liu<sup>1</sup>, Yangting Lin<sup>1</sup>, Qiongjie Cui<sup>2</sup>

<sup>1</sup>State Key Laboratory of General Artificial Intelligence and Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Peking University, Shenzhen Graduate School, Shenzhen, Guangdong 518055, China<br>
<sup>2</sup>National Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Xi'an Jiaotong University, Xi'an, Shaanxi 710049, China

**Published in IEEE Transactions on Image Processing (TIP), 2026.**

[![IEEE TIP](https://img.shields.io/badge/IEEE%20TIP-10.1109%2FTIP.2026.3718431-00629B)](https://doi.org/10.1109/TIP.2026.3718431)
[![arXiv](https://img.shields.io/badge/arXiv-2610.16624-b31b1b.svg)](https://arxiv.org/abs/2610.16624)

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11643466) | [arXiv](https://arxiv.org/abs/2610.16624) | [DOI](https://doi.org/10.1109/TIP.2026.3718431)

## Environment

The experiments in the paper were run with:

- Python 3.9
- PyTorch 2.1.0
- CUDA 12.1
- One NVIDIA GeForce RTX 4090 GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

If your pip index cannot resolve the CUDA-tagged PyTorch wheels in `requirements.txt`, install PyTorch 2.1.0 with the CUDA 12.1 wheel from the official PyTorch instructions first, then run the command above again.

## Datasets

AINet evaluates human-robot motion prediction on three 3D skeleton datasets. Download each dataset from its official project page and pass the local root with `--data-root`.

| Dataset | Official source | Expected `--data-root` |
| --- | --- | --- |
| HARPER | https://github.com/intelligolabs/HARPER | Directory containing `train/*.pkl` and `test/*.pkl` |
| CoMaD | https://portal.cs.cornell.edu/robots/ | Directory containing `train/<subtask>/HR/<episode>/data.json` and `test/<subtask>/HR/<episode>/data.json` |
| CHICO | https://github.com/AlessioSam/CHICO-PoseForecasting | Directory containing subject folders such as `S00`, `S01`, ..., `S19` |

The code expects the following paper splits:

- HARPER: 12 training subjects and 5 testing subjects, 21 human joints and 23 robot joints.
- CoMaD: 5 subtasks: `cabinet`, `cart`, `handover`, `react`, `tabletop`; 11 human joints and 10 robot joints.
- CHICO: train subjects `S05-S15`, `S01`, `S16`, `S17`; validation subjects `S00`, `S04`; test subjects `S02`, `S03`, `S18`, `S19`; 15 human joints and 9 robot joints.

## Reproducible Settings

The paper uses MPJPE in millimeters. The model observes past human and robot motion and recursively predicts future motion:

| Dataset | Input frames | Train horizon | Eval horizon | Reported frames |
| --- | ---: | ---: | ---: | --- |
| HARPER | 60 | 10 | 30 | 100ms to 1000ms |
| CoMaD | 60 | 10 | 30 | 100ms to 1000ms |
| CHICO | 50 | 10 | 25 | 80ms, 160ms, 320ms, 400ms, 560ms, 720ms, 880ms, 1000ms |

Training uses batch size 256, Adam, learning rate `3e-4` for the first 30000 iterations and `1e-5` afterwards, relative velocity loss, and AWA coefficient regularization with weight 100.

## Training

Run commands from the repository root. Replace each data path with your local dataset path.

HARPER:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset harper \
  --data-root ./data/harper_3d_30 \
  --seed 888 \
  --exp-name HARPER_result.txt \
  --layer-norm-axis spatial \
  --with-normalization \
  --total-iters 31000 \
  --save-every 1000
```

CoMaD:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset comad \
  --data-root ./data/comad_only_3d_skeleton \
  --seed 888 \
  --exp-name COMAD_result.txt \
  --layer-norm-axis spatial \
  --with-normalization \
  --total-iters 31000 \
  --save-every 1000
```

CHICO:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset chico \
  --data-root ./data/CHICO/poses \
  --seed 888 \
  --exp-name CHICO_result.txt \
  --layer-norm-axis spatial \
  --with-normalization \
  --total-iters 31000 \
  --save-every 1000
```

Checkpoints are saved to `config/log2/snapshot*/model/`. TensorBoard logs are written to the same snapshot directory.

## Testing

Evaluate the full test split:

```bash
python test.py --dataset harper --data-root ./data/harper_3d_30 --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth
python test.py --dataset comad --data-root ./data/comad_only_3d_skeleton --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth
python test.py --dataset chico --data-root ./data/CHICO/poses --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth
```

Evaluate each action/subtask separately:

```bash
python test.py --dataset harper --data-root ./data/harper_3d_30 --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth --action-wise --output HARPER_actionwise.txt
python test.py --dataset comad --data-root ./data/comad_only_3d_skeleton --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth --action-wise --output COMAD_actionwise.txt
python test.py --dataset chico --data-root ./data/CHICO/poses --model-pth config/log2/snapshotXXXX/model/model-iter-31000.pth --action-wise --output CHICO_actionwise.txt
```

## Code Layout

- `config/`: dataset-specific experiment configs.
- `dataset/`: HARPER, CoMaD, and CHICO dataloaders.
- `network/`: AINet, LGSI/AWA interaction modules, and MLP blocks.
- `train.py`: unified training entry point.
- `test.py`: unified evaluation entry point.
- `utils/`: logging, filesystem helpers, and experiment utilities.

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

Please also cite the original dataset papers for HARPER, CoMaD, and CHICO when using those datasets.
