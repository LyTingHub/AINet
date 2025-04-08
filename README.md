
<h1 align="center">  Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration</h1>



This is the official implementation of the approach described in the paper of AINet :

> **Adaptive Interaction Network for Human Motion Prediction During Human-Robot Collaboration**
            
> Mengyuan Liu<sup>1</sup>, Yangting Lin<sup>1</sup>, Qiongjie Cui<sup>2</sup>

> <sup>1</sup>State Key Laboratory of General Artificial Intelligence, Peking University, Shenzhen Graduate School, <sup>2</sup>Singapore University of Technology and Design


## 💡 Environment
The project is developed under the following environment:
- Python 3.9.x
- PyTorch 2.1.0
- CUDA 12.1

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 


## ✨ Training
You can train the model as follows:
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name HARPER_result.txt --layer-norm-axis spatial --with-normalization
```
where config files are located at `configs/harper_config.py`.