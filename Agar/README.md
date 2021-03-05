This is the training&rendering code for Agar.io environment [https://github.com/staghuntrpg/agar.git](https://github.com/staghuntrpg/agar.git).

## 1. User Guide

1. The subfolder a2c_ppo_acker/ in this repository is heavily based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. More details can be found in [this URL](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/README.md)
2. The subfolder agar/ is a submodule and related to the Agar.io [repository](https://github.com/staghuntrpg/agar.git), make sure it is not empty when you start to use the training&rendering code. Or you can use the following command to download the Agar.io source code. 
```
cd agar
git clone https://github.com/staghuntrpg/agar.git
```

## 2. Installation

Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

```
# create conda environment
conda create -n rpg python==3.6.1
conda activate rpg
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install gym matplotlib imageio opencv-python tensorboardX pillow pyglet
```

## 3. Enjoy

### 3.1 Train

```
chmod +x train.sh
./train.sh
```

### 3.2 Render

```
python render.py
```