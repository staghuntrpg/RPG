# RPG (Reward-Randomized Policy Gradient)
### Zhenggang Tang*, Chao Yu*, Boyuan Chen, Huazhe Xu, Xiaolong Wang, Fei Fang, Simon Shaolei Du, Yu Wang, Yi Wu (* equal contribution)

**Website**: https://sites.google.com/view/staghuntrpg

This is the source code for RPG (Reward-Randomized Policy Gradient), which is proposed in the paper "Discovering Diverse Multi-agent Strategic Behavior via Reward Randomization"[(TODO: arxiv link)](arxiv link).

## 1. Supported environments

### 1.1 *Agar.io*

<div align=center><img src=https://github.com/staghuntrpg/agar/tree/main/gif/agar_demo.gif/> </div>
 

 
[*Agar*](http://en.wikipedia.org/wiki/Agar.io) is a popular multi-player online game. Players control one or more cells in a Petri dish. The goal is to gain as much mass as possible by eating cells smaller than the player's cell while avoiding being eaten by larger ones. Larger cells move slower. Each player starts with one cell but can split a sufficiently large cell into two, allowing them to control multiple cells. The control is performed by mouse motion: all the cells of a player move towards the mouse position. 

We transform the Free-For-All (FFA) mode of *Agar* (https://agar.io/) into an Reinforcement Learning (RL) environment and we believe it can be utilized as a new Multi-agent RL testbed for a wide range of problems, such as cooperation, team formation, intention modeling, etc. If you want to use *Agar.io* as your testbed, welcome to visit the agar repository: [https://github.com/staghuntrpg/agar](https://github.com/staghuntrpg/agar).

### 1.2 Grid World

- Monster-Hunt
- Escalation

## 2. Usage

```
git clone https://github.com/staghuntrpg/RPG.git --recursive
```

**Tips:** Please don't forget the `--recursive` in the command, or else you will not have Agar.io environment in your fold.

This repository is separated into two folds, GridWorld and Agar, corresponding to the environments used in the paper "Discovering Diverse Multi-agent Strategic Behavior via Reward Randomization". The installation&training instructions can be found in the subfolders of each environment.

## 3. Publication

If you find this repository useful, please cite our paper: TODO
