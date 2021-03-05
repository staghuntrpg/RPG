# RPG (Reward-Randomized Policy Gradient)
### Zhenggang Tang*, Chao Yu*, Boyuan Chen, Huazhe Xu, Xiaolong Wang, Fei Fang, Simon Shaolei Du, Yu Wang, Yi Wu (* equal contribution)

**Website**: https://sites.google.com/view/staghuntrpg

This is the source code for RPG (Reward-Randomized Policy Gradient), which is proposed in the paper "Discovering Diverse Multi-agent Strategic Behavior via Reward Randomization"[(TODO: arxiv link)](arxiv link).

## 1. Supported environments

### 1.1 *Agar.io*

<p align="center"><img src="https://github.com/staghuntrpg/agar/blob/main/gif/agar_demo.gif" align="middle" /></p>


[*Agar*](http://en.wikipedia.org/wiki/Agar.io) is a popular multi-player online game. Players control one or more cells in a Petri dish. The goal is to gain as much mass as possible by eating cells smaller than the player's cell while avoiding being eaten by larger ones. Larger cells move slower. Each player starts with one cell but can split a sufficiently large cell into two, allowing them to control multiple cells. The control is performed by mouse motion: all the cells of a player move towards the mouse position. 

We transform the Free-For-All (FFA) mode of *Agar* (https://agar.io/) into an Reinforcement Learning (RL) environment and we believe it can be utilized as a new Multi-agent RL testbed for a wide range of problems, such as cooperation, team formation, intention modeling, etc. If you want to use *Agar.io* as your testbed, welcome to visit the agar repository: [https://github.com/staghuntrpg/agar](https://github.com/staghuntrpg/agar).

### 1.2 Grid World

- Monster-Hunt
In Monster-Hunt, there is a monster and two apples. The monster keeps moving towards its closest agent while apples are static. When a single agent meets the monster, it **losses** a penalty of 2; if two agents catch the monster at the same time, they both earn a bonus of 5. Eating an apple always gives an agent a bonus of 2. Whenever an apple is eaten or the monster meets an agent, the apple or the monster will respawn randomly. The monster may move over the apple during the chase, in this case, the agent will gain the sum of points if it catches the monster and the apple exactly.

- Escalation
In Escalation, two agents appear randomly and one grid lights up at the initialization. If two agents step on the lit grid simultaneously, each agent can gain 1 point, and the lit grid will go out with an adjacent grid lighting up. Both agents can gain 1 point again if they step on the next lit grid together. But if one agent steps off the path, the other agent will *lose* 0.9L points, where L is the current length of stepping together, and the game is over. Another option is that two agents choose to step off the path simultaneously, neither agent will be punished, and the game continues.

## 2. Usage

```
git clone https://github.com/staghuntrpg/RPG.git --recursive
```

**Tips:** Please don't forget the `--recursive` in the command, or else you will not have Agar.io environment in your fold.

This repository is separated into two folds, GridWorld and Agar, corresponding to the environments used in the paper "Discovering Diverse Multi-agent Strategic Behavior via Reward Randomization". The installation&training instructions can be found in the subfolders of each environment.

## 3. Publication

If you find this repository useful, please cite our paper: TODO
