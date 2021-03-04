## Environments supported:

- Matrix Game (MG)

  - [StagHunt](https://en.wikipedia.org/wiki/Stag_hunt)
  - [PrisonerDilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
  - [Chicken](https://en.wikipedia.org/wiki/Chicken_(game))

- Grid World (GW)

  - Monster-Hunt

  - Escalation

The Monster-Hunt and Escalation tasks are implemented according to the Markov Stag-hunt and Coordinated Escalation tasks, which are proposed in the paper "Prosocial learning agents solve generalized Stag Hunts better than selfish ones" [(https://arxiv.org/abs/1709.02865)](https://arxiv.org/abs/1709.02865) for the first time. 

## Usage

- The algorithm/ subfolder contains algorithm-specific code for PPO (Proximal Policy Optimization).

- The envs/ subfolder contains environment wrapper implementations for the Matrix Game and Grid World.

- Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named in the following manner: xxx.sh. 
  - Python training scripts for each environment can be found in the scripts/train/ folder.
  - Python evaling scripts for each environment can be found in the scripts/eval/ folder.

- The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones used in the paper; however, please refer to the appendix for a full list of hyperparameters used.

## 3.Train

Here we use train_GW.sh as an example:

  ```
  cd scripts
  chmod +x ./train_GW.sh
  ./train_GW.sh
  ```

Local results are stored in subfold scripts/results.