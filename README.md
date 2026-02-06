# Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning
## Introduction
- This repository is the implementation of "Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning". [Paper](https://ieeexplore.ieee.org/document/10051712)
- This study proposed a dual (time and spcace)-domain energy harvesting (EH) approach to maximize EH efficiency of the UAV—RIS system by jointly optimizing the RIS phase shifts vector, the RIS scheduling martix, the length of energy harvesting phase, and the transmit power. 
- For the UAV trajectory design, we considered the density-aware and Fermat point-based algorithms.
- The implementation of DDPG and TD3 using [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/).
- The implementation of SD3 is based on Dr. Pan's research: [Softmax Deep Double Deterministic Policy Gradients](https://github.com/ling-pan/SD3).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me (haoranpeng@cuhk.edu.hk). Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
```
Haoran Peng, and Li-Chun Wang, “Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning”, IEEE Trans. Wireless Commun., vol. 22, no. 10, pp. 6826——6838, Oct., 2023, doi: 10.1109/TWC.2023.3245820 
```
**Bibtex:**
```
  @ARTICLE{10051712,
  author={Peng, Haoran and Wang, Li-Chun},
  journal={IEEE Trans. Wireless Commun.}, 
  title={Energy Harvesting Reconfigurable Intelligent Surface for {UAV} Based on Robust Deep Reinforcement Learning}, 
  year={2023},
  month={Oct.}
  volume={22},
  number={10},
  pages={6826--6838},
 }
```
```
@INPROCEEDINGS{peng1570767WCNC,
  author={Peng, Haoran and Wang, Li-Chun and Li, Geoffrey Ye and Tsai, Ang-Hsun},
  booktitle={Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)}, 
  title={Long-Lasting {UAV}-aided {RIS} Communications based on {SWIPT}},
  address={Austin, TX},
  year={2022},
  month = {Apr.}
}
```
## Requirements (modernized)
- Python: 3.9/3.10
- Pytorch: 1.12+
- gym: 0.26+
- gymnasium (for Stable-Baselines3 v2)
- numpy
- matplotlib
- Stable-Baselines3

## Usage
#### Descriptions of folders
- The folder "XXXX-MultiUT-Time" is the source code for the time-domain EH scheme in the multiple user scenario.
- The folder "XXXX-MultiUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the multiple user scenario.
- The folder "XXXX-SingleUT-Time" is the source code for the time-domain EH scheme in the single user scenario.
- The folder "XXXX-SingleUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the single user scenario.
- The folder "CreateData" is the source code for generating dataset of trajectories files for users and the UAV.

#### Descriptions of files
- For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
- For DRL-based algorithms, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
- You can change the dataset and the scenario in 'gym_foo/envs/foo_env.py'.

#### Training phase (legacy)
1. For the TD3 and DDPG, please execute the TD3.py and DDPG.py to train the model, such as
```
python TD3.py / python DDPG.py
```
***Please change the training mode in the file "gym_foo/envs/foo_env.py" before you executing the training progress.***
For example:
```
class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, LoadData = True, Train = False, multiUT = True, Trajectory_mode = 'Fermat', MaxStep = 41):        
```
If you want to conduct the training phase, the value of "Train" should be "True", otherwise, the value of "Train" should be "Flase" when excuting the testing phase.

2. For the exhaustive search, please execute the ExhaustiveSearch.py to reproduce the simulation results.
3. For the SD3, please execute main.py to train a new model. 

***Legacy scripts were built for Gym 0.15.3. For modern Gym/Stable-Baselines3, use the scripts below.***

## Colab quick start (Gym 0.26+ / Stable-Baselines3)
1. Install dependencies (uses `requirements.txt`):
```
!pip install -r requirements.txt
```
2. Train SAC:
```
!python train_sac.py --timesteps 200000 --model-name sac_colab --device cuda --torch-threads 1 --torch-inter-op-threads 1
```
3. Train PPO:
```
!python train_ppo.py --timesteps 200000 --model-name ppo_colab --device cuda --torch-threads 1 --torch-inter-op-threads 1
```
4. Evaluate and plot SINR/Sum-Rate:
```
!python eval_plots.py --algo sac --model-path sac_logs/sac_colab.zip --episodes-per-pt 5
```

#### Testing phase
Please execute test.py to evaluate DRL models. Before you produce the testing results, please change the dataset and scenario in 'gym_foo/envs/foo_env.py'.

#### The EH efficiency
The EH efficiency = the harvested energy / the received energy from RF signals
