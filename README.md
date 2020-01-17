# Trust Region-Guided Proximal Policy Optimization

Source code for the paper: [Truly Proximal Policy Optmization](https://arxiv.org/abs/1903.07940). The original code was forked from [OpenAI baselines](https://github.com/openai/baselines).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks and [Atari](https://www.atari.com/) discrete game tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [tensorflow1.10](https://www.tensorflow.org/) and Python 3.6.



# Installation

```
git clone --recursive https://github.com/wangyuhuix/TrulyPPO
cd TrulyPPO
pip install -r requirements.txt
```



# Usage

### Command Line arguments

* env: environment ID
* seed: random seed
* num_timesteps: number of timesteps

### Continuous Task

```shell
python -m baselines.ppo2_AdaClip.run --alg=trulyppo --env=InvertedPendulum-v2 --seed=0
```

You can try `--alg=pporb` for PPO-RB and `--alg-trppo` for TR-PPO.

### Discrete Task

```
python -m baselines.ppo2_AdaClip.run alg=trulyppo --env=BeamRiderNoFrameskip-v4 --seed=0 --isatari
```

