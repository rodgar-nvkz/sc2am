## StarCraft II Autonomous Model (SC2AM)

> 🚧 **Work in progress** — not usable yet

SC2 based parallel Gymnasium environment optimized for RL training at a high throughput.


### Reasoning

AlphaStar proved RL can master StarCraft II but their brute-force approach demanded massive hardware. We believe there's another way. Proving it requires a solid foundation built for RL training from the ground up.

Existing bot frameworks are built for a different purpose. They prioritize developer ergonomics, wrapping `s2clientprotocol` with convenient abstractions like `self.workers.idle`. Excellent for writing competitive rule-based bots, but not designed for running millions of training games.

SC2AM prioritizes RL needs providing Gymnasium-native interface, lean state extraction and parallel execution to achieve as high as possible throughput utilising modern hardware.


## RL References
- [RLlib](https://docs.ray.io/en/latest/rllib/)
- [PettingZoo](https://pettingzoo.farama.org/)
- [DreamerV3](https://arxiv.org/pdf/2301.04104v1) [Impl](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dreamerv3)
- [LeanRL](https://arxiv.org/abs/2111.08819) [Impl](https://github.com/meta-pytorch/LeanRL)
- [EPyMARL](https://arxiv.org/pdf/2006.07869) [Impl](https://github.com/uoe-agents/epymarl)
- [SMACv2](https://arxiv.org/abs/2212.07489) [Impl](https://github.com/oxwhirl/smacv2)

## General ML References
- [Chinchilla Law](https://arxiv.org/abs/2203.15556)



## No BIAS Techniques
- Hindsight Experience Replay (HER)
- Go-Explore (Uber AI, Montezuma’s Revenge)
- Curriculum Learning
- Quality Diversity (MAP-Elites)



APPO RLLib
1 env Steps/s: 730
2 envs Steps/s: 1340
4 envs Steps/s: 2230
8 envs Steps/s: 3300

PPO GPU
1 envs fps 420
4 envs fps 1320
8 Envs fps 1880

PPO CPU
1 envs fps 630
4 envs fps 1700
8 Envs fps 2400


PPO x4: CPU 900, GPU 2000
