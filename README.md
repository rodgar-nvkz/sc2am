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


## Progress
* 20260111: gym_v0 / ppo_v0 Win rate: 20/20 (100.0%) Average reward: 1.75
