## StarCraft II Autonomous Model (SC2AM)

> 🚧 **Work in progress** — not usable yet

SC2 based parallel Gymnasium environment optimized for RL training at a high throughput.


### Reasoning

AlphaStar proved RL can master StarCraft II but their brute-force approach demanded massive hardware. We believe there's another way. Proving it requires a solid foundation built for RL training from the ground up.

Existing bot frameworks are built for a different purpose. They prioritize developer ergonomics, wrapping `s2clientprotocol` with convenient abstractions like `self.workers.idle`. Excellent for writing competitive rule-based bots, but not designed for running millions of training games.

SC2AM prioritizes RL needs providing Gymnasium-native interface, lean state extraction and parallel execution to achieve as high as possible throughput utilising modern hardware.
