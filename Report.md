# Results report

## Selected hyperparameters

|Name|Value|
|---|---:|
|Episodes|750|
|Actor learning rate|0.0001|
|Critic learning rate|0.0005|
|Gamma|0.95|
|Tau|0.001|
|Alpha|0.5|
|Beta|0.5|
|Buffer size|100000|
|Batch size|64|
|Target|30.0|

## [Deep Deterministic Policy Gradients](https://arxiv.org/abs/1509.02971)

### Actor network

- ReLU linear layer (in: number of states, out: 256)
- Tanh Linear layer (in: 256, out: number of actions)

### Critic network

- Leaky ReLU linear layer (in: number of states, out: 256)
- Leaky ReLU linear layer (in: number of actions + 256, out: 256)
- Leaky ReLU linear layer (in: 256, out: 128)
- Linear layer (in: 128, out: 1)

Rewards per episode with sampled replay buffer:
![Sampled replay buffer](plots/sample.png)

Rewards per episode with prioritized replay buffer:
![Prioritized replay buffer](plots/prio.png)
