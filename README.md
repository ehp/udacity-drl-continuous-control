# Udacity deep reinforcement learning - continuous control

## Introduction

This project train one agent (option 1) to move a double-jointed arm to target locations.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
Thus, the goal of this agent is to maintain its position at the target location for as many time steps as possible.
The agent must receive an average reward (over 100 episodes) of at least +30.

* Link to [original repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).
* Link to [training reports](Report.md).

## Environment details

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Installation

Exclusive virtualenv is recommended:

```bash
virtualenv --python /usr/bin/python3 .venv
. .venv/bin/activate
``` 

Install dependencies:
```bash
pip install -r requirements.txt
```

If you encounter missing tensorflow 1.7.1 dependency try this:
```bash
pip install torch numpy pillow matplotlib grpcio protobuf
pip install --no-dependencies unityagents
```

Download unity environment files:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

And/or [NoVis alternative](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (only for Linux).

## Training

Run training with default parameters:

```bash
python3 training.py
```

All training parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Reacher_Linux_NoVis/Reacher.x86_64|
|--actor_model|Path to save actor model|checkpoint_actor.pth|
|--critic_model|Path to save critic model|checkpoint_critic.pth|
|--buffer|Replay buffer type - sample or prioritized|prioritized|
|--episodes|Maximum number of training episodes|2000|
|--frames|Maximum number of frames in training episode|1000|
|--target|Desired minimal average per 100 episodes|30.0|
|--buffer_size|Replay buffer size|1000000|
|--batch_size|Minibatch size|1024|
|--gamma|Discount factor|0.85|
|--tau|For soft update of target parameters|0.001|
|--alpha|Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)|0.5|
|--beta|Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)|0.5|
|--actor_learning_rate|Learning rate|0.0005|
|--critic_learning_rate|Learning rate|0.0005|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

## Testing

Run test with default parameters:

```bash
python3 testing.py
```

All testing parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Reacher_Linux/Reacher.x86_64|
|--actor_model|Path to save actor model|checkpoint_actor.pth|
|--critic_model|Path to save critic model|checkpoint_critic.pth|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

Pretrained models for [actor](models/actor.pth) and [critic](models/critic.pth). 

## Future work

- Implement other RL algorithms like REINFORCE, TNPG, RWR, REPS, TRPO, CEM, CMA-ES etc..
- Use more hw/time to do full hyperparameter space search for the best hyperparameters of this task.

## Licensing

Code in this repository is licensed under the MIT license. See [LICENSE](LICENSE).
