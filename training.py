from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import argparse

from agent import Agent

def ddpg(n_episodes=2000, max_t=1000,
        target=100.0, actor_model='checkpoint_actor.pth', critic_model='checkpoint_critic.pth'):
    """Deep Deterministic Policy Gradients.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        target (float): desired minimal average per 100 episodes
        actor_model (str): path to save actor model
        critic_model (str): path to save critic model
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), actor_model)
            torch.save(agent.critic_local.state_dict(), critic_model)
            break
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Reacher_Linux_NoVis/Reacher.x86_64')
    parser.add_argument('--actor_model', type=str, help='Path to save actor model',
                        default='checkpoint_actor.pth')
    parser.add_argument('--critic_model', type=str, help='Path to save critic model',
                        default='checkpoint_critic.pth')
    parser.add_argument('--buffer', type=str, help='Replay buffer type - sample or prioritized',
                        default='prioritized')
    parser.add_argument('--episodes', type=int, help='Maximum number of training episodes',
                        default=2000)
    parser.add_argument('--frames', type=int, help='Maximum number of frames in training episode',
                        default=1000)
    parser.add_argument('--target', type=float, help='Desired minimal average per 100 episodes',
                        default=30.0)
    parser.add_argument('--buffer_size', type=int, help='Replay buffer size',
                        default=100000)
    parser.add_argument('--batch_size', type=int, help='Minibatch size',
                        default=64)
    parser.add_argument('--gamma', type=float, help='Discount factor',
                        default=0.99)
    parser.add_argument('--tau', type=float, help='For soft update of target parameters',
                        default=0.01)
    parser.add_argument('--alpha', type=float, help='Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)',
                        default=0.5)
    parser.add_argument('--beta', type=float, help='Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)',
                    default=0.5)
    parser.add_argument('--actor_learning_rate', type=float, help='Actor learning rate',
                        default=0.001)
    parser.add_argument('--critic_learning_rate', type=float, help='Critic learning rate',
                        default=0.005)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Training')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # initialize agent
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  training=True, args=args)

    scores = ddpg(n_episodes=args.episodes, max_t=args.frames, target=args.target,
                  actor_model=args.actor_model, critic_model=args.critic_model)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(args.model + '.png')

    env.close()
