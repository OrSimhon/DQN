import gym
import torch
import sys

from Arguments import get_args
from Network import FeedForwardNN
from ReplayBuffer import ReplayBuffer
from DQN import DQNAgent
from EvaluatePolicy import eval_policy


def train(env, hyperparameters, model):
    # Create a model for DQN
    agent = DQNAgent(env=env, nn_class=FeedForwardNN, replay_buffer_class=ReplayBuffer, **hyperparameters)

    # Tries to load in an existing model to continue training
    if model != '':
        print("Loading in {}...".format(model), flush=True)
        agent.online_net.load_state_dict(torch.load(f'./SavedNets/{model}'))
        agent.target_net.load_state_dict(torch.load(f'./SavedNets/{model}'))
        print("Successfully loaded.", flush=True)
    else:
        print("Training from scratch", flush=True)

    agent.train()


def test(env, model):
    print("Testing {}".format(model), flush=True)

    # If the actor model is not specified, then exit
    if model == '':
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(f'./SavedNets/{model}'))
    eval_policy(policy=policy, env=env, render=True)


def main(args):
    env = gym.make(args.env)
    hyperparameters = {
        'lr': 0.0001,
        'gamma': 0.99,
        'batch_size': 128,
        'buffer_size': 10_000,
        'epsilon': 1,
        'min_epsilon': 0.01,
        'epsilon_decay_rate': 0.999,
        'target_update_period': 100,
        'max_episodes': 5000,
        'save_every': 50,
        'render': True,
        'seed': 1234
    }
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, model=args.model)
    else:
        test(env=env, model=args.model)


if __name__ == '__main__':
    args = get_args()
    main(args)
