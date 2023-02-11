import torch
from torch import nn
import numpy as np
import time
import itertools


class DQNAgent:
    def __init__(self, env, nn_class, replay_buffer_class, **hyper_parameters):

        # look for a gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device.type} ")

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self._init_hyper_parameters(hyper_parameters)  # Initialize Hyper Parameters
        self.memory = replay_buffer_class(capacity=self.buffer_size)  # Initialize replay buffer

        # Initialize online and target q-networks
        self.online_net = nn_class(self.obs_dim, self.act_dim)
        self.target_net = nn_class(self.obs_dim, self.act_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.requires_grad_(requires_grad=False)

        # optimizer and loss function
        self.online_optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            'Episode': 0,  # current episode
            'total_steps': 0,
            'rewards': [],  # episodic returns in batch
            'losses': [],  # losses of actor network in current iteration
            'total_time': 0  # time from start of the learning
        }

    def store_experience(self, experience):
        self.memory.store(experience)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            return torch.argmax(self.online_net(observation).detach()).item()
        else:
            return self.env.action_space.sample()

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_step(self):
        if len(self.memory) < self.batch_size:
            return 0

        # sample minibatch
        states, actions, next_states, rewards, not_dones = self.memory.sample(self.batch_size)

        pred_q = self.online_net(states)
        target_q_next = self.target_net(next_states)
        TD_target = pred_q.clone().detach()
        batch_index = np.arange(self.batch_size)
        TD_target[batch_index, actions] = torch.from_numpy(rewards) + self.gamma * target_q_next.max(dim=1)[
            0] * not_dones

        self.epsilon_decay()

        self.online_optimizer.zero_grad()
        loss = self.criterion(pred_q, TD_target)  # MSE

        loss.backward()
        self.online_optimizer.step()
        return loss.detach()

    def save_render_log(self, step):
        if self.logger['Episode'] > 0 and (self.logger['Episode'] % self.save_every == 0):
            if step == 0:
                torch.save(self.online_net.state_dict(), './SavedNets/' + str(self.env.unwrapped.spec.id) + '_dqn.pth')
                self._log_summary()
            if self.render:
                self.env.render()
        elif self.render and (self.logger['Episode'] - 1) % self.save_every == 0:
            self.env.close()

    def train(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_score = 0
            for step in itertools.count():
                self.save_render_log(step)
                self.logger['total_steps'] += 1
                action = self.choose_action(observation=state)
                next_state, reward, done, info = self.env.step(action)
                episode_score += reward
                self.store_experience((state, action, next_state, reward, not done))

                state = next_state
                loss = self.update_step()
                self.logger['losses'].append(loss)
                if not self.logger['total_steps'] % self.target_update_period:
                    self.update_target_network()
                if done:
                    self.logger['rewards'].append(episode_score)
                    self.logger['Episode'] = episode + 1
                    break

    def _init_hyper_parameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters
        :param hyperparameters: the extra arguments included when creating DQN model,
        should onlt include hyperparameters defined below with custom values
        """
        self.lr = 0.0001
        self.gamma = 0.99
        self.batch_size = 128
        self.buffer_size = 10_000
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.999
        self.target_update_period = 100
        self.max_episodes = 5_000
        self.save_every = 50

        # Miscellaneous parameters
        self.render = True
        self.seed = None

        # Change any default values to custom values for specified HP
        for param, val in hyperparameters.items():
            exec('self.' + param + '=' + str(val))

        # Sets the seed if specified
        if self.seed is not None:
            # validity check
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.env.seed(self.seed)
            self.env.action_space.seed(self.seed)
            print("Successfully set seed to {}".format(self.seed))

    def _log_summary(self):
        """ Print to stdout what we have logged so far in the most recent batch """

        # Calculate logging values
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        self.logger['total_time'] += delta_t

        total_hours = self.logger['total_time'] // 3600
        total_minutes = self.logger['total_time'] // 60 - total_hours * 60

        episode = self.logger['Episode']

        avg_ep_rews = np.mean(self.logger['rewards'][-100:])
        avg_loss = np.mean(self.logger['losses'][-100:])

        # Print logging statements
        print(flush=True)
        print("-------------------- Episode #{} --------------------".format(episode), flush=True)
        print("Average Episodic Return: {:.3f}".format(avg_ep_rews), flush=True)
        print("Average Loss: {:.5f}".format(avg_loss), flush=True)
        print("Total learning time: Hours: {:.0f} | Minutes: {:.0f}".format(total_hours, total_minutes), flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['rewards'] = []
        self.logger['losses'] = []
