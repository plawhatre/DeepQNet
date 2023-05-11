import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, s):
        x = torch.relu(self.fc1(s))
        y = self.fc2(x)
        return y

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

class ExplorationRateDecay:
    def __init__(self, min_rate, max_rate, decay_rate):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.decay_rate = decay_rate

    def __call__(self, episode):
        epsilon = self.min_rate + (self.max_rate - self.min_rate) * \
                np.exp(- self.decay_rate * episode)
        return epsilon 
      
class DQNAgent:
    def __init__(self,
                 env,
                 n_hidden=64,
                 buffer_size=1000,
                 discount_factor=0.99,
                 learing_rate=0.001,
                 target_net_update_freq=10,
                 min_rate=0.01,
                 max_rate=1,
                 decay_rate=0.001):
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size) 
        self.policy_net = DQN(n_states=1,
                              n_actions=env.action_space.n,
                              n_hidden=n_hidden)
        self.target_net = DQN(n_states=1,
                              n_actions=env.action_space.n,
                              n_hidden=n_hidden)
        self.explortion_rate = ExplorationRateDecay(min_rate, max_rate, decay_rate)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learing_rate)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.target_net_update_freq = target_net_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

    def policy(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            # exploration
            action = self.env.action_space.sample()
        else:
            # exploitation
            state = torch.tensor([state], dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1).item()
        return action

    def train(self, batch_size, n_episodes, n_steps_per_episode):
        rewards_per_episode = []
        # set up memory buffer 
        state = self.env.reset()[0]
        for i in range(self.replay_buffer.buffer_size):
            action = self.policy(state, self.explortion_rate(i))
            next_state, reward, done, _, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)             

        #  start training
        for episode in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            epsilon = self.explortion_rate(episode)

            for step in range(n_steps_per_episode):
                total_reward = 0

                # determine and perform action
                action = self.policy(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)

                # store in experience replay
                self.replay_buffer.push(state, action, reward, next_state, done) 
                sampled_batch = self.replay_buffer.sample(batch_size)

                #  compute q values
                # step 1: Pass through policy network                
                sampled_batch_states = sampled_batch[0][:]\
                    .clone().detach().reshape((batch_size, 1))
                sampled_batch_actions = sampled_batch[1][:]\
                    .clone().detach().reshape((batch_size, 1))
                
                pred_q_values = self.policy_net(sampled_batch_states).gather(1, sampled_batch_actions)
                
                # step 2: Pass through target network
                sampled_batch_next_states = sampled_batch[3][:]\
                    .clone().detach().reshape((batch_size, 1))
                sampled_batch_rewards = sampled_batch[2][:]\
                    .clone().detach().reshape((batch_size, 1))
                sampled_batch_dones = sampled_batch[4][:]\
                    .clone().detach().reshape((batch_size, 1))

                target_q_values_for_all_actions = self.target_net(sampled_batch_next_states)
                                
                target_max_q_values = target_q_values_for_all_actions.max(dim=1).values

                target_q_values = sampled_batch_rewards \
                    + self.discount_factor * target_max_q_values.reshape(batch_size, 1) \
                        * (1 - sampled_batch_dones)

                # Compute loss
                loss = self.criterion(target_q_values, pred_q_values)

                # Backpropagation
                self.criterion.zero_grad()
                loss.backward()
                self.optimizer.step()

                # reward increament for present step
                total_reward += sampled_batch_rewards.sum().item()

                if self.target_net_update_freq == step:
                    self.update_target_net()

            assert (
                total_reward <= batch_size
            ), f"Total reward per batch {total_reward} but shouldn't exceed {batch_size}"

            rewards_per_episode.append(total_reward)

            if episode % 100 == 99:
                print(f"\x1B[35mEpisode: {episode}, ",
                      f"Reward: {sum(rewards_per_episode[-1000:])/(batch_size*1000.0)}\x1B[0m""")


if __name__ == '__main__':
    # Params
    n_hidden = 64
    buffer_size = 1000
    batch_size = 32
    discount_factor = 0.99
    learing_rate = 0.001
    target_net_update_freq = 10
    min_rate = 0.01
    max_rate = 1
    decay_rate = 0.001
    n_episodes = 10000
    n_steps_per_episode = 100

    # Envionment and Agent
    env = gym.make('FrozenLake-v1', render_mode='ansi') 
    agent = DQNAgent(env,
                     n_hidden=n_hidden,
                     buffer_size=buffer_size,
                     discount_factor=discount_factor,
                     learing_rate=learing_rate,
                     target_net_update_freq=target_net_update_freq,
                     min_rate=min_rate,
                     max_rate=max_rate,
                     decay_rate=decay_rate)

    # Train the agent on environment
    agent.train(batch_size, n_episodes, n_steps_per_episode)