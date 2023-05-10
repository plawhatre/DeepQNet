import gymnasium as gym
import numpy as np

class ExplorationRateDecay:
    def __init__(self, min_rate, max_rate, decay_rate):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.decay_rate = decay_rate

    def __call__(self, episode):
        epsilon = self.min_rate + (self.max_rate - self.min_rate) * \
                np.exp(- self.decay_rate * episode)
        return epsilon 

class FrozenLakeAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 alpha=0.1):
        
        # Environment for agent
        self.env = env
        # discount factor
        self.gamma = gamma
        # learning rate
        self.alpha = alpha
        # Q values
        self.Q_table = np.zeros((self.env.observation_space.n, 
                                 self.env.action_space.n))

    def policy(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            # exploration
            action = self.env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(self.Q_table[state, :])

        return action

    def update_q(self, state, action, reward, next_state):
        self.Q_table[state, action] = (1 - self.alpha) * self.Q_table[state, action] + \
                                       self.alpha * (reward + self.gamma * \
                                                     np.max(self.Q_table[next_state, :])
                                                     )

    def train(self, n_episodes, n_steps_per_episode, explortion_rate):
        rewards_per_episode = []
        for episode in range(n_episodes):
            state = self.env.reset()[0]
            total_reward = 0
            done = False
            epsilon = explortion_rate (episode)

            for _ in range(n_steps_per_episode):
                # select action
                action = self.policy(state, epsilon)
                # update Q table
                next_state, reward, done, _, _ = self.env.step(action) 
                self.update_q(state, action, reward, next_state)   
                # set new states
                state = next_state

                total_reward += reward

                if done:
                    break

            rewards_per_episode.append(total_reward)

            if episode % 1000 == 999:
                print(f"\x1B[35mEpisode: {episode}, Reward: {sum(rewards_per_episode[-1000:])/1000.0}\x1B[0m")

        
if __name__ == '__main__':
    n_episodes = 10000
    n_steps_per_episode = 100
    min_rate = 0.01
    max_rate = 1
    decay_rate = 0.001
    learning_rate = 0.1
    discount_rate = 0.99

    env = gym.make('FrozenLake-v1', render_mode='ansi') 
    explortion_rate = ExplorationRateDecay(min_rate, max_rate, decay_rate)
    agent = FrozenLakeAgent(env, discount_rate, learning_rate)
    agent.train(n_episodes, n_steps_per_episode, explortion_rate)

    print(f"\x1B[33m-----Q_table-----\x1B[0m")
    np.set_printoptions(precision=2)
    print('\x1B[34m', agent.Q_table)