import gymnasium as gym
import numpy as np

class LearningRateDecay:
    def __init__(self, min_rate, max_rate, decay_rate):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.decay_rate = decay_rate

    def __call__(self, episode):
        alpha = self.min_rate + (self.max_rate - self.min_rate) * \
                np.exp(- self.decay_rate * episode)
        return alpha 

class FrozenLakeAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 epsilon=1,
                 epsilon_decay=0.001):
        
        # Environment for agent
        self.env = env
        # discount factor
        self.gamma = gamma
        # exploration exploitation tradeoff parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # Q values
        self.Q_table = np.zeros((self.env.observation_space.n, 
                                 self.env.action_space.n))

    def policy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # exploration
            action = self.env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(self.Q_table[state, :])

        return action

    def update_q(self, alpha, state, action, reward, next_state):
        self.Q_table[state, action] = (1 - alpha) * self.Q_table[state, action] + \
                                       alpha * (reward + self.gamma * \
                                                     np.argmax(self.Q_table[next_state, :])
                                                     )

    def train(self, n_episodes, learning_rate):
        rewards_per_episode = []
        for episode in range(n_episodes):
            state = self.env.reset()[0]
            total_reward = 0
            done = False
            alpha = learning_rate(episode)

            while not done:
                # select action
                action = self.policy(state)
                # update Q table
                next_state, reward, done, _, _ = self.env.step(action) 
                self.update_q(alpha, state, action, reward, next_state)   
                # set new states
                state = next_state

                total_reward += reward

                if done:
                    break

            rewards_per_episode.append(total_reward)

            if episode % 1000 == 999:
                print(f"\x1B[35mEpisode: {episode}, Reward: {sum(rewards_per_episode[-1000:])/1000.0}\x1B[0m")

        
if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='ansi') 
    learning_rate = LearningRateDecay(0.2, 1, 0.0000001)
    agent = FrozenLakeAgent(env)
    agent.train(10000, learning_rate)
    print(f"\x1B[33m-----Q_table-----\x1B[0m")
    np.set_printoptions(precision=3)
    print('\x1B[34m', agent.Q_table)