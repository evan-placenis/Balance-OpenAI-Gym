
import gym
import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

class RunModel():
    def __init__(self):
        #initialize variables so they can be overwritten
        self.env = 0
        self.model = 0
        self.agent = 0 

    def create_env(self):
        self.env = gym.make("CartPole-v1")
        states = self.env.observation_space.shape[0]
        actions = self.env.action_space.n

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,states)))
        self.model.add(Dense(24, activation = "relu"))
        self.model.add(Dense(24, activation = "relu"))
        self.model.add(Dense(actions, activation = "linear"))


        self.agent = DQNAgent(
            model = self.model,
            memory = SequentialMemory(limit = 5000, window_length = 1),
            policy = BoltzmannQPolicy(),
            nb_actions = actions,
            nb_steps_warmup = 10,
            target_model_update = 0.01
        )

    def run_full_model(self):
        self.create_env()
        self.agent.compile(tf.keras.optimizers.legacy.Adam(lr = 0.001), metrics = ["mae"])
        self.agent.load_weights("./Weights/trained_max_weights.h5")
        results = self.agent.test(self.env, nb_episodes = 4, visualize = True)
        print(np.mean(results.history["episode_reward"]))
        self.env.close()


    def run_half_model(self):
        self.create_env()
        self.agent.compile(tf.keras.optimizers.legacy.Adam(lr = 0.001), metrics = ["mae"])
        self.agent.load_weights("./Weights/trained_half_weights.h5")
        results = self.agent.test(self.env, nb_episodes = 8, visualize = True)
        print(np.mean(results.history["episode_reward"]))
        self.env.close()

    def run_untrained_model(self):
        self.create_env()
        min_steps = 60
        episode_rewards = []
        for episode in range(10):  # Number of episodes to run
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done or steps < min_steps:
                action = self.agent.forward(state)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                self.env.render()
            
            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        print("Mean episode reward:", mean_reward)

        self.env.close()#

test = RunModel()
test.run_half_model()