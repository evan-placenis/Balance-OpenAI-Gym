import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24, activation = "relu"))
model.add(Dense(24, activation = "relu"))
model.add(Dense(actions, activation = "linear"))

agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit = 5000, window_length = 1),
    policy = BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup = 10,
    target_model_update = 0.01
)

agent.compile(tf.keras.optimizers.legacy.Adam(lr = 0.001), metrics = ["mae"])
agent.fit(env, nb_steps = 20000, visualize = False, verbose = 1)

agent.save_weights("trained_max_weights.h5", overwrite = True)

results = agent.test(env, nb_episodes = 10, visualize = True)
print(np.mean(results.history["episode_reward"]))



env.close()




