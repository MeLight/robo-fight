import numpy as np
import tensorflow as tf

import pickle
from game_env import FightGameEnv

with open('model_20i.pickle', 'rb') as f:
    model = pickle.load(f)

env = FightGameEnv()
state = env.reset()
done = False
while not done:
    # Choose an action based on the current state and the model's predictions
    action_probs = model.predict(np.array([state]))
    action = np.argmax(action_probs[0])

    # Take the chosen action and observe the resulting state and reward
    state, reward, done, _ = env.step(action)