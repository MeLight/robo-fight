import sys
import numpy as np
import tensorflow as tf

import pickle
from game_env import FightGameEnv

print(f"ARGV = = = = = =: {sys.argv}")
fight_model_name = sys.argv[1] if len(sys.argv) > 1 else None

if not fight_model_name:
    sys.exit("Provide a model file name")

with open(fight_model_name, 'rb') as f:
    model = pickle.load(f)

env = FightGameEnv()

NUM_OF_FIGHTS = 5

fights_fought = 0
num_of_wins = 0

while fights_fought < NUM_OF_FIGHTS:
    state = env.reset()
    done = False
    while not done:
        # Choose an action based on the current state and the model's predictions
        action_probs = model.predict(np.array([state]))
        action = np.argmax(action_probs[0])

        # Take the chosen action and observe the resulting state and reward
        state, reward, done = env.step(action)

    fights_fought += 1
    if env.fight.get_state()['winner'] > 0:
        num_of_wins += 1

print(f"Done running. Won {num_of_wins} out of {fights_fought} fights.")
