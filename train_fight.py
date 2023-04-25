import sys
import time
import numpy as np
import tensorflow as tf

import pickle
from game_env import FightGameEnv

fight_model_name = sys.argv[1] if len(sys.argv) > 1 else time.time()

env = FightGameEnv()

# Define the game environment
NUM_OF_ACTIONS = 3
NUM_OF_EPISODES = 5

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(NUM_OF_ACTIONS, activation='softmax')
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Define the training loop
def train_step(state, action, reward, next_state, done):
    with tf.GradientTape() as tape:
        # Compute the predicted action probabilities for the current state
        action_probs = model(np.array([state]), training=True)

        # Compute the loss between the predicted and actual action probabilities
        action_mask = tf.one_hot(action, NUM_OF_ACTIONS)
        loss = loss_fn(action, action_probs, sample_weight=reward)

        # Compute the gradients of the loss with respect to the model's parameters
        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters to update them
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Train the model
for episode in range(NUM_OF_EPISODES):
    print(f"Starting eposide {episode} {'-='*100}")
    state = env.reset()
    done = False
    while not done:
        # Choose an action based on the current state and the model's predictions
        action_probs = model.predict(np.array([state]))
        action = np.random.choice(NUM_OF_ACTIONS, p=action_probs[0])

        # Take the chosen action and observe the resulting state and reward
        next_state, reward, done = env.step(action)
        print(f"{reward=}, {done=}")
        # Train the model on the observed transition
        train_step(state, action, reward, next_state, done)

        # Update the current state to be the next state
        state = next_state

with open(f'fight_model_{fight_model_name}.pickle', 'wb') as f:
    pickle.dump(model, f)
