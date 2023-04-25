import sys
import time
import numpy as np
import tensorflow as tf

import pickle
from game_env import FightGameEnv

fight_model_name = sys.argv[1] if len(sys.argv) > 1 else time.time()

env = FightGameEnv()

# Define the game environment
num_actions = 2
num_episodes = 20

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax')
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
        action_mask = tf.one_hot(action, num_actions)
        loss = loss_fn(action, action_probs, sample_weight=reward)

        # Compute the gradients of the loss with respect to the model's parameters
        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters to update them
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Train the model
for episode in range(num_episodes):
    print(f"Starting eposide {episode}" + str())
    state = env.reset()
    done = False
    while not done:
        # Choose an action based on the current state and the model's predictions
        action_probs = model.predict(np.array([state]))
        action = np.random.choice(num_actions, p=action_probs[0])

        # Take the chosen action and observe the resulting state and reward
        next_state, reward, done, _ = env.step(action)
        print(f"{reward=}, {done=}")
        # Train the model on the observed transition
        train_step(state, action, reward, next_state, done)

        # Update the current state to be the next state
        state = next_state

with open(f'fight_model_{fight_model_name}.pickle', 'wb') as f:
    pickle.dump(model, f)
