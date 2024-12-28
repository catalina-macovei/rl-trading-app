import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras.backend as K


# Huber loss definition (for stability in training)
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta  # Use tf.abs instead of K.abs
    squared_loss = 0.5 * tf.square(error)
    linear_loss = clip_delta * (tf.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, pretrained=False, model_name=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate (random action selection)
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model = self._build_model()

        # Target model for stability
        self.target_model = self._build_model()  # Initialize target model
        self.target_model.set_weights(self.model.get_weights())  # Initially set the same weights as the model

    def _build_model(self):
        """Builds the neural network model."""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=huber_loss)
        return model

    def act(self, state):
        """Select an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Exploration
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploitation: take the action with the highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train the agent using a random sample of the experience replay memory."""
        if len(self.memory) < batch_size:
            return  # If memory is not enough to sample, do nothing

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Calculate the target for the current state-action pair using the target model
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

            # Get the current Q-values for the state
            target_f = self.model.predict(state)
            target_f[0][action] = target  # Update the Q-value for the current action

            # Fit the model with the new target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Reduce epsilon to encourage more exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """Update the target model's weights to match the current model's weights."""
        self.target_model.set_weights(self.model.get_weights())

    def save(self, episode):
        """Save the model to a file."""
        self.model.save(f"models/{self.model_name}_{episode}")

    def load(self):
        """Load a pretrained model from a file."""
        self.model = tf.keras.models.load_model(f"models/{self.model_name}", custom_objects={"huber_loss": huber_loss})
        self.target_model.set_weights(self.model.get_weights())  # Set target model to match the pretrained model

