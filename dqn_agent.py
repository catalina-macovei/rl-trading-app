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

        self.priority_memory = []
        self.priority_weight = 0.6

    def _build_model(self):
        """Builds the neural network model."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=huber_loss)
        return model

    def act(self, state):
        """Select action using epsilon-greedy with price trend consideration"""
        if np.random.rand() <= self.epsilon:
            # Smart random selection based on price trend
            price_change = state[0][3] - state[0][0]  # Close - Open price
            if price_change > 0:
                # Upward trend - prefer buy or hold
                return np.random.choice([0, 1], p=[0.4, 0.6])
            else:
                # Downward trend - prefer sell or hold
                return np.random.choice([0, 2], p=[0.4, 0.6])

        # Get Q-values from model prediction
        q_values = self.model.predict(state)

        # Add risk management
        action = np.argmax(q_values[0])

        # Risk management checks
        portfolio_value = state[0][-1]  # Assuming last state value is portfolio value
        max_loss_threshold = -0.05  # 5% max loss

        if action == 1:  # Buy
            potential_loss = (state[0][3] - state[0][2]) / state[0][3]  # (Close - Low) / Close
            if potential_loss > max_loss_threshold:
                return 0  # Hold instead

        if action == 2:  # Sell
            potential_gain = (state[0][1] - state[0][3]) / state[0][3]  # (High - Close) / Close
            if potential_gain < 0.02:  # 2% minimum gain
                return 0  # Hold instead

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        # Calculate priority based on reward magnitude
        priority = abs(reward) + 0.01  # Small constant to avoid zero priority
        self.priority_memory.append((priority, (state, action, reward, next_state, done)))
        if len(self.priority_memory) > 10000:  # Memory limit
            self.priority_memory = self.priority_memory[1:]

    def replay(self, batch_size):
        """Train the agent using a random sample of the experience replay memory."""
        if len(self.memory) < batch_size:
            return  # If memory is not enough to sample, do nothing

        # Sort by priority and select top experiences
        self.priority_memory.sort(key=lambda x: x[0], reverse=True)
        priority_batch = self.priority_memory[:int(batch_size * self.priority_weight)]
        random_batch = random.sample(self.priority_memory[int(batch_size * self.priority_weight):],
                                     int(batch_size * (1 - self.priority_weight)))

        minibatch = [item[1] for item in priority_batch + random_batch]

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

