import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import keras.backend as K


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.001, strategy=""):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Increased memory size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.strategy = strategy

        # PER (Prioritized Experience Replay) parameters
        self.priority_memory = deque(maxlen=100000)
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_epsilon = 1e-6

        # Model architecture
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Training parameters
        self.batch_size = 64
        self.learning_rate_decay = 0.995
        self.min_learning_rate = 0.0001
        self.update_target_frequency = 5

    def _build_model(self):
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.action_size, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=huber_loss)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Calculate priority for experience
        priority = abs(reward) + self.priority_epsilon
        self.priority_memory.append((priority, (state, action, reward, next_state, done)))

    def act(self, state, testing=False):
        if not testing and np.random.rand() <= self.epsilon:
            # Smart exploration: bias towards promising actions
            if len(self.memory) > self.batch_size:
                q_values = self.model.predict(state)
                if np.random.rand() < 0.5:
                    return np.random.choice([np.argmax(q_values[0]),
                                             np.random.randint(self.action_size)])
            return np.random.randint(self.action_size)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.priority_memory) < batch_size:
            return

        # Calculate sampling probabilities
        priorities = np.array([x[0] for x in self.priority_memory])
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()

        # Sample batch based on priorities
        indices = np.random.choice(len(self.priority_memory), batch_size, p=probs)
        batch = [self.priority_memory[idx][1] for idx in indices]

        states = np.array([x[0][0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3][0] for x in batch])
        dones = np.array([x[4] for x in batch])

        if self.strategy == "double-dqn":
            # Double DQN update
            current_q_values = self.model.predict(next_states)
            target_q_values = self.target_model.predict(next_states)
            max_actions = np.argmax(current_q_values, axis=1)
            targets = rewards + (1 - dones) * self.gamma * \
                      target_q_values[np.arange(batch_size), max_actions]
        else:
            # Standard DQN update
            target_q_values = self.target_model.predict(next_states)
            targets = rewards + (1 - dones) * self.gamma * \
                      np.amax(target_q_values, axis=1)

        target_f = self.model.predict(states)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Decay learning rate
        if self.learning_rate > self.min_learning_rate:
            self.learning_rate *= self.learning_rate_decay
            K.set_value(self.model.optimizer.learning_rate, self.learning_rate)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath,
                                                custom_objects={'huber_loss': huber_loss})
        self.target_model = clone_model(self.model)
