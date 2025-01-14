import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization
import tensorflow as tf

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=10,
            name='critic', chkpt_dir='tmp/critic'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.model = keras.Sequential([
            Dense(fc1_dims, activation='relu'),
            Dense(1, activation=None)
        ])


    def call(self, state):
        return self.model(state)