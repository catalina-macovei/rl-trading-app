import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='critic', chkpt_dir='tmp/critic'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_c')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.ln1 = LayerNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.ln2 = LayerNormalization()
        self.v = Dense(1, activation=None)


    def call(self, state):
        value = self.fc1(state)
        value = self.ln1(value)
        value = self.fc2(value)
        value = self.ln2(value)

        v = self.v(value)

        return v