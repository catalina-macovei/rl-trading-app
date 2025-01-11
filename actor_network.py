import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='actor', chkpt_dir='tmp/actor'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.ln1 = LayerNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.ln2 = LayerNormalization()

        # Temperature parameter for softmax; without temperature scaling, it chooses the same action almost every time
        self.temperature = 1.0

        self.pi = Dense(n_actions, 
                       activation='softmax',
                       kernel_initializer=keras.initializers.RandomUniform(-0.01, 0.01))

    def call(self, state):
        value = self.fc1(state)
        value = self.ln1(value)
        value = self.fc2(value)
        value = self.ln2(value)

        # Apply temperature scaling
        logits = self.pi(value) / self.temperature
        return logits