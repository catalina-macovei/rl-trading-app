import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow import nn

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=10,
            name='actor', chkpt_dir='tmp/actor'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.model = keras.Sequential([
            Dense(fc1_dims, activation='relu'),
            LayerNormalization(),
            # with temperature scaling, temperature = 1
            Dense(n_actions, 
                  activation=lambda x: nn.softmax(x / 1.0),
                  kernel_initializer=keras.initializers.RandomUniform(-0.01, 0.01))
        ])
        

    def call(self, state):
        return self.model(state)