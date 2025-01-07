import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from actor_network import ActorNetwork
from critic_network import CriticNetwork

class A2CAgent:
    def __init__(self, alpha=1e-4, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions)
        self.critic = CriticNetwork(n_actions=n_actions)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation]) # add an extra dimension (the neural network expects a batch)
        probs = self.actor(state)          
        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True) 

        # choose the action according to the distribution returned by the actor network
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        
    def learn(self, state, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # doesn't need an extra dimension, because isn't fed into the network
        
        # GradientTape allows us to calculate maually the gradients
        with tf.GradientTape(persistent=True) as tape:
            state_value = self.critic(state)
            next_state_value = self.critic(next_state)

            # for the loss function, a scalar is better
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            probs = self.actor(state)

            probs = tf.clip_by_value(probs, 1e-10, 1.0)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
            
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            advantage = reward + self.gamma*next_state_value*(1-int(done)) - state_value # if it is a terminal state, we don't have a next_state_value
            
            # encourage exploring with entropy
            entropy_coeff = 0.5
            entropy = tf.reduce_mean(action_probs.entropy())

            actor_loss = -log_prob * tf.stop_gradient(advantage) - entropy_coeff * entropy
            critic_loss = 0.5 * tf.square(advantage)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)

        # gradient clipping for avoiding exploding gradients
        max_grad_norm = 0.5
        actor_gradients, actor_grad_norm = tf.clip_by_global_norm(actor_gradients, max_grad_norm)
        critic_gradients, critic_grad_norm = tf.clip_by_global_norm(critic_gradients, max_grad_norm)

        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        del tape
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }