import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
import tensorflow_probability as tfp
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import os

class A2CAgent:
    def __init__(self, critic_alpha=1e-4, actor_alpha=1e-4, gamma=0.99, entropy_coeff=0.5, max_grad_norm=0.5, n_actions=3, critic_fc1=10, actor_fc1=10):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=actor_fc1, name="online")
        self.critic = CriticNetwork(n_actions=n_actions, fc1_dims=critic_fc1, name="online")

        sample_state = tf.random.normal([1, 9])
        self.actor(sample_state)
        self.critic(sample_state)

        self.actor.compile(optimizer=AdamW(learning_rate=actor_alpha, weight_decay=1e-5))
        self.critic.compile(optimizer=AdamW(learning_rate=critic_alpha, weight_decay=1e-5))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation]) # add an extra dimension (the neural network expects a batch)
        probs = self.actor(state)          
        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True) 

        # choose the action according to the distribution returned by the actor network
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        return action.numpy()[0]

    def save_models(self, episode_no=''):
        print('... saving models ...')
        try:
            if not os.path.exists(self.actor.checkpoint_dir):
                os.makedirs(self.actor.checkpoint_dir)
            if not os.path.exists(self.critic.checkpoint_dir):
                os.makedirs(self.critic.checkpoint_dir)

            actor_path = os.path.join(self.actor.checkpoint_dir, 'actor_checkpoint_online_' + episode_no)
            critic_path = os.path.join(self.critic.checkpoint_dir, 'critic_checkpoint_online_' + episode_no)
            
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            
            print(f"Models saved to:\n{actor_path}\n{critic_path}")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")

    def load_models(self):
        print('... loading models ...')
        print('... loading models ...')
        try:
            if not os.path.exists(self.actor.checkpoint_dir):
                os.makedirs(self.actor.checkpoint_dir)
            if not os.path.exists(self.critic.checkpoint_dir):
                os.makedirs(self.critic.checkpoint_dir)

            print("Loading actor weights...")
            actor_checkpoint = tf.train.latest_checkpoint(self.actor.checkpoint_dir)
            if actor_checkpoint:
                self.actor.load_weights(actor_checkpoint)
            else:
                raise FileNotFoundError("No actor checkpoint found")

            print("Loading critic weights...")
            critic_checkpoint = tf.train.latest_checkpoint(self.critic.checkpoint_dir)
            if critic_checkpoint:
                self.critic.load_weights(critic_checkpoint)
            else:
                raise FileNotFoundError("No critic checkpoint found")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        
    def learn(self, state, reward, next_state, action, done):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # doesn't need an extra dimension, because isn't fed into the network
        
        # GradientTape allows us to calculate maually the gradients
        with tf.GradientTape(persistent=True) as tape:

            # 2. calculate V_hat_pi_theta(s) and V_hat_pi_theta(s')
            state_value = self.critic(state)
            next_state_value = self.critic(next_state)

            # for the loss function, a scalar is better
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            # 3. evaluate A_hat_pi(s, a) = r(s, a) + gamma * V_hat_pi_theta(s') - V_hat_pi_theta(s')
            advantage = reward + self.gamma*next_state_value*(1-int(done)) - state_value # if it is a terminal state, we don't have a next_state_value

            # 4. calculate log probabilitiy: log pi_theta(a | s)
            probs = self.actor(state)
            probs = tf.clip_by_value(probs, 1e-10, 1.0)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            # encourage exploring with entropy
            entropy = tf.reduce_mean(action_probs.entropy())

            # 4. calculate log(pi_theta(a | s)) * A_hat_pi(s, a) (adjusted with the entropy)
            actor_loss = -log_prob * tf.stop_gradient(advantage) - self.entropy_coeff * entropy

            # 2. update V_hat_pi_theta using target r + gamma * V_hat_pi_theta(s')
            # MSE = 0.5 * (target - predict)^2 = 0.5 * (r + gamma * V_hat_pi_theta(s') - V_hat_pi_theta(s))^2 = 0.5 * advantage^2
            critic_loss = 0.5 * tf.square(advantage)

        # 2. + 4. calculate gradients for updating the actor and the critic
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables) # delta_theta J(theta)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)

        # gradient clipping for avoiding exploding gradients
        actor_gradients, actor_grad_norm = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
        critic_gradients, critic_grad_norm = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)

        # 2. + 4. backpropagation for the actor and the critic
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables)) # theta <- theta + alpha * delta_theta J(theta)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        del tape
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }