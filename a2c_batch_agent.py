import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
import tensorflow_probability as tfp
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import os
import psutil

class A2CBatchAgent:
    def __init__(self, critic_alpha=1e-4, actor_alpha=1e-4, gamma=0.99, entropy_coeff=0.5, max_grad_norm=0.5, n_actions=2, critic_fc1=10, actor_fc1=10):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=actor_fc1, name="batch")
        self.critic = CriticNetwork(n_actions=n_actions, fc1_dims=critic_fc1, name="batch")

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

            actor_path = os.path.join(self.actor.checkpoint_dir, 'actor_checkpoint' + '_' + episode_no)
            critic_path = os.path.join(self.critic.checkpoint_dir, 'critic_checkpoint' + '_' + episode_no)
            
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            
            print(f"Models saved to:\n{actor_path}\n{critic_path}")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")

    def load_models(self):
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
        
    @tf.function
    def learn(self, states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch):

        """
        initially, we calculated all for each batch of trajectories, but the training consumed too many resources, 
        so now we make the calculations once for all batches and trajectories
        """

        print("Memory before gradient computation:", psutil.Process().memory_info().rss / 1024 / 1024)

        with tf.GradientTape(persistent=True) as tape:

            states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
            next_states_batch = tf.convert_to_tensor(next_states_batch, dtype=tf.float32)
            actions_batch = tf.convert_to_tensor(actions_batch, dtype=tf.int32)
            rewards_batch = tf.convert_to_tensor(rewards_batch, dtype=tf.float32)
            dones_batch = tf.convert_to_tensor(dones_batch, dtype=tf.float32)

            batch_size = tf.shape(states_batch)[0]
            max_seq_length = tf.shape(states_batch)[1]
            
            # we process all the trajectories at the same time, so we need them reshaped
            states_reshaped = tf.reshape(states_batch, [-1, tf.shape(states_batch)[-1]])
            next_states_reshaped = tf.reshape(next_states_batch, [-1, tf.shape(next_states_batch)[-1]])
            
            # predict the V_hat values
            state_values = self.critic(states_reshaped)
            next_state_values = self.critic(next_states_reshaped)
            
            # reshape back, so to have them in batches
            state_values = tf.reshape(state_values, [batch_size, max_seq_length])
            next_state_values = tf.reshape(next_state_values, [batch_size, max_seq_length])
            
            # 3. evaluate A_hat_pi(s_i, a_i) = r(s_i, a_i) + gamma * V_hat_pi_theta(s_i') - V_hat_pi_theta(s_i)
            # to optimize, we compute all batches at once
            advantages = rewards_batch + self.gamma * next_state_values * (1 - dones_batch) - state_values
            
            # predict action probability distribution to find out the log probabilities
            probs_batch = self.actor(states_reshaped)
            probs_batch = tf.reshape(probs_batch, [batch_size, max_seq_length, -1])
            
            probs_batch = tf.clip_by_value(probs_batch, 1e-10, 1.0)
            probs_batch = probs_batch / tf.reduce_sum(probs_batch, axis=-1, keepdims=True)
            
            action_distributions = tfp.distributions.Categorical(probs=probs_batch)
            actions_batch = tf.clip_by_value(actions_batch, 0, 2)
            log_probs = action_distributions.log_prob(actions_batch)
            entropies = action_distributions.entropy()
            
            # losses for each trajectory
            actor_losses = -log_probs * tf.stop_gradient(advantages) - self.entropy_coeff * entropies
            critic_losses = 0.5 * tf.square(advantages)

            # losses for the entire batch
            # L(phi) = 0.5 * sum_i || V_hat_pi_phi(s_i) - y_i||^2; y_i = the target = r(s_i, a_i) + gamma * V_hat_pi_phi(s_i')
            # grad_theta J(theta) approx. = sum_i grad_theta log pi_theta(a_i, s_i) A_hat_pi(s_i, a_i)
            actor_loss = tf.reduce_sum(actor_losses)
            critic_loss = tf.reduce_sum(critic_losses)

        # 2. + 4. calculate gradients for updating the actor and the critic
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)

        # gradient clipping for avoiding exploding gradients
        actor_gradients, actor_grad_norm = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
        critic_gradients, critic_grad_norm = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)

        # 2. + 4. backpropagation for the actor and the critic
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        del tape
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }