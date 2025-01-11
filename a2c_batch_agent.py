import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from actor_network import ActorNetwork
from critic_network import CriticNetwork

class A2CBatchAgent:
    def __init__(self, critic_alpha=1e-4, actor_alpha=1e-4, gamma=0.99, entropy_coeff=0.5, max_grad_norm=0.5, n_actions=2, critic_fc1=1024, critic_fc2=512, actor_fc1=1024, actor_fc2=512):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=actor_fc1, fc2_dims=actor_fc2, name="batch")
        self.critic = CriticNetwork(n_actions=n_actions, fc1_dims=critic_fc1, fc2_dims=critic_fc2, name="batch")

        self.actor.compile(optimizer=Adam(learning_rate=actor_alpha))
        self.critic.compile(optimizer=Adam(learning_rate=critic_alpha))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation]) # add an extra dimension (the neural network expects a batch)
        probs = self.actor(state)          
        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True) 

        # choose the action according to the distribution returned by the actor network
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        
    def learn(self, states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch):

        states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
        next_states_batch = tf.convert_to_tensor(next_states_batch, dtype=tf.float32)
        actions_batch = tf.convert_to_tensor(actions_batch, dtype=tf.float32)
        rewards_batch = tf.convert_to_tensor(rewards_batch, dtype=tf.float32)
        dones_batch = tf.convert_to_tensor(dones_batch, dtype=tf.float32)

        # GradientTape allows us to calculate manually the gradients
        with tf.GradientTape(persistent=True) as tape:

            # 2. calculate V_hat_pi_theta(s) and V_hat_pi_theta(s') for all samples in batch
            state_values = self.critic(states_batch)
            next_state_values = self.critic(next_states_batch)

            # for the loss function, scalars are better
            state_values = list(map(tf.squeeze, state_values))
            next_state_values = list(map(tf.squeeze, next_state_values))

             # 3. evaluate A_hat_pi(s_i, a_i) = r(s_i, a_i) + gamma * V_hat_pi_theta(s_i') - V_hat_pi_theta(s_i')
            advantages = [rewards_batch[i] + self.gamma * next_state_values[i] * (1 - dones_batch[i]) - state_values[i] for i in range(len(states_batch))]

            log_probs = []
            entropies = []

            # 4. calculate log probabilities: log pi_theta(a_i | s_i)
            probs_batch = self.actor(states_batch)
            
            for i in range(len(probs_batch)):
                
                probs = tf.clip_by_value(probs_batch[i], 1e-10, 1.0)
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions_batch[i])
                log_probs.append(log_prob)

                # encourage exploring with entropy
                entropy = tf.reduce_mean(action_probs.entropy())
                entropies.append(entropy)

            # 4. calculate log(pi_theta(a_i | s_i)) * A_hat_pi(s_i, a_i) (adjusted with the entropy)
            actor_losses = [(-1) * log_probs[i] * tf.stop_gradient(advantages[i]) - self.entropy_coeff * entropies[i] for i in range(len(advantages))]
            actor_losses_sum = sum(actor_losses)

            # 2. update V_hat_pi_theta using target r + gamma * V_hat_pi_theta(s')
            # MSE = 0.5 * (target - predict)^2 = 0.5 * (r + gamma * V_hat_pi_theta(s') - V_hat_pi_theta(s))^2 = 0.5 * advantage^2
            critic_losses = 0.5 * tf.square(advantages)
            critic_losses_sum = sum(critic_losses)


        # 2. + 4. calculate gradients for updating the actor and the critic
        actor_gradients = tape.gradient(actor_losses_sum, self.actor.trainable_variables) # delta_theta J(theta)
        critic_gradients = tape.gradient(critic_losses_sum, self.critic.trainable_variables)

        # gradient clipping for avoiding exploding gradients
        actor_gradients, actor_grad_norm = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
        critic_gradients, critic_grad_norm = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)

        # 2. + 4. backpropagation for the actor and the critic
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables)) # theta <- theta + alpha * delta_theta J(theta)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        del tape
        
        return {
            'actor_loss': actor_losses_sum,
            'critic_loss': critic_losses_sum,
            'entropy': entropies,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }