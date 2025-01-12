import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import os
import psutil

class A2CBatchAgent:
    def __init__(self, critic_alpha=1e-4, actor_alpha=1e-4, gamma=0.99, entropy_coeff=0.5, max_grad_norm=0.5, n_actions=2, critic_fc1=1024, critic_fc2=512, actor_fc1=1024, actor_fc2=512):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]

        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=actor_fc1, fc2_dims=actor_fc2, name="batch")
        self.critic = CriticNetwork(n_actions=n_actions, fc1_dims=critic_fc1, fc2_dims=critic_fc2, name="batch")

        input_shape = (None, 9)
        self.actor.build(input_shape)
        self.critic.build(input_shape)

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
        try:
            if not os.path.exists(self.actor.checkpoint_dir):
                os.makedirs(self.actor.checkpoint_dir)
            if not os.path.exists(self.critic.checkpoint_dir):
                os.makedirs(self.critic.checkpoint_dir)

            # Save weights with explicit naming
            actor_path = os.path.join(self.actor.checkpoint_dir, 'actor_checkpoint')
            critic_path = os.path.join(self.critic.checkpoint_dir, 'critic_checkpoint')
            
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            
            print(f"Models saved to:\n{actor_path}\n{critic_path}")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_models(self):
        print('... loading models ...')
        try:
            if not os.path.exists(self.actor.checkpoint_dir):
                os.makedirs(self.actor.checkpoint_dir)
            if not os.path.exists(self.critic.checkpoint_dir):
                os.makedirs(self.critic.checkpoint_dir)

            # Load weights piece by piece
            print("Loading actor weights...")
            actor_checkpoint = tf.train.latest_checkpoint(self.actor.checkpoint_dir)
            if actor_checkpoint:
                print(f"Found actor checkpoint: {actor_checkpoint}")
                status = self.actor.load_weights(actor_checkpoint)
                status.expect_partial()  # Suppress warnings about optimizer states
            else:
                raise FileNotFoundError("No actor checkpoint found")

            print("Loading critic weights...")
            critic_checkpoint = tf.train.latest_checkpoint(self.critic.checkpoint_dir)
            if critic_checkpoint:
                print(f"Found critic checkpoint: {critic_checkpoint}")
                status = self.critic.load_weights(critic_checkpoint)
                status.expect_partial()  # Suppress warnings about optimizer states
            else:
                raise FileNotFoundError("No critic checkpoint found")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def learn(self, states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch):
        print("Memory before gradient computation:", psutil.Process().memory_info().rss / 1024 / 1024)
        # GradientTape allows us to calculate manually the gradients
        with tf.GradientTape(persistent=True) as tape:

            actor_losses = []
            critic_losses = []
            batch_size = len(states_batch)

            max_seq_length = max(len(seq) for seq in states_batch)

            for i in range(batch_size):

                states_batch_i = tf.convert_to_tensor(states_batch[i], dtype=tf.float32)
                next_states_batch_i = tf.convert_to_tensor(next_states_batch[i], dtype=tf.float32)
                actions_batch_i = tf.convert_to_tensor(actions_batch[i], dtype=tf.float32)
                rewards_batch_i = tf.convert_to_tensor(rewards_batch[i], dtype=tf.float32)
                dones_batch_i = tf.convert_to_tensor(dones_batch[i], dtype=tf.float32)

                # 2. calculate V_hat_pi_theta(s) and V_hat_pi_theta(s') for all samples in batch
                state_values = self.critic(states_batch_i)
                next_state_values = self.critic(next_states_batch_i)

                # for the loss function, scalars are better
                state_values = list(map(tf.squeeze, state_values))
                next_state_values = list(map(tf.squeeze, next_state_values))

                # 3. evaluate A_hat_pi(s_i, a_i) = r(s_i, a_i) + gamma * V_hat_pi_theta(s_i') - V_hat_pi_theta(s_i')
                advantages_i = [rewards_batch_i[i] + self.gamma * next_state_values[i] * (1 - dones_batch_i[i]) - state_values[i] for i in range(len(states_batch_i))]

                log_probs_i = []
                entropies_i = []

                # 4. calculate log probabilities: log pi_theta(a_i | s_i)
                probs_batch_i = self.actor(states_batch_i)
                
                for i in range(len(probs_batch_i)):
                    
                    probs = tf.clip_by_value(probs_batch_i[i], 1e-10, 1.0)
                    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                    action_probs = tfp.distributions.Categorical(probs=probs)
                    log_prob = action_probs.log_prob(actions_batch_i[i])
                    log_probs_i.append(log_prob)

                    # encourage exploring with entropy
                    entropy = tf.reduce_mean(action_probs.entropy())
                    entropies_i.append(entropy)

                # 4. calculate log(pi_theta(a_i | s_i)) * A_hat_pi(s_i, a_i) (adjusted with the entropy)
                actor_losses_i = [(-1) * log_probs_i[i] * tf.stop_gradient(advantages_i[i]) - self.entropy_coeff * entropies_i[i] for i in range(len(advantages_i))]
                actor_losses_sum_i = sum(actor_losses_i)
                actor_losses.append(actor_losses_sum_i)

                # 2. update V_hat_pi_theta using target r + gamma * V_hat_pi_theta(s')
                # MSE = 0.5 * (target - predict)^2 = 0.5 * (r + gamma * V_hat_pi_theta(s') - V_hat_pi_theta(s))^2 = 0.5 * advantage^2
                critic_losses_i = 0.5 * tf.square((advantages_i))
                critic_losses_sum_i = sum(critic_losses_i)
                critic_losses.append(critic_losses_sum_i)

                del states_batch_i
                del actions_batch_i
                del next_states_batch_i
                del rewards_batch_i
                del dones_batch_i

            actor_losses_sum = sum(actor_losses) / batch_size
            critic_losses_sum = sum(critic_losses)


        # 2. + 4. calculate gradients for updating the actor and the critic
        actor_gradients = tape.gradient(actor_losses_sum, self.actor.trainable_variables) # delta_theta J(theta)
        critic_gradients = tape.gradient(critic_losses_sum, self.critic.trainable_variables)

        # gradient clipping for avoiding exploding gradients
        actor_gradients, actor_grad_norm = tf.clip_by_global_norm(actor_gradients, self.max_grad_norm)
        critic_gradients, critic_grad_norm = tf.clip_by_global_norm(critic_gradients, self.max_grad_norm)

        print("Memory after gradient computation:", psutil.Process().memory_info().rss / 1024 / 1024)

        # 2. + 4. backpropagation for the actor and the critic
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables)) # theta <- theta + alpha * delta_theta J(theta)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        print("Memory after network update:", psutil.Process().memory_info().rss / 1024 / 1024)

        del tape
        
        return {
            'actor_loss': actor_losses_sum,
            'critic_loss': critic_losses_sum,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }