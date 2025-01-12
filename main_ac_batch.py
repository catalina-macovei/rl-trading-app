import numpy as np
from utils.data_loader import load_data, preprocess_data
# from utils.environment_draft import TradingEnvironment
from utils.env import TradingEnvironment
from a2c_batch_agent import A2CBatchAgent
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import gc
from utils.config import *

# Initialize TensorBoard writer
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/A2C_{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

# Load and preprocess data
train_data = load_data(TRAIN_DATA_PATH)
train_data = preprocess_data(train_data)
test_data = load_data(TEST_DATA_PATH)
test_data = preprocess_data(test_data)
test_data = test_data[TEST_DATA_START:]

env = TradingEnvironment(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CBatchAgent(n_actions=action_size, critic_fc1=10, actor_fc1=10, critic_alpha=0.001, actor_alpha=0.001, gamma=0.95, entropy_coeff=1)

episodes = 500
best_score = env.reward_range[0]
score_history = []

def train_agent(agent, train_data, episodes, batch_size=32):
    env = TradingEnvironment(train_data)

    for episode in tqdm(range(episodes)):
        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        max_steps = 50

        # generate a batch of trajectories
        states_batch = [[] for _ in range(batch_size)]
        next_states_batch = [[] for _ in range(batch_size)]
        rewards_batch = [[] for _ in range(batch_size)]
        actions_batch = [[] for _ in range(batch_size)]
        dones_batch = [[] for _ in range(batch_size)]

        for traj in range(batch_size):
            state = env.reset()
            done = False
            step = 0

            while not done or step < max_steps:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)

                states_batch[traj].append(state)
                next_states_batch[traj].append(next_state)
                actions_batch[traj].append(action)
                rewards_batch[traj].append(reward)
                dones_batch[traj].append(done)
                    
                state = next_state
                step += 1

        states_np = np.array(states_batch, dtype=np.float32)
        next_states_np = np.array(next_states_batch, dtype=np.float32)
        actions_np = np.array(actions_batch, dtype=np.int32)
        rewards_np = np.array(rewards_batch, dtype=np.float32)
        dones_np = np.array(dones_batch, dtype=np.float32)

        # train the agent on the batch using numpy arrays
        metrics = agent.learn(states_np, next_states_np, actions_np, rewards_np, dones_np)

        # Clear memory
        del states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch
        del states_np, next_states_np, actions_np, rewards_np, dones_np
        gc.collect()
        
        if episode % 1 == 0:
            agent.save_models(str(episode))
            tf.keras.backend.clear_session()

        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
                        
        # Log training metrics
        with summary_writer.as_default():
            tf.summary.scalar('Metrics/Reward', reward, step=env.current_step)
            tf.summary.scalar('Metrics/Portfolio_Value', info['portfolio_value'], step=env.current_step)
                            
            if metrics:
                tf.summary.scalar('Loss/Actor', metrics.get('actor_loss', 0), step=env.current_step)
                tf.summary.scalar('Loss/Critic', metrics.get('critic_loss', 0), step=env.current_step)
                tf.summary.scalar('Gradients/Actor_Norm', metrics.get('actor_grad_norm', 0), step=env.current_step)
                tf.summary.scalar('Gradients/Critic_Norm', metrics.get('critic_grad_norm', 0), step=env.current_step)            


def plot_decisions(prices, buy_points, sell_points):
    """
    Plots the price graph with buy and sell points.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label='Price', color='blue', alpha=0.6)

    if buy_points:
        buy_x, buy_y = zip(*buy_points)  # Unzip buy points
        plt.scatter(buy_x, buy_y, color='green', label='Buy', marker='^')

    if sell_points:
        sell_x, sell_y = zip(*sell_points)  # Unzip sell points
        plt.scatter(sell_x, sell_y, color='red', label='Sell', marker='v')

    plt.title("Trading Decisions (Buy/Sell) Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("./graphs/ac_batch.png")


def test_agent(agent, test_data):
    env = TradingEnvironment(test_data)
    state = env.reset()
    total_reward = 0
    actions_taken = []
    done = False

    buy_points = []
    sell_points = []
    prices = [] 
    decisions_log = []
    
    with summary_writer.as_default():
        while not done:
            action = agent.choose_action(state)
            actions_taken.append(action)
            next_state, reward, done, info = env.step(action)
            
            # Log test metrics
            tf.summary.scalar('Test/Step_Reward', reward, step=env.current_step)
            tf.summary.scalar('Test/Portfolio_Value', info['portfolio_value'], step=env.current_step)
            
            state = next_state
            total_reward += reward

            # Log decisions and rewards for each step
            current_price = test_data.iloc[env.current_step]["Close"]
            decision = "Hold" if action == 0 else "Buy" if action == 1 else "Sell"
            decisions_log.append((env.current_step, decision, current_price, reward))

            # Store action points for graph
            if action == 1:  # Buy
                buy_points.append((env.current_step, current_price))
            elif action == 2:  # Sell
                sell_points.append((env.current_step, current_price))

            prices.append(current_price)

        # Print decisions and rewards for each step
        print("Trading Log:")
        print("Step | Decision | Price | Reward")
        for step, decision, price, step_reward in decisions_log:
            print(f"{step:4} | {decision:<8} | {price:.2f} | {step_reward:.2f}")

        # Print final reward and portfolio value
        print(f"\nTest Reward: {total_reward}")
        print(f"Final Portfolio Value: {env.balance + (env.shares_held * prices[-1])}")

        # Call function to plot the graph
        plot_decisions(prices, buy_points, sell_points)
        
        # Log final test metrics
        tf.summary.scalar('Test/Final_Reward', total_reward, step=0)
        
        # Log test action distribution
        actions_array = np.array(actions_taken)
        for action_idx in range(action_size):
            action_freq = np.mean(actions_array == action_idx)
            tf.summary.scalar(f'Test/Action_{action_idx}_Frequency', action_freq, step=0)


train_agent(agent, train_data, episodes, batch_size=12)
test_agent(agent, test_data)

