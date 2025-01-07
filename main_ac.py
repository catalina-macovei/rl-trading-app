import numpy as np
from utils.data_loader import load_data, preprocess_data
from utils.environment_draft import TradingEnvironment
from a2c_agent import A2CAgent
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# Initialize TensorBoard writer
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/A2C_{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

# Load and preprocess data
data = load_data('./data/AAPL.csv')
data = preprocess_data(data)
train_data = data[:200]
test_data = data[200:250]

env = TradingEnvironment(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CAgent(n_actions=action_size)

episodes = 10
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

def train_agent(agent, train_data, episodes):
    env = TradingEnvironment(train_data)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        actions_taken = []
        
        while not done:
            # 1. a) take action a~pi_theta(a | s)
            action = agent.choose_action(state)
            actions_taken.append(action)

            # 1. b) get (s, a, s', r)
            state_, reward, done, info = env.step(action)
            total_reward += reward
            
            metrics = agent.learn(state, reward, state_, action, done)
                
            # Log training metrics per step
            with summary_writer.as_default():
                tf.summary.scalar('Metrics/Reward', reward, step=env.current_step)
                tf.summary.scalar('Metrics/Portfolio_Value', info['portfolio_value'], step=env.current_step)
                    
                if metrics:
                    tf.summary.scalar('Loss/Actor', tf.squeeze(metrics.get('actor_loss', 0)), step=env.current_step)
                    tf.summary.scalar('Loss/Critic', metrics.get('critic_loss', 0), step=env.current_step)
                    tf.summary.scalar('Policy/Entropy', metrics.get('entropy', 0), step=env.current_step)
                    tf.summary.scalar('Gradients/Actor_Norm', metrics.get('actor_grad_norm', 0), step=env.current_step)
                    tf.summary.scalar('Gradients/Critic_Norm', metrics.get('critic_grad_norm', 0), step=env.current_step)            
            
            state = state_
            
        # Log episode-level metrics
        with summary_writer.as_default():
            tf.summary.scalar('Episode/Total_Reward', total_reward, step=episode)
            
            # Log action distribution
            actions_array = np.array(actions_taken)
            for action_idx in range(action_size):
                action_freq = np.mean(actions_array == action_idx)
                tf.summary.scalar(f'Actions/Action_{action_idx}_Frequency', action_freq, step=episode)
                
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")


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
    plt.savefig("./graphs/demo.png")


def test_agent(agent, test_data):
    env = TradingEnvironment(test_data)
    state = env.reset()
    total_reward = 0
    actions_taken = []
    done = False

    buy_points = []  # Track buy points for graph
    sell_points = []  # Track sell points for graph
    prices = []  # Track prices for the graph
    decisions_log = []  # Log decisions per episode
    
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


train_agent(agent, train_data, episodes)
test_agent(agent, test_data)
