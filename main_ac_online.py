import numpy as np
from utils.data_loader import load_data, preprocess_data
from utils.env import TradingEnvironment
from a2c_agent import A2CAgent
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from utils.config import *
import psutil
import matplotlib.dates as mdates

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/A2C_{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

train_data = load_data(TRAIN_DATA_PATH)
train_data = preprocess_data(train_data)
test_data = load_data(TEST_DATA_PATH)
test_data = preprocess_data(test_data)
test_data = test_data[TEST_DATA_START:]

env = TradingEnvironment(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CAgent(n_actions=action_size, actor_fc1=8, critic_fc1=8, actor_alpha=1e-4, critic_alpha=1e-4, gamma=0.95, entropy_coeff=1)

# for loading previously trained agent
dummy_state = tf.random.normal([1, 9])
agent.actor(dummy_state)
agent.critic(dummy_state)
agent.load_models()

episodes = 91
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

        if episode % 10 == 0:
            agent.save_models(str(50+episode))
            tf.keras.backend.clear_session()
        
        print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
        with summary_writer.as_default():
            tf.summary.scalar('Episode/Total_Reward', total_reward, step=episode)
            
            actions_array = np.array(actions_taken)
            for action_idx in range(action_size):
                action_freq = np.mean(actions_array == action_idx)
                tf.summary.scalar(f'Actions/Action_{action_idx}_Frequency', action_freq, step=episode)
                
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")


def plot_decisions(dates, prices, buy_points, sell_points, revenue, save=False, filename='decision_plot_a2c_online'):
    """
    Plots the price graph with buy and sell points, and optionally saves it.
    - `dates`: List of datetime objects corresponding to each price.
    - `prices`: List of prices.
    - `buy_points`: List of tuples (date, price) where buys occurred.
    - `sell_points`: List of tuples (date, price) where sells occurred.
    - `revenue`: List of cumulative revenue values corresponding to dates.
    """
    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Plot price on the left y-axis
    ax1.plot(dates, prices, label='Price', color='blue', alpha=ALPHA)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add buy and sell markers
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        ax1.scatter(buy_dates, buy_prices, color='green', label='Buy', marker='*', s=MARKER_SIZE, alpha=MARKER_ALPHA)

    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        ax1.scatter(sell_dates, sell_prices, color='red', label='Sell', marker='*', s=MARKER_SIZE, alpha=MARKER_ALPHA)

    # Set x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add grid and legend
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')

    # Plot revenue on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(dates, revenue, label='Revenue', color='orange', alpha=0.8, linestyle='--')
    ax2.set_ylabel("Revenue", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Title
    plt.title("Trading Decisions (Buy/Sell) and Revenue Over Time")

    # Save plot if required
    if save:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Graph saved at: {filename}")


def test_agent(agent, test_data, plot_name):
    env = TradingEnvironment(test_data)
    state = env.reset()
    total_reward = 0
    actions_taken = []
    done = False

    buy_points = []
    sell_points = []
    prices = []
    dates = [] 
    revenue = []
    decisions_log = []
    
    with summary_writer.as_default():
        while not done:
            action = agent.choose_action(state)
            actions_taken.append(action)
            next_state, reward, done, info = env.step(action)
            
            tf.summary.scalar('Test/Step_Reward', reward, step=env.current_step)
            tf.summary.scalar('Test/Portfolio_Value', info['portfolio_value'], step=env.current_step)
            
            state = next_state
            total_reward += reward

            current_price = test_data.iloc[env.current_step]["Close"]
            current_date = test_data.index[env.current_step]
            total_portfolio_value = env.balance + (env.shares_held * current_price)
            revenue.append(total_portfolio_value)
            dates.append(current_date)

            decision = "Hold" if action == 0 else "Buy" if action == 1 else "Sell"
            decisions_log.append((env.current_step, decision, current_price, reward))

            if action == 1:
                buy_points.append((current_date, current_price))
            elif action == 2:
                sell_points.append((current_date, current_price))

            prices.append(current_price)

        print("Trading Log:")
        print("Step | Decision | Price | Reward")
        for step, decision, price, step_reward in decisions_log:
            print(f"{step:4} | {decision:<8} | {price:.2f} | {step_reward:.2f}")

        print(f"\nTest Reward: {total_reward}")
        print(f"Final Portfolio Value: {env.balance + (env.shares_held * prices[-1])}")

        plot_decisions(dates, prices, buy_points, sell_points, revenue, save=True, filename=plot_name)
        
        tf.summary.scalar('Test/Final_Reward', total_reward, step=0)
        
        actions_array = np.array(actions_taken)
        for action_idx in range(action_size):
            action_freq = np.mean(actions_array == action_idx)
            tf.summary.scalar(f'Test/Action_{action_idx}_Frequency', action_freq, step=0)


train_agent(agent, train_data, episodes)
test_agent(agent, test_data, 'decision_plot_a2c_online')
