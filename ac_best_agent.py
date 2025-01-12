import pandas as pd
import numpy as np
from utils.environment_draft import TradingEnvironment
from a2c_batch_agent import A2CBatchAgent
from a2c_agent import A2CAgent
from utils.data_loader import load_data, preprocess_data
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

def calculate_sharpe(df: pd.DataFrame):
    df["daily_return"] = df["account_values"].pct_change(1)
    if df["daily_return"].std() != 0:
        sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
        return sharpe
    else:
        return 0

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/A2C_test_{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

def plot_decisions(prices, buy_points, sell_points, plot_name):
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
    plt.savefig("./graphs/"+plot_name+".png")


# Load and preprocess data
data = load_data('./data/AAPL.csv')
data = preprocess_data(data)
test_data = data[1000:]

def run_agent(agent, test_data, plot_name):
    env = TradingEnvironment(test_data)
    state = env.reset()
    total_reward = 0
    actions_taken = []
    done = False

    buy_points = []  # Track buy points for graph
    sell_points = []  # Track sell points for graph
    prices = []  # Track prices for the graph
    decisions_log = []  # Log decisions per episode

    df = pd.DataFrame(columns=["daily_return", "account_values"])
    
    with summary_writer.as_default():
        while not done:
            action = agent.choose_action(state)
            actions_taken.append(action)
            next_state, reward, done, info = env.step(action)

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

            next_price = env.data.iloc[env.current_step]['Close'] if not done else env.data.iloc[env.current_step - 1]['Close']
            portfolio_value_after = env.balance + (env.shares_held * next_price)
            # df.loc[len(df)] = [portfolio_value_after, 0]
            # sharpe = calculate_sharpe(df)

            tf.summary.scalar('Test/Total Portfolio Value/'+plot_name, portfolio_value_after, step=env.current_step-1)
            # tf.summary.scalar('Test/Sharpe Ratio/'+plot_name, sharpe, step=env.current_step - 1)


        # Print decisions and rewards for each step
        print("Trading Log:")
        print("Step | Decision | Price | Reward")
        for step, decision, price, step_reward in decisions_log:
            print(f"{step:4} | {decision:<8} | {price:.2f} | {step_reward:.2f}")

        # Print final reward and portfolio value
        print(f"\nTest Reward: {total_reward}")
        print(f"Final Portfolio Value: {env.balance + (env.shares_held * prices[-1])}")

        # Call function to plot the graph
        plot_decisions(prices, buy_points, sell_points, plot_name)

online_agent = A2CAgent(actor_fc1=512, actor_fc2=256, critic_fc1=512, critic_fc2=256, n_actions=3)
dummy_state = tf.random.normal([1, 9])
online_agent.actor(dummy_state)
online_agent.critic(dummy_state)
online_agent.load_models()

# batch_agent = A2CBatchAgent(actor_fc1=512, actor_fc2=256, critic_fc1=512, critic_fc2=256, n_actions=3)
# dummy_state = tf.random.normal([1, 9])
# batch_agent.actor(dummy_state)
# batch_agent.critic(dummy_state)
# batch_agent.load_models()

run_agent(online_agent, test_data, "online")
# run_agent(batch_agent, test_data, "batch")