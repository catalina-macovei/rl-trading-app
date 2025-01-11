
import pandas as pd
from utils.data_loader import load_data, preprocess_data
#from utils.environment import TradingEnvironment
from utils.env_draft import TradingEnvironment
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import MaxNLocator
import numpy as np

# Enhanced training parameters
EPISODES = 1000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5
VALIDATION_INTERVAL = 50

# Load and preprocess data
train_data = load_data('./data/AAPL.csv')
train_data = preprocess_data(train_data)
test_data = load_data('./data/GOOG.csv')
test_data = preprocess_data(test_data)
size = 1000
train_data = train_data[:size]
test_data = test_data[:size]

# Initialize environment and agent
env = TradingEnvironment(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("action=", action_size)
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    learning_rate=0.001
)

# Track metrics
training_rewards = []
portfolio_values = []
best_reward = float('-inf')


def plot_decisions(prices, buy_points, sell_points, dates, balances, save=False, filename='decision_plot.png', label_step=5):
    """
    Enhanced plotting function with reduced buy/sell labels for clarity.
    Parameters:
        prices: List of prices.
        buy_points: List of tuples [(index, price)] for buy decisions.
        sell_points: List of tuples [(index, price)] for sell decisions.
        dates: List of corresponding dates.
        balances: List of balance values over time.
        save: Whether to save the plot.
        filename: Name of the file to save the plot to.
        label_step: Interval for subsampling buy/sell annotations (e.g., every nth point).
    """
    fig, ax1 = plt.subplots(figsize=(20, 10))  # Larger figure size

    # Plot price line
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', color='blue', fontsize=12)
    ax1.plot(dates, prices, label='Price', color='blue', alpha=0.6, linewidth=2)

    # Plot buy points with reduced labels
    if buy_points:
        buy_x, buy_y = zip(*buy_points)
        buy_dates = [dates[i] for i in buy_x]
        ax1.scatter(buy_dates, buy_y, color='green', label='Buy', marker='^', s=200)

        # Label only every nth buy point
        for idx, (date, price) in enumerate(zip(buy_dates, buy_y)):
            if idx % label_step == 0:  # Label every nth point
                ax1.annotate(f'Buy\n${price:.2f}', (date, price), textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=8, color='green')

    # Plot sell points with reduced labels
    if sell_points:
        sell_x, sell_y = zip(*sell_points)
        sell_dates = [dates[i] for i in sell_x]
        ax1.scatter(sell_dates, sell_y, color='red', label='Sell', marker='v', s=200)

        # Label only every nth sell point
        for idx, (date, price) in enumerate(zip(sell_dates, sell_y)):
            if idx % label_step == 0:  # Label every nth point
                ax1.annotate(f'Sell\n${price:.2f}', (date, price), textcoords="offset points", xytext=(0, -15),
                             ha='center', fontsize=8, color='red')

    # Add balance line on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Balance ($)', color='orange', fontsize=12)
    ax2.plot(dates, balances, label='Balance', color='orange', linestyle='--', linewidth=2)

    # Improve x-axis readability
    locator = AutoDateLocator(maxticks=10)  # Limit the number of ticks
    formatter = ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    # Enhanced legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Title and styling
    plt.title("Trading Decisions Over Time", fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        graph_path = './graphs/trading_decisions_test.png'
        plt.savefig(graph_path, bbox_inches='tight', dpi=300)
        print(f"Graph saved at: {graph_path}")

    plt.show()



def test_agent(agent, test_data):
    env = TradingEnvironment(test_data)
    state = env.reset()
    state = np.reshape(state, [1, len(state)])
    total_reward = 0
    done = False

    buy_points = []
    sell_points = []
    prices = []
    balances = []
    dates = test_data.index  # Assuming your data has datetime index
    decisions_log = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, len(next_state)])
        state = next_state
        total_reward += reward

        current_price = test_data.iloc[env.current_step]["Close"]
        current_balance = env.balance + (env.shares_held * current_price)

        decision = "Hold" if action == 0 else "Buy" if action == 1 else "Sell"
        decisions_log.append((env.current_step, decision, current_price, reward))

        if action == 1:
            buy_points.append((env.current_step, current_price))
        elif action == 2:
            sell_points.append((env.current_step, current_price))

        prices.append(current_price)
        balances.append(current_balance)

    print("Trading Log:")
    print("Step | Decision | Price | Reward")
    for step, decision, price, step_reward in decisions_log:
        print(f"{step:4} | {decision:<8} | {price:.2f} | {step_reward:.2f}")

    print(f"\nTest Reward: {total_reward}")
    print(f"Final Portfolio Value: {env.balance + (env.shares_held * prices[-1])}")

    plot_decisions(prices, buy_points, sell_points, dates[:len(prices)], balances, save=True)
    return total_reward


# def plot_decisions(prices, buy_points, sell_points, save=False, filename='decision_plot.png'):
#     """Plots the price graph with buy and sell points, and optionally saves it."""
#     plt.figure(figsize=(10, 5))
#     plt.plot(prices, label='Price', color='blue', alpha=0.6)
#
#     if buy_points:
#         buy_x, buy_y = zip(*buy_points)
#         plt.scatter(buy_x, buy_y, color='green', label='Buy', marker='^')
#
#     if sell_points:
#         sell_x, sell_y = zip(*sell_points)
#         plt.scatter(sell_x, sell_y, color='red', label='Sell', marker='v')
#
#     plt.title("Trading Decisions (Buy/Sell) Over Time")
#     plt.xlabel("Time Step")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.grid()
#
#     if save:
#         graph_path = './graphs/trading_decisions_test.png'
#         plt.savefig(graph_path)
#         print(f"Graph saved at: {graph_path}")
#
#     plt.show()
#
#
# def test_agent(agent, test_data):
#     env = TradingEnvironment(test_data)
#     state = env.reset()
#     state = np.reshape(state, [1, len(state)])
#     total_reward = 0
#     total_portfolio_value = env.initial_balance
#     done = False
#
#     buy_points = []
#     sell_points = []
#     prices = []
#     decisions_log = []
#
#     while not done:
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, len(next_state)])
#         state = next_state
#         total_reward += reward
#
#         current_price = test_data.iloc[env.current_step]["Close"]
#         decision = "Hold" if action == 0 else "Buy" if action == 1 else "Sell"
#         decisions_log.append((env.current_step, decision, current_price, reward))
#
#         if action == 1:
#             buy_points.append((env.current_step, current_price))
#         elif action == 2:
#             sell_points.append((env.current_step, current_price))
#
#         prices.append(current_price)
#
#     print("Trading Log:")
#     print("Step | Decision | Price | Reward")
#     for step, decision, price, step_reward in decisions_log:
#         print(f"{step:4} | {decision:<8} | {price:.2f} | {step_reward:.2f}")
#
#     print(f"\nTest Reward: {total_reward}")
#     print(f"Final Portfolio Value: {env.balance + (env.shares_held * prices[-1])}")
#
#     plot_decisions(prices, buy_points, sell_points, save=True)
#     return total_reward


# Training loop
for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    training_rewards.append(total_reward)

    # if episode % VALIDATION_INTERVAL == 0:
    #     validation_reward = test_agent(agent, test_data[:100])
    #     print(f"Validation Reward at Episode {episode}: {validation_reward}")

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")

# Save the trained model
model_path = './models/trained_dqn_model_test.keras'
agent.model.save(model_path)
print(f"Model saved at: {model_path}")

# Final test
test_agent(agent, test_data)

# Plot training metrics
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(training_rewards)
plt.title('Training Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.tight_layout()
plt.savefig('./graphs/training_metrics_test.png')
plt.show()
