from utils.data_loader import load_data, preprocess_data
from utils.config import *
from utils.environment import TradingEnvironment
#from utils.env_draft import TradingEnvironment
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


train_data = load_data(TRAIN_DATA_PATH)
train_data = preprocess_data(train_data)
test_data = load_data(TEST_DATA_PATH)
test_data = preprocess_data(test_data)
test_data = test_data[TEST_DATA_START:]

#environment and agent
env = TradingEnvironment(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    learning_rate=LEARNING_RATE
)

# Track metrics
training_rewards = []
portfolio_values = []
best_reward = float('-inf')

def plot_decisions(dates, prices, buy_points, sell_points, revenue, save=False, filename=DECISION_PLOT_DQN_PATH):
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

    plt.show()

def test_agent(agent, test_data):
    env = TradingEnvironment(test_data)
    state = env.reset()
    state = np.reshape(state, [1, len(state)])
    total_reward = 0
    total_portfolio_value = env.initial_balance
    done = False

    buy_points = []
    sell_points = []
    prices = []
    dates = []  # To track dates for x-axis
    revenue = []  # To track portfolio value over time
    decisions_log = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, len(next_state)])
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

    plot_decisions(dates, prices, buy_points, sell_points, revenue, save=True)
    return total_reward

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
plt.savefig(DQN_TRAINING_METRICS_PATH)
plt.show()
