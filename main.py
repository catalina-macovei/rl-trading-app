import numpy as np
import random
from utils.data_loader import load_data, preprocess_data
from utils.environment import TradingEnvironment
from dqn_agent import DQNAgent

# Load and preprocess data
data = load_data('./data/AAPL.csv')
data = preprocess_data(data)
train_data = data[:10]  # Use the first 100 rows for training
test_data = data[100:120]  # Use the next 50 rows for testing

# Initialize environment and agent
env = TradingEnvironment(train_data)  # Initialize with training data
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("action=", action_size)
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 10  # Reduced for testing
batch_size = 16  # Smaller batch size for quicker training
target_update_freq = 10
# Training loop

agent = DQNAgent(state_size=state_size, action_size=action_size)

for episode in range(episodes):
    state = env.reset()  # Reset the environment
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)  # Take an action based on the state
        next_state, reward, done, _ = env.step(action)  # Step through the environment
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)  # Store the experience
        state = next_state
        total_reward += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # Train the agent

    if episode % target_update_freq == 0:
        agent.update_target_model()  # Periodically update the target model

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")








import matplotlib.pyplot as plt


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
    plt.show()


def test_agent(agent, test_data):
    env = TradingEnvironment(test_data)  # Initialize with testing data
    state = env.reset()
    state = np.reshape(state, [1, len(state)])  # Adjust for observation shape
    total_reward = 0
    total_portfolio_value = env.initial_balance  # Start with initial balance
    done = False

    buy_points = []  # Track buy points for graph
    sell_points = []  # Track sell points for graph
    prices = []  # Track prices for the graph
    decisions_log = []  # Log decisions per episode

    while not done:
        action = agent.act(state)  # Choose action
        next_state, reward, done, _ = env.step(action)  # Take action in env
        next_state = np.reshape(next_state, [1, len(next_state)])
        state = next_state
        total_reward += reward  # Increment total reward

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


test_agent(agent, test_data)