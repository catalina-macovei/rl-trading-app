import numpy as np
import random
import pandas as pd
from dqn_agent import DQNAgent
from utils.environment import TradingEnvironment

# Hyperparameter tuning
def hyperparameter_tuning(data, trials=20, episodes=50, max_steps=200):
    best_score = -float('inf')
    best_params = None

    param_space = {
        'gamma': [0.9, 0.95, 0.99],
        'epsilon': [1.0],
        'epsilon_min': [0.01],
        'epsilon_decay': [0.99, 0.995],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64],
        'memory_size': [10000, 20000]
    }

    for trial in range(trials):
        # Randomly sample hyperparameters
        gamma = random.choice(param_space['gamma'])
        epsilon = random.choice(param_space['epsilon'])
        epsilon_min = random.choice(param_space['epsilon_min'])
        epsilon_decay = random.choice(param_space['epsilon_decay'])
        learning_rate = random.choice(param_space['learning_rate'])
        batch_size = random.choice(param_space['batch_size'])
        memory_size = random.choice(param_space['memory_size'])

        # Initialize environment and agent
        env = TradingEnvironment(data)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(
            state_size, action_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, memory_size
        )

        # Train agent
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for step in range(max_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            agent.replay(batch_size)
            total_rewards.append(total_reward)

        # Evaluate performance
        average_reward = np.mean(total_rewards)
        print(f"Trial {trial + 1}/{trials} - Average Reward: {average_reward} with params: "
              f"gamma={gamma}, epsilon_decay={epsilon_decay}, learning_rate={learning_rate}, "
              f"batch_size={batch_size}, memory_size={memory_size}")

        # Update best parameters
        if average_reward > best_score:
            best_score = average_reward
            best_params = {
                'gamma': gamma,
                'epsilon': epsilon,
                'epsilon_min': epsilon_min,
                'epsilon_decay': epsilon_decay,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'memory_size': memory_size
            }

    print(f"Best Average Reward: {best_score}")
    print(f"Best Parameters: {best_params}")
    return best_params

# Example usage
if __name__ == "__main__":
    # Sample data (Replace with your stock market data)
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 1000),
        'High': np.random.uniform(100, 200, 1000),
        'Low': np.random.uniform(100, 200, 1000),
        'Close': np.random.uniform(100, 200, 1000),
        'Volume': np.random.randint(1000, 5000, 1000)
    })

    best_params = hyperparameter_tuning(data)
