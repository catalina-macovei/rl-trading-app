import numpy as np
import gym
from gym import spaces

"""
Params:
    initial_balance: The starting capital for the agent.
    data: The historical stock market data.
Attributes:
    data: The historical stock market data.
    current_step: Tracks the index of the current step.
    balance: Tracks the agent s available cash.
    shares_held: Tracks the number of shares currently held.
    action_space: Defines the possible actions (0: Hold, 1: Buy, 2: Sell).
    observation_space: Defines the shape of observations, based on the number of features in the data.
"""


class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.done = False

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.data.columns),), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value_before = self.balance + (self.shares_held * current_price)

        reward = 0

        if action == 1:  # Buy
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.balance -= cost

                # Enhanced buy reward
                if self.current_step > 0:
                    previous_price = self.data.iloc[self.current_step - 1]['Close']
                    price_change = (current_price - previous_price) / previous_price
                    if price_change < -0.02:  # 2% drop
                        reward = 1.0  # Stronger positive reward for buying dips

        elif action == 2:  # Sell
            if self.shares_held > 0:
                sale_value = self.shares_held * current_price
                self.balance += sale_value

                # Enhanced sell reward
                profit = sale_value - (self.shares_held * self.data.iloc[self.current_step - 1]['Close'])
                if profit > 0:
                    reward = profit / portfolio_value_before  # Proportional to profit
                self.shares_held = 0

        else:  # Hold
            # Reduced holding penalty
            reward = -0.001  # Smaller penalty for holding

        # Calculate portfolio value change
        portfolio_value_after = self.balance + (self.shares_held * current_price)
        portfolio_change = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        reward += portfolio_change

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_observation(), reward, done, {'portfolio_value': portfolio_value_after}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")



# testing
"""
import pandas as pd

# Sample data for testing
data = pd.DataFrame({
    'Open': [100, 102, 104, 103],
    'High': [105, 106, 108, 107],
    'Low': [99, 100, 102, 101],
    'Close': [102, 104, 103, 105],
    'Volume': [1000, 1500, 1200, 1300]
})

# Initialize the environment
env = TradingEnvironment(data)

print("test loop:")
obs = env.reset()
while not env.done:
    action = np.random.choice([0, 1, 2])  # Random action: Hold, Buy, or Sell
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()

print("Environment has terminated:", env.done)

"""