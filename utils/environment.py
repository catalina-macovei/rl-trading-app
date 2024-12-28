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
        """0: Hold, 1: Buy, 2: Sell"""
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.data.columns),), dtype=np.float32
        )

    """resetting all variables that track the current state of the agent for each episode"""
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.done = False
        return self._get_observation()

    """extract this first observation, which is the values of the first row in the dataset (e.g., Open, High, Low, Close, Volume for the first day)."""
    def _get_observation(self):
        return self.data.iloc[self.current_step].values

    """takes an action and returns the next state (reward, done flag, and an empty dictionary for additional info.)"""
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        if action == 1:  # Buy
            self.shares_held += self.balance // current_price
            self.balance %= current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        ### reward is the difference between the agent's current portfolio value (balance + shares value) and the initial balance.
        reward = self.balance + (self.shares_held * current_price) - self.initial_balance
        return self._get_observation(), reward, self.done, {}

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
    obs, reward, done, info = env.step(action)
    env.render()

print("Environment has terminated:", env.done)

"""
