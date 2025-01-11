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

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/A2C_{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

def calculate_sharpe(df: pd.DataFrame):
    df["daily_return"] = df["account_values"].pct_change(1)
    if df["daily_return"].std() != 0:
        sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
        return sharpe
    else:
        return 0
    
def train_ac_agent(agent, train_data, episodes, trial_no=0):
    env = TradingEnvironment(train_data)

    trial_no = str(trial_no)

    for episode in tqdm(range(episodes)):
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
            
            state = state_
            

def train_ac_batch_agent(agent, train_data, episodes, batch_size=256, trial_no=0):
    env = TradingEnvironment(train_data)
    trial_no = str(trial_no)

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        actions_taken = []

        # generate a batch
        states_batch = []
        next_states_batch = []
        rewards_batch = []
        actions_batch = []
        dones_batch = []

        while not done:
            # 1. sample {s_i, a_i} from pi_theta(a|s)
            for step in range(batch_size):
                if done:
                    return
                action = agent.choose_action(state)
                actions_taken.append(action)
                next_state, reward, done, info = env.step(action)

                states_batch.append(state)
                next_states_batch.append(next_state)
                actions_batch.append(action)
                rewards_batch.append(reward)
                dones_batch.append(done)
                
                state = next_state
                total_reward += reward
            
            metrics = agent.learn(states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch)

            
                
    
def validate_agent(agent, validation_data):
    account_values = []
    env = TradingEnvironment(validation_data)
    state = env.reset()
    total_reward = 0
    actions_taken = []
    done = False
    
    while not done:
        action = agent.choose_action(state)
        actions_taken.append(action)
        next_state, reward, done, info = env.step(action)
            
        state = next_state
        total_reward += reward

        next_price = env.data.iloc[env.current_step]['Close'] if not done else env.data.iloc[env.current_step - 1]['Close']
        portfolio_value_after = env.balance + (env.shares_held * next_price)
        account_values.append(portfolio_value_after)

    df = pd.DataFrame({"account_values": account_values, "daily_return": np.zeros((len(account_values)))})
    sharpe = calculate_sharpe(df)
    total_portfolio_value = account_values[-1]

    return sharpe, total_portfolio_value

def sample_hyperparameters(batch=False):
    episodes = np.random.choice([1, 5, 10, 20])
    critic_alpha = np.random.uniform(1e-4, 0.001)
    actor_alpha = np.random.uniform(1e-4, 0.001)
    gamma = np.random.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    entropy_coeff = np.random.uniform(0.5, 5)
    max_grad_norm = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
    actor_fc1 = critic_fc1 = np.random.choice([512, 1024])
    actor_fc2 = critic_fc2 = actor_fc1 / 2

    if batch:
        batch_size = np.random.choice([32, 64, 128, 256])
    else:
        batch_size = None

    return {
        'episodes': episodes, 
        'critic_alpha': critic_alpha, 
        'actor_alpha': actor_alpha, 
        'gamma': gamma, 
        'entropy_coeff': entropy_coeff, 
        'max_grad_norm': max_grad_norm, 
        'actor_fc1': actor_fc1, 
        'actor_fc2': actor_fc2,
        'critic_fc1': critic_fc1,
        'critic_fc2': critic_fc2,
        'batch_size': batch_size
    }
   


def tune_agent(train_data, validation_data, withBatch=False):
    results = []
    best_sharpe = None
    actor_fc1 = 0
    actor_fc2 = 0
    critic_fc1 = 0
    critic_fc2 = 0

    if withBatch: 
        name = 'Batch'
    else:
        name = 'Online'

    for i in range(10):
        print('Trial ', i)
        hyperparameters = sample_hyperparameters(withBatch)

        if withBatch:
            agent = A2CBatchAgent(
                n_actions=3, 
                critic_alpha=hyperparameters.get('critic_alpha'),
                actor_alpha=hyperparameters.get('actor_alpha'),
                gamma=hyperparameters.get('gamma'),
                entropy_coeff=hyperparameters.get('entropy_coeff'),
                max_grad_norm=hyperparameters.get('max_grad_norm'),
                critic_fc1=hyperparameters.get('critic_fc1'),
                critic_fc2=hyperparameters.get('critic_fc2'),
                actor_fc1=hyperparameters.get('actor_fc1'),
                actor_fc2=hyperparameters.get('actor_fc2')
                )
            train_ac_batch_agent(agent, train_data, hyperparameters.get('episodes'), hyperparameters.get('batch_size'), i)

        else:
            agent = A2CAgent(
                n_actions=3, 
                critic_alpha=hyperparameters.get('critic_alpha'),
                actor_alpha=hyperparameters.get('actor_alpha'),
                gamma=hyperparameters.get('gamma'),
                entropy_coeff=hyperparameters.get('entropy_coeff'),
                max_grad_norm=hyperparameters.get('max_grad_norm'),
                critic_fc1=hyperparameters.get('critic_fc1'),
                critic_fc2=hyperparameters.get('critic_fc2'),
                actor_fc1=hyperparameters.get('actor_fc1'),
                actor_fc2=hyperparameters.get('actor_fc2')
                )
            train_ac_agent(agent, train_data, hyperparameters.get('episodes'), i)

        sharpe, total_portfolio_value = validate_agent(agent, validation_data)
        results.append((sharpe, total_portfolio_value, hyperparameters))

        if best_sharpe is None:
            best_sharpe = sharpe
            agent.save_models()
            critic_fc1=hyperparameters.get('critic_fc1'),
            critic_fc2=hyperparameters.get('critic_fc2'),
            actor_fc1=hyperparameters.get('actor_fc1'),
            actor_fc2=hyperparameters.get('actor_fc2')
        else:
            if best_sharpe < sharpe:
                best_sharpe = sharpe
                agent.save_models()
                critic_fc1=hyperparameters.get('critic_fc1'),
                critic_fc2=hyperparameters.get('critic_fc2'),
                actor_fc1=hyperparameters.get('actor_fc1'),
                actor_fc2=hyperparameters.get('actor_fc2')

        with summary_writer.as_default():
            tf.summary.scalar('Tuning/Trial/Total Portfolio Value/'+name, total_portfolio_value, step=i)
            tf.summary.scalar('Tuning/Trial/Sharpe Ratio/'+name, sharpe, step=i)

    print("best actor fc1", actor_fc1)
    print("best actor fc2", actor_fc2)
    print("best critic fc1", critic_fc1)
    print("best critic fc2", critic_fc2)

    return results


# Load and preprocess data
data = load_data('./data/AAPL.csv')
data = preprocess_data(data)
train_data = data[:800]
validation_data = data[800:1300]
test_data = data[1300:]

a2c_results = tune_agent(train_data, validation_data)
a2c_batch_results = tune_agent(train_data, validation_data, True)

print('A2C results:')
print('############################')
for result in a2c_results:
    print('Sharpe: ', result[0])
    print('Total Portfolio Value: ', result[1])
    print("Hyperparameters:")
    print(result[2])
    print('----------------------------\n')

print('\n')

print('A2C Batch results:')
print('############################')
for result in a2c_batch_results:
    print('Sharpe: ', result[0])
    print('Total Portfolio Value: ', result[1])
    print("Hyperparameters:")
    print(result[2])
    print('----------------------------\n')
