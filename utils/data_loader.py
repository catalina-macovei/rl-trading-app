import pandas as pd
import numpy as np

### Loading stock data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Sort in chronological order
    df.sort_index(inplace=True)

    return df

### cleaning the stock data, adding features
def preprocess_data(df):
    # price change and percentage change
    df['Price_Change'] = df['Close'] - df['Open']
    df['Pct_Change'] = df['Close'].pct_change()

    # add standard deviation of returns over a window
    df['Volatility'] = df['Close'].rolling(window=7).std()

    df.dropna(inplace=True)  # remove rows if missing data persists

    # ensure numeric columns have no infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


