# Training parameters
EPISODES = 1000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5
VALIDATION_INTERVAL = 50

# Data paths
TRAIN_DATA_PATH = './data/AAPL.csv'
TEST_DATA_PATH = './data/GOOG.csv'
TEST_DATA_START = 1400

# Model parameters
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# Plotting parameters
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6
ALPHA = 0.6
MARKER_SIZE = 50
MARKER_ALPHA = 0.7
GRID_ALPHA = 0.3


# File paths
DQN_MODEL_SAVE_PATH = './models/trained_dqn_model_test.keras'
DECISION_PLOT_DQN_PATH = './graphs/decision_plot_test.png'
DQN_TRAINING_METRICS_PATH = './graphs/training_metrics_test.png'