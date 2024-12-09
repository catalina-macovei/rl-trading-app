#  Reinforcement Learning Trading Algorithms
A comprehensive implementation of stock price prediction using reinforcement learning techniques: Deep Q-Learning and Actor-Critic algorithms.    
Explore how AI can make trading decisions based on dynamic market data. :purple_heart:  

## Workflow
<ul>
  <li>Installing Python dependencies</li>
  <li>Data preparation:</li>
   <ul>
     <li>Acquiring data from Alpha Vantage stock APIs</li>
     <li>Normalizing raw data</li>
     <li>Generating training and validation datasets</li>
   </ul>
  <li>Defining algorithm models</li>
  <li>Training and evaluation</li>
</ul>

### Installation steps 

1. Create a Virtual Environment
    A virtual environment helps isolate project dependencies    
    ```python -m venv venv```
2. Activate the virtual environment:   
    ```source venv/bin/activate```
3. Install NumPy:     
    ```pip install numpy```
4. Install PyTorch:   
    ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
5. Install Matplotlib:    
    ```pip install matplotlib```
6. Install Alpha Vantage:   
    ```pip install alpha_vantage```
7. Deactivate venv:    
    ```deactivate```