# FedG-ELITE Core Components

This folder contains the core implementation of the **FedG-ELITE** (Federated Graph-based ELITE) algorithm. These components replace the traditional Tabular Q-Learning approach with a Deep Graph Learning approach.

## Files Overview

### 1. `model.py` (The Brain)
*   **Class**: `FedG_DQN`
*   **Purpose**: Defines the Neural Network architecture.
*   **Key Features**:
    *   **GraphSAGE Encoder**: Uses `SAGEConv` layers to extract topological features from the road network.
    *   **DQN Head**: A flexible MLP that predicts Q-values for each neighbor.
    *   **Dynamic Routing**: Designed to handle variable numbers of neighbors (3-way, 4-way junctions) by using a "Link-based" scoring approach.

### 2. `data_converter.py` (The Translator)
*   **Class**: `GraphConverter`
*   **Purpose**: Bridges the gap between raw ELITE data and the Neural Network.
*   **Key Features**:
    *   **ID Mapping**: Converts large OSM Node IDs (e.g., `811022916`) into continuous indices (0, 1, 2...) required by PyTorch.
    *   **Graph Construction**: Converts the `adj` dictionary into a PyTorch Geometric `edge_index` tensor.
    *   **Feature Engineering**: Constructs node feature vectors (Normalized Position, Degree, Speed, Density).

### 3. `dqn_utils.py` (The Training Tools)
*   **Components**: `ReplayBuffer`, `train_step`
*   **Purpose**: Provides the machinery for Reinforcement Learning.
*   **Key Features**:
    *   **ReplayBuffer**: Stores experiences `(State, Action, Reward, NextState)` to stabilize training.
    *   **train_step**: Implements the **DQN Update Rule** (Bellman Equation). It computes the loss between the *Predicted Q* and the *Target Q* ($R + \gamma \max Q'$) and updates the model weights.

## How They Fit Together
1.  **`data_converter.py`** prepares the `Data` object from the global map.
2.  **`model.py`** takes this `Data` object and predicts the best route.
3.  **`dqn_utils.py`** takes the feedback (Reward) from the environment and updates **`model.py`** to make it smarter.
