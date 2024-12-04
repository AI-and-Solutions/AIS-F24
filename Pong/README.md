# Teaching AI to play PONG using DQN

## Authors: Jovanny Aguilar & Asher Schalet

## Description
This project shows how to teach an AI agent how to play a simple game of Pong using Deep Q-Learning (DQN).
The AI agent learns to interact with a custom Pong evnironment through reinforcement learning, where it optimizes
its gameplay strategy based on rewards and penalties. The project uses a custom physics engine and neural network for decision making and training. The program is written in Python, using libraries such as Pytorch for the nueral network and Pygame for game visuals.

## Features
- **Custom Pong Environment**: A fully functional Pong game implemented with Pygame, featuring adjustable screen dimensions, paddle speeds, and ball dynamics.
- **Deep Q-Learning Implementation**: A Convolutional Neural Network (CNN)-based DQN model that learns to play the game by optimizing its Q-values.
- **Replay Buffer**: Efficient storage and sampling of gameplay experiences for stable and effective training.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation during the agent's training process.
- **Custom Frame Preprocessing**: Converts game frames to a grayscale format and resizes them for input into the CNN.
- **Training Visualization**: Real-time rendering of the Pong environment to monitor agent behavior during training.
- **GPU Support**: Accelerates training using CUDA if available.

## How it works
1. **Game Environment**:
   - A custom Pong game is built using Pygame, designed to simulate the Pong gameplay mechanics.
   - The environment provides observations (grayscale game frames), rewards, and done flags after each action.

2. **Deep Q-Network (DQN)**:
   - The DQN model uses convolutional layers to extract spatial features from game frames.
   - Fully connected layers compute Q-values, predicting the best actions to take in a given state.

3. **Training Loop**:
   - At each step, the agent chooses an action using the epsilon-greedy method.
   - The game environment returns a reward and the next game frame based on the chosen action.
   - The agent stores the experience in a replay buffer.
   - Periodically, the agent samples experiences from the buffer to train the DQN.
   - The target network is updated at regular intervals to stabilize training.

4. **Preprocessing**:
   - Game frames are resized to 84x84 pixels and normalized for input into the CNN.

5. **Reward System**:
   - Positive rewards are given when the agent scores a point.
   - Negative rewards are assigned when the opponent scores.
   - This guides the agent to maximize its score while minimizing the opponent's.

6. **Evaluation**:
   - The trained model's performance is assessed by letting it play multiple episodes of Pong.

## Warning
- **Dependencies**: Ensure all required Python libraries (e.g., PyTorch, Pygame, NumPy) are installed before running the project.
- **Potential Bugs**: Since the project involves custom implementations of physics and AI training, there may be bugs.
