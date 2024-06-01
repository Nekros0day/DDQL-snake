# Snake Game with Deep Q-Learning (DQN)

This project implements the classic Snake game using Pygame and integrates a Deep Q-Learning (DQN) agent to play the game.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Game Mechanics](#game-mechanics)
- [AI Agent](#ai-agent)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using pip:


## Usage

You can choose to either train a new model or load an existing model to play the game.

1. **Training a new model:**

   
When prompted, enter `t` to start training the model.

2. **Loading an existing model:**

   
When prompted, enter `l` to load the pre-trained model and watch the AI play the game.

## Game Mechanics

The game consists of controlling a snake to eat food appearing on the screen. The snake grows longer each time it eats food, and the game ends if the snake collides with itself or the walls.

### Controls
- **LEFT:** Move the snake left.
- **RIGHT:** Move the snake right.
- **UP:** Move the snake up.
- **DOWN:** Move the snake down.

### Colors
- **Snake:** Green
- **Food:** White
- **Background:** Black

## AI Agent

The AI agent uses a Deep Q-Network (DQN) to learn the best moves in the game. It leverages a replay memory to store past experiences and train the network.

### Neural Network Architecture
- Multiple fully connected layers with ReLU activation and Layer Normalization for stability.

### Training
- **Replay Memory:** Stores past experiences to train the network.
- **Target Network:** Uses a target network to stabilize training.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation.

## Project Structure

- `main.py`: Main script to run the game and train/load the model.
- `dqn_agent.py`: Contains the DQN agent and neural network definition.
- `game.py`: Implements the game logic and rendering.
