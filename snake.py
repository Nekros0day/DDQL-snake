import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory_size = 1000000

# Initialize Pygame
pygame.init()

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Define display dimensions
display_width = 100
display_height = 100

# Set up display
dis = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Snake Game')

# Define snake block size and speed
snake_block = 10
snake_speed = 15

# Improved Neural Network for DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(12, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        self.fc4 = nn.Linear(32, 4)


    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Snake game environment
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x1 = display_width / 2
        self.y1 = display_height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_List = []
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, display_width - snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, display_height - snake_block) / 10.0) * 10.0
        self.direction = None
        self.score = 0
        return self.get_state()

    def get_state(self):
        state = [
            self.x1_change == snake_block,  # Moving right
            self.x1_change == -snake_block, # Moving left
            self.y1_change == snake_block,  # Moving down
            self.y1_change == -snake_block, # Moving up
            self.foodx > self.x1,           # Food is right
            self.foodx < self.x1,           # Food is left
            self.foody > self.y1,           # Food is down
            self.foody < self.y1,           # Food is up
            self.x1 >= display_width,       # Collision with right wall
            self.x1 < 0,                    # Collision with left wall
            self.y1 >= display_height,      # Collision with bottom wall
            self.y1 < 0,                    # Collision with top wall
        ]
        return np.array(state, dtype=int)

    def step(self, action):
        reward = 0
        if action == 0:  # Move left
            if self.direction != 'RIGHT':
                self.x1_change = -snake_block
                self.y1_change = 0
                self.direction = 'LEFT'
        elif action == 1:  # Move right
            if self.direction != 'LEFT':
                self.x1_change = snake_block
                self.y1_change = 0
                self.direction = 'RIGHT'
        elif action == 2:  # Move up
            if self.direction != 'DOWN':
                self.x1_change = 0
                self.y1_change = -snake_block
                self.direction = 'UP'
        elif action == 3:  # Move down
            if self.direction != 'UP':
                self.x1_change = 0
                self.y1_change = snake_block
                self.direction = 'DOWN'

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if self.x1 >= display_width:
            self.x1 = 0
        elif self.x1 < 0:
            self.x1 = display_width - snake_block
        if self.y1 >= display_height:
            self.y1 = 0
        elif self.y1 < 0:
            self.y1 = display_height - snake_block

        snake_Head = [self.x1, self.y1]
        self.snake_List.append(snake_Head)
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        for x in self.snake_List[:-1]:
            if x == snake_Head:
                return self.get_state(), -100, True, self.score

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, display_width - snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, display_height - snake_block) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.score += 1
            reward = 10

        return self.get_state(), reward, False, self.score

    def render(self):
        dis.fill(black)
        pygame.draw.rect(dis, white, [self.foodx, self.foody, snake_block, snake_block])
        for x in self.snake_List:
            pygame.draw.rect(dis, green, [x[0], x[1], snake_block, snake_block])
        pygame.display.update()

# DQN agent
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

# Main training loop
def train():
    global epsilon
    game = SnakeGame()
    agent = DQNAgent()

    episodes = 1000
    best_score = 0
    
    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        time_max = (int) (100000 * (1 + (5*e/1000)))
        for time_t in range(time_max):
            process_events()  # Ensure we process events
            game.render()
            action = agent.act(state)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()
                if score > best_score:
                    best_score = score
                    agent.save("dqn_snake_best.pth")
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2}")
                break
            agent.replay()
            pygame.time.delay(10)  # Use Pygame's delay
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay ** e)

    agent.save("dqn_snake.pth")
    pygame.quit()
    quit()

def play():
    global epsilon
    epsilon = 0
    game = SnakeGame()
    agent = DQNAgent()
    agent.load("dqn_snake_best.pth")

    while True:
        state = game.reset()
        while True:
            process_events()  # Ensure we process events
            game.render()
            action = agent.act(state)
            next_state, _, done, score = game.step(action)
            state = next_state
            if done:
                print(f"Score: {score}")
                break
            pygame.time.delay(100)  # Use Pygame's delay

def main():
    choice = input("Do you want to train (t) a new model or load (l) an existing model? ")
    if choice == 't':
        train()
    elif choice == 'l':
        play()
    else:
        print("Invalid choice. Please enter 't' or 'l'.")

if __name__ == "__main__":
    main()
