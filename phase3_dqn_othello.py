import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque


# -----------------------------
# DQN NETWORK
# -----------------------------

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# REPLAY BUFFER
# -----------------------------

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def size(self):
        return len(self.buffer)


# -----------------------------
# AGENT
# -----------------------------

class Agent:

    def __init__(self):

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

        self.batch_size = 64

        self.memory = ReplayBuffer(10000)

        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.target_update = 1000
        self.step_count = 0

        self.loss_fn = nn.MSELoss()


    def select_action(self, state, valid_actions):
        if len(valid_actions) == 0:
        # Pas de coup légal : passer le tour
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state).detach().numpy()[0]

    # masque actions invalides
        mask = np.full(64, -1e9)
    
        for a in valid_actions:
            mask[a] = q_values[a]
        return np.argmax(mask)


    def train(self):

        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)

        next_q_values = self.target_model(next_states).detach()

        max_next_q = torch.max(next_q_values, dim=1)[0]

        target = rewards + self.gamma * max_next_q * (1 - dones)

        current = q_values.gather(1, actions).squeeze()

        loss = self.loss_fn(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.step_count += 1

        if self.step_count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())


# -----------------------------
# SIMPLE OTHELLO ENV
# -----------------------------

class OthelloEnv:

    def __init__(self):
        self.reset()

    def reset(self):

        self.board = np.zeros((8,8), dtype=int)

        self.board[3][3] = -1
        self.board[4][4] = -1
        self.board[3][4] = 1
        self.board[4][3] = 1

        return self.board.flatten()


    def inside(self,x,y):
        return 0 <= x < 8 and 0 <= y < 8


    def get_flips(self,x,y,player):

        if self.board[x][y] != 0:
            return []

        directions = [
            (-1,-1),(-1,0),(-1,1),
            (0,-1),(0,1),
            (1,-1),(1,0),(1,1)
        ]

        flips = []

        for dx,dy in directions:

            nx = x + dx
            ny = y + dy

            line = []

            while self.inside(nx,ny) and self.board[nx][ny] == -player:
                line.append((nx,ny))
                nx += dx
                ny += dy

            if self.inside(nx,ny) and self.board[nx][ny] == player:
                flips.extend(line)

        return flips


    def legal_actions(self,player):

        actions = []

        for i in range(8):
            for j in range(8):

                flips = self.get_flips(i,j,player)

                if len(flips) > 0:
                    actions.append(i*8 + j)

        return actions


    def apply_move(self,action,player):

        x = action // 8
        y = action % 8

        flips = self.get_flips(x,y,player)

        if len(flips) == 0:
            return False

        self.board[x][y] = player

        for fx,fy in flips:
            self.board[fx][fy] = player

        return True


    def game_over(self):

        if len(self.legal_actions(1)) > 0:
            return False

        if len(self.legal_actions(-1)) > 0:
            return False

        return True


    def result(self):

        player = np.sum(self.board == 1)
        opponent = np.sum(self.board == -1)

        if player > opponent:
            return 1
        elif player < opponent:
            return -1
        else:
            return 0


    def step(self,action):

       # Tour agent
        if action is not None:
            self.apply_move(action,1)
        # Vérifier fin de partie
        if self.game_over():
            return self.board.flatten(), self.result(), True
        # Tour adversaire aléatoire
        opp_moves = self.legal_actions(-1)
        if len(opp_moves) > 0:
            opp_action = random.choice(opp_moves)
            self.apply_move(opp_action,-1)
        # Vérifier fin de partie
        if self.game_over():
            return self.board.flatten(), self.result(), True
        return self.board.flatten(), 0, False

# -----------------------------
# TRAINING
# -----------------------------

env = OthelloEnv()
agent = Agent()

episodes = 5000

start_time = time.time()
wins = 0  # compteur victoires agent

for episode in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0
    moves = 0

    while not done:
        valid_actions = env.legal_actions(1)
        action = agent.select_action(state, valid_actions)
        next_state, reward, done = env.step(action)
        moves += 1
        if action is not None:
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
        state = next_state
        total_reward += reward
    
    # Mise à jour compteur victoire
    if total_reward > 0:
        wins += 1

    print(f"Episode {episode} Reward: {total_reward} | Moves: {moves}")
    
end_time = time.time()
win_rate = wins / episodes * 100
print(f"Taux de victoire final : {win_rate:.2f}%")
print("Temps total d'entraînement :", round(end_time - start_time,2), "secondes")