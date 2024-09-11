import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from game import Game

class DuelingDQN(nn.Module):
    def __init__(self, hidden_dim, drop_out):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)  # 4 actions: left, up, right, down
        )
        
    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        Q = value + advantage - advantage.mean()
        return Q

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_dim = 128
drop_out = 0.2

model = DuelingDQN(hidden_dim, drop_out).to(device)
target_model = DuelingDQN(hidden_dim, drop_out).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss().to(device)

losses = []
scores = []
max_tiles = []

def update_target_model():
    target_model.load_state_dict(model.state_dict())

def train_game(game, it, epsilon):
    global losses
    batch_label, batch_output = [], []
    step = 1
    state = torch.tensor(game.vector(), dtype=torch.float32).unsqueeze(0).to(device)
    while not game.game_over():
        Q_values = model(state)
        if random.random() < epsilon:
            best_action = random.choice([a for a in range(4) if game.is_action_available(a)])
        else:
            Q_valid_values = [Q_values[0, a].detach() if game.is_action_available(a) else float('-inf') for a in range(4)]
            best_action = np.argmax(Q_valid_values)
        reward = game.do_action(best_action)
        Q_star = Q_values[0, best_action]
        next_state, vec, reward = game.get_next_state(best_action)
        next_state = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            Q_next = target_model(next_state)
            best_action_next = torch.argmax(model(next_state)).item()
        
        batch_output.append(Q_star)
        batch_label.append(torch.tensor(reward + gamma * Q_next[0, best_action_next], dtype=torch.float32).to(device))
        
        if step % batch_size == 0 or game.game_over():
            if len(batch_label) == 0: return
            optimizer.zero_grad()
            label_tensor = torch.stack(batch_label).to(device)
            output_tensor = torch.stack(batch_output).to(device).requires_grad_(True)
            batch_label, batch_output = [], []
            loss = criterion(output_tensor, label_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if game.game_over():
                print("epoch: {}, game score: {}, max tile: {}".format(it, game.score(), game.max_tile()))
                return
        state = next_state
        step += 1

def eval_game():
    global scores, max_tiles
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            game = Game()
            state = torch.tensor(game.vector(), dtype=torch.float32).unsqueeze(0).to(device)
            while not game.game_over():
                Q_values = model(state)
                Q_valid_values = [Q_values[0, a].detach() if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = np.argmax(Q_valid_values)
                game.do_action(best_action)
                next_state, vec, _ = game.get_next_state(best_action)
                state = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
            print("game score: {}, max tile: {}".format(game.score(), game.max_tile()))
            scores.append(game.score())
            max_tiles.append(game.max_tile())

def plot_results(losses, scores, max_tiles):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title('Score Distribution across Games')
    plt.grid(True)
    plt.show()

    unique, counts = np.unique(max_tiles, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts, color='orange', edgecolor='black', alpha=0.7, width=50)
    plt.xlabel('Best tile')
    plt.ylabel('Frequency')
    plt.title('Best tile distribution')
    plt.xticks(unique)
    plt.grid(True)
    plt.show()

batch_size = 64 
n_epoch = 5000
n_eval = 100
gamma = 0.99
epsilon_start = 1
epsilon_end = 0.1
epsilon_decay = n_epoch // 2
target_update_freq = 10

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

losses = []
scores = []
max_tiles = []

if __name__ == "__main__":
    model.train()
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for it in range(n_epoch):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * it / epsilon_decay)
        game = Game()
        train_game(game, it, epsilon)
        
        if it % target_update_freq == 0:
            update_target_model()
        
        current_loss = losses[-1]
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Stopping early at epoch {it} due to no improvement in loss")
            break

    eval_game()
    torch.save(model.state_dict(), '2048_dueling_dqn_model.pth')
    print("The mean of the score is {}".format(np.mean(scores)))
    plot_results(losses, scores, max_tiles)
