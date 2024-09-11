from game import Game
import numpy as np
import random
import sys
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, hidden_dim, drop_out):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 4)
        )
    def forward(self, x):
        x = x.to(device)
        return self.fc(x)

def train_game(game, it, epsilon):
    global losses, rewards
    batch_label, batch_output = [], []
    step = 1
    total_reward = 0
    while not game.game_over():
        Q_values = model(game.vector())
        if random.random() < epsilon:
            best_action = random.choice([a for a in range(4) if game.is_action_available(a)])
        else:
            Q_valid_values = [Q_values[a].detach() if game.is_action_available(a) else float('-inf') for a in range(4)]
            best_action = np.argmax(Q_valid_values)
        reward = game.do_action(best_action)
        total_reward += reward
        Q_star = Q_values[best_action]
        try:
            new_state, vec, reward = game.get_next_state(best_action)
        except Exception as e:
            print("Error:", e)
        with torch.no_grad():
            Q_next = target_model(vec)
            best_action_next = torch.argmax(model(vec)).item()
        batch_output.append(Q_star)
        batch_label.append(torch.tensor(reward + gamma * Q_next[best_action_next], dtype=torch.float32).to(device))
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
                print("epoch: {}, game score: {}, total reward: {}".format(it, game.score(), total_reward))
                rewards.append(total_reward)
                return
        step += 1

max_tiles = []
def eval_game():
    global scores
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            game = Game()
            while not game.game_over():
                Q_values = model(game.vector())
                Q_valid_values = [Q_values[a].detach() if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = np.argmax(Q_valid_values)
                game.do_action(best_action)
            print("score is: {} max tile is: {}".format(game.score(), game.max_tile()))
            scores.append(game.score())
            max_tiles.append(game.max_tile())

def plot_results(losses, scores, max_tiles, rewards):
    # 1. Biểu đồ Hàm Loss
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Biểu đồ Điểm Số Thống Kê Tương Ứng với Số Lượng Màn Chơi
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title('Score Distribution across Games')
    plt.grid(True)
    plt.show()

    # 3. Biểu đồ Số Lượng Màn Chơi Có Ô Giá Trị Lớn Nhất Tương Ứng
    plt.figure(figsize=(12, 6))
    plt.hist(max_tiles, bins=np.arange(min(max_tiles), max(max_tiles) + 1, 1), color='orange', edgecolor='black', alpha=0.7)
    plt.xlabel('Max Tile')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Max Tile across Games')
    plt.grid(True)
    plt.show()
    
    # 4. Biểu đồ Reward qua các Epoch
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
hidden_dim = 128
drop_out = 0.2
n_epoch = 4000
n_eval = 100
gamma = 0.99
epsilon_start = 1
epsilon_end = 0.012
epsilon_decay = n_epoch // 4
target_update_freq = 10

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(hidden_dim, drop_out).to(device)
target_model = Net(hidden_dim, drop_out).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
criterion = criterion.to(device)

losses = []
scores = []
max_tiles = []
rewards = []

def update_target_model():
    target_model.load_state_dict(model.state_dict())

if __name__ == "__main__":
    model.train()
    best_loss = float('inf')
    patience = 50
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
    torch.save(model.state_dict(), '2048_Double_DQN.pth')
    print("The mean of the score is {}".format(np.mean(scores)))
    plot_results(losses, scores, max_tiles, rewards)
