from game import Game
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

def play_game(test_game):
    while True:
        print(test_game.available_actions())
        test_game.print_state()
        try:
            x = int(input("action: "))
        except:
            continue
        if x == -1: break
        test_game.do_action(x)

def random_play(test_game):
    while not test_game.game_over():
        action = random.choice(test_game.available_actions())
        test_game.do_action(action)
    return test_game.score(), test_game.max_tile()

def plot_results(scores, max_tiles):
    # Biểu đồ điểm số thống kê tương ứng với số lượng màn chơi
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title('Score Distribution across Games')
    plt.grid(True)
    plt.show()

    # Biểu đồ số lượng màn chơi có ô giá trị lớn nhất tương ứng
    unique, counts = np.unique(max_tiles, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.hist(max_tiles, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.xlabel('Best tile')
    plt.ylabel('Frequency')
    plt.title('Best tile distribution')
    plt.grid(True)
    plt.show()

def create_table(max_tiles):
    unique, counts = np.unique(max_tiles, return_counts=True)
    table = pd.DataFrame(data={'Max tile': unique, 'Game count': counts})
    table = table.set_index('Max tile').transpose()
    return table

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    randoms_scores = []
    randoms_max_tiles = []
    for i in range(100):
        game = Game()
        score, max_tile = random_play(game)
        randoms_scores.append(score)
        randoms_max_tiles.append(max_tile)
    print("The mean of the score is {}".format(np.mean(randoms_scores)))
    plot_results(randoms_scores, randoms_max_tiles)
    
    table = create_table(randoms_max_tiles)
    print(table)
