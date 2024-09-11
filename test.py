from game import Game
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd

def get_neighbors(game):
    neighbors = [game.copy() for a in game.available_actions()]
    for i, a in enumerate(game.available_actions()): neighbors[i].do_action(a)
    return neighbors

def beam_search(g, width=16, depth=5):
    beam = [(a, game) for a, game in zip(g.available_actions(), get_neighbors(g))]
    for i in range(depth):
        neighbors = [(b[0], n) for b in beam for n in get_neighbors(b[1])]
        scores = np.array(list(map(lambda x: x[1].eval(), neighbors)))
        indexes = np.argsort(scores)[-width:]
        beam = np.array(neighbors)[indexes]
    return beam[-1][0] if len(beam) != 0 else g.available_actions()[0]

def play_game(game):
    while not game.game_over():
        game.do_action(beam_search(game, width=search_width, depth=search_depth))
    print("score is: {} max tile is: {}".format(game.score(), game.max_tile()))
    return game.score(), game.max_tile()

def plot_results(scores, max_tiles):
    # 1. Biểu đồ Điểm Số Thống Kê Tương Ứng với Số Lượng Màn Chơi
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title('Score Distribution across Games')
    plt.grid(True)
    plt.show()

    # 2. Biểu đồ Số Lượng Màn Chơi Có Ô Giá Trị Lớn Nhất Tương Ứng
    unique, counts = np.unique(max_tiles, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts, color='orange', edgecolor='black', alpha=0.7, width=2)
    plt.xlabel('Best tile')
    plt.ylabel('Frequency')
    plt.title('Best tile distribution')
    plt.xticks(unique)
    plt.grid(True)
    plt.show()

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
search_width = 16
search_depth = 5
scores = []
max_tiles = []

if __name__ == "__main__":
    for i in range(100):
        game = Game()
        score, max_tile = play_game(game)
        scores.append(score)
        max_tiles.append(max_tile)
    print("The mean of the score is {}".format(np.mean(scores)))
    plot_results(scores, max_tiles)
