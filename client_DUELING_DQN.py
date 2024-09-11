import torch
import torch.nn as nn
import pygame
import random
import math
import time
from game import Game
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
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
        x = self.fc1(x.to(device))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        Q = value + advantage - advantage.mean()
        return Q

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_dim = 128
drop_out = 0.2

model = DuelingDQN(hidden_dim, drop_out)
model.load_state_dict(torch.load('2048_dueling_dqn_model.pth'))
model.eval()
model = model.to(device)

# Rest of the code remains the same
pygame.init()

FPS = 30
WIDTH, HEIGHT = 400, 400
ROWS = 4
COLS = 4
RECT_HEIGHT = HEIGHT // ROWS
RECT_WIDTH = WIDTH // COLS
OUTLINE_COLOR = (187, 173, 160)
OUTLINE_THICKNESS = 10
BACKGROUND_COLOR = (205, 192, 180)
FONT_COLOR = (119, 110, 101)
FONT = pygame.font.SysFont("comicsans", 60, bold=True)
MOVE_VEL = 10  # Reduced move velocity

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")

class Tile:
    COLORS = [
        (237, 229, 218),
        (238, 225, 201),
        (243, 178, 122),
        (246, 150, 101),
        (247, 124, 95),
        (247, 95, 59),
        (237, 208, 115),
        (237, 204, 99),
        (236, 202, 80),
    ]

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.x = col * RECT_WIDTH
        self.y = row * RECT_HEIGHT

    def get_color(self):
        color_index = int(math.log2(self.value)) - 1
        color = self.COLORS[color_index]
        return color

    def draw(self, window):
        color = self.get_color()
        pygame.draw.rect(window, color, (self.x, self.y, RECT_WIDTH, RECT_HEIGHT))
        text = FONT.render(str(self.value), 1, FONT_COLOR)
        window.blit(
            text,
            (
                self.x + (RECT_WIDTH / 2 - text.get_width() / 2),
                self.y + (RECT_HEIGHT / 2 - text.get_height() / 2),
            ),
        )

    def set_pos(self, ceil=False):
        if ceil:
            self.row = math.ceil(self.y / RECT_HEIGHT)
            self.col = math.ceil(self.x / RECT_WIDTH)
        else:
            self.row = math.floor(self.y / RECT_HEIGHT)
            self.col = math.floor(self.x / RECT_WIDTH)

    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]

def get_game_vector(game):
    return game.vector()

def draw_grid(window):
    for row in range(1, ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR, (0, y), (WIDTH, y), OUTLINE_THICKNESS)
    for col in range(1, COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR, (x, 0), (x, HEIGHT), OUTLINE_THICKNESS)
    pygame.draw.rect(window, OUTLINE_COLOR, (0, 0, WIDTH, HEIGHT), OUTLINE_THICKNESS)

def draw(window, game):
    window.fill(BACKGROUND_COLOR)
    for row in range(ROWS):
        for col in range(COLS):
            value = 2 ** game._state[row, col] if game._state[row, col] > 0 else 0
            if value > 0:
                tile = Tile(value, row, col)
                tile.draw(window)
    draw_grid(window)
    pygame.display.update()

def smooth_move(window, game, direction):
    delta_map = {
        ACTION_LEFT: (-RECT_WIDTH // 10, 0),
        ACTION_RIGHT: (RECT_WIDTH // 10, 0),
        ACTION_UP: (0, -RECT_HEIGHT // 10),
        ACTION_DOWN: (0, RECT_HEIGHT // 10)
    }

    initial_positions = {}
    moving_tiles = []

    for row in range(ROWS):
        for col in range(COLS):
            if game._state[row, col] > 0:
                tile = Tile(2 ** game._state[row, col], row, col)
                initial_positions[(row, col)] = (tile.x, tile.y)

    game.do_action(direction)

    for row in range(ROWS):
        for col in range(COLS):
            if game._state[row, col] > 0:
                tile = Tile(2 ** game._state[row, col], row, col)
                if (row, col) in initial_positions:
                    moving_tiles.append(tile)
                    tile.x, tile.y = initial_positions[(row, col)]

    for step in range(10):
        window.fill(BACKGROUND_COLOR)
        for tile in moving_tiles:
            tile.move(delta_map[direction])
            tile.draw(window)
        draw_grid(window)
        pygame.display.update()
        pygame.time.delay(1)

    draw(window, game)

def main(window):
    clock = pygame.time.Clock()
    run = True
    game = Game()

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        # Sử dụng mô hình để quyết định hành động
        game_vector = get_game_vector(game).to(device)
        with torch.no_grad():
            Q_values = model(game_vector)
            Q_valid_values = [Q_values[a] if game.is_action_available(a) else float('-inf') for a in range(4)]
            best_action = torch.argmax(torch.tensor(Q_valid_values)).item()

        smooth_move(WINDOW, game, best_action)
        draw(WINDOW, game)
        time.sleep(0.01)  # Adding delay between actions

        if game.game_over():
            score = game.score()
            max_tile = game.max_tile()
            print("Game Over! Score is: {} Max tile is: {}".format(score, max_tile))
            run = False

    pygame.quit()

if __name__ == "__main__":
    main(WINDOW)
