import torch
import torch.nn as nn
import pygame
import random
import math

class DuelingDQN(nn.Module):
    def __init__(self, hidden_dim, drop_out):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out)
        )
        # self.value_stream = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 1)
        # )
        # self.advantage_stream = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 4)  # 4 actions: left, up, right, down
        # )
        
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
model.load_state_dict(torch.load('2048_DQN.pth'))
model.eval()
model = model.to(device)

# Rest of the code remains the same
pygame.init()

FPS = 60
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
MOVE_VEL = 20

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

def get_game_vector(tiles):
    vec = torch.zeros(256)
    for row in range(ROWS):
        for col in range(COLS):
            value = tiles.get(f"{row}{col}")
            if value:
                index = int(math.log2(value.value)) - 1
                vec[row * 4 * 16 + col * 16 + index] = 1
    return vec

def draw_grid(window):
    for row in range(1, ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR, (0, y), (WIDTH, y), OUTLINE_THICKNESS)
    for col in range(1, COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR, (x, 0), (x, HEIGHT), OUTLINE_THICKNESS)
    pygame.draw.rect(window, OUTLINE_COLOR, (0, 0, WIDTH, HEIGHT), OUTLINE_THICKNESS)

def draw(window, tiles):
    window.fill(BACKGROUND_COLOR)
    for tile in tiles.values():
        tile.draw(window)
    draw_grid(window)
    pygame.display.update()

def get_random_pos(tiles):
    row = None
    col = None
    while True:
        row = random.randrange(0, ROWS)
        col = random.randrange(0, COLS)
        if f"{row}{col}" not in tiles:
            break
    return row, col

def move_tiles(window, tiles, clock, direction):
    updated = True
    blocks = set()
    if direction == "left":
        sort_func = lambda x: x.col
        reverse = False
        delta = (-MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col - 1}")
        merge_check = lambda tile, next_tile: tile.x > next_tile.x + MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL
        )
        ceil = True
    elif direction == "right":
        sort_func = lambda x: x.col
        reverse = True
        delta = (MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == COLS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col + 1}")
        merge_check = lambda tile, next_tile: tile.x < next_tile.x - MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.x + RECT_WIDTH + MOVE_VEL < next_tile.x
        )
        ceil = False
    elif direction == "up":
        sort_func = lambda x: x.row
        reverse = False
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row - 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y > next_tile.y + MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.y > next_tile.y + RECT_HEIGHT + MOVE_VEL
        )
        ceil = True
    elif direction == "down":
        sort_func = lambda x: x.row
        reverse = True
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row + 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y < next_tile.y - MOVE_VEL
        move_check = (
            lambda tile, next_tile: tile.y + RECT_HEIGHT + MOVE_VEL < next_tile.y
        )
        ceil = False
    while updated:
        clock.tick(FPS)
        updated = False
        sorted_tiles = sorted(tiles.values(), key=sort_func, reverse=reverse)
        for i, tile in enumerate(sorted_tiles):
            if boundary_check(tile):
                continue
            next_tile = get_next_tile(tile)
            if not next_tile:
                tile.move(delta)
            elif (
                tile.value == next_tile.value
                and tile not in blocks
                and next_tile not in blocks
            ):
                if merge_check(tile, next_tile):
                    tile.move(delta)
                else:
                    next_tile.value *= 2
                    sorted_tiles.pop(i)
                    blocks.add(next_tile)
            elif move_check(tile, next_tile):
                tile.move(delta)
            else:
                continue
            tile.set_pos(ceil)
            updated = True
        update_tiles(window, tiles, sorted_tiles)
    return end_move(tiles)

def is_action_available(tiles, action):
    if action == 0:  # Left
        for row in range(ROWS):
            for col in range(1, COLS):
                if tiles.get(f"{row}{col}") and (tiles.get(f"{row}{col-1}") is None or tiles[f"{row}{col}"].value == tiles[f"{row}{col-1}"].value):
                    return True
    elif action == 1:  # Up
        for col in range(COLS):
            for row in range(1, ROWS):
                if tiles.get(f"{row}{col}") and (tiles.get(f"{row-1}{col}") is None or tiles[f"{row}{col}"].value == tiles[f"{row-1}{col}"].value):
                    return True
    elif action == 2:  # Right
        for row in range(ROWS):
            for col in range(COLS-1):
                if tiles.get(f"{row}{col}") and (tiles.get(f"{row}{col+1}") is None or tiles[f"{row}{col}"].value == tiles[f"{row}{col+1}"].value):
                    return True
    elif action == 3:  # Down
        for col in range(COLS):
            for row in range(ROWS-1):
                if tiles.get(f"{row}{col}") and (tiles.get(f"{row+1}{col}") is None or tiles[f"{row}{col}"].value == tiles[f"{row+1}{col}"].value):
                    return True
    return False

def get_score_and_max_tile(tiles):
    score = sum(tile.value for tile in tiles.values())
    max_tile = max(tile.value for tile in tiles.values())
    return score, max_tile

def end_move(tiles):
    if len(tiles) == 16:
        return "lost"
    row, col = get_random_pos(tiles)
    tiles[f"{row}{col}"] = Tile(random.choice([2, 4]), row, col)
    return "continue"

def update_tiles(window, tiles, sorted_tiles):
    tiles.clear()
    for tile in sorted_tiles:
        tiles[f"{tile.row}{tile.col}"] = tile
    draw(window, tiles)

def generate_tiles():
    tiles = {}
    for _ in range(2):
        row, col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2, row, col)
    return tiles

def main(window):
    clock = pygame.time.Clock()
    run = True
    tiles = generate_tiles()
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        # Sử dụng mô hình để quyết định hành động
        game_vector = get_game_vector(tiles).to(device)
        with torch.no_grad():
            Q_values = model(game_vector)
            Q_valid_values = [Q_values[a] if is_action_available(tiles, a) else float('-inf') for a in range(4)]
            best_action = torch.argmax(torch.tensor(Q_valid_values)).item()
        if best_action == 0:
            result = move_tiles(window, tiles, clock, "left")
        elif best_action == 1:
            result = move_tiles(window, tiles, clock, "up")
        elif best_action == 2:
            result = move_tiles(window, tiles, clock, "right")
        elif best_action == 3:
            result = move_tiles(window, tiles, clock, "down")
        draw(window, tiles)
        if result == "lost":
            score, max_tile = get_score_and_max_tile(tiles)
            print("Game Over! Score is: {} Max tile is: {}".format(score, max_tile))
            run = False
    pygame.quit()

if __name__ == "__main__":
    main(WINDOW)
