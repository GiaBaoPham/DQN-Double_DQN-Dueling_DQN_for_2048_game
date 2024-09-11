from game import Game
import numpy as np
import random
import pygame
import math

def get_neighbors(game):
    neighbors = [game.copy() for a in game.available_actions()]
    for i, a in enumerate(game.available_actions()):
        neighbors[i].do_action(a)
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
    return game.score()

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

def smooth_move(window, game, direction):
    delta_map = {
        0: (-MOVE_VEL, 0),
        1: (0, -MOVE_VEL),
        2: (MOVE_VEL, 0),
        3: (0, MOVE_VEL)
    }

    delta = delta_map[direction]
    initial_positions = {}
    target_positions = {}

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
                target_positions[(row, col)] = (tile.x, tile.y)

    for i in range(RECT_WIDTH // MOVE_VEL):
        window.fill(BACKGROUND_COLOR)
        for row in range(ROWS):
            for col in range(COLS):
                if game._state[row, col] > 0:
                    tile = Tile(2 ** game._state[row, col], row, col)
                    initial_x, initial_y = initial_positions.get((row, col), (tile.x, tile.y))
                    target_x, target_y = target_positions.get((row, col), (tile.x, tile.y))
                    tile.x = initial_x + delta[0] * i
                    tile.y = initial_y + delta[1] * i
                    tile.draw(window)
        draw_grid(window)
        pygame.display.update()
        pygame.time.delay(1)

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

        # Use beam search to determine the best action
        best_action = beam_search(game, width=search_width, depth=search_depth)

        smooth_move(WINDOW, game, best_action)
        draw(WINDOW, game)
        pygame.time.delay(500)  # Adding delay between actions

        if game.game_over():
            score = game.score()
            max_tile = game.max_tile()
            print("Game Over! Score is: {} Max tile is: {}".format(score, max_tile))
            run = False

    pygame.quit()

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
search_width = 16
search_depth = 5
scores = []

if __name__ == "__main__":
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
    MOVE_VEL = 20

    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048")

    main(WINDOW)
