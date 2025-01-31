import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROWS = 9
COLS = 9
WIDTH = 400
HEIGHT = 400
TILESIZE = 40
AMOUNT_MINES = 10
FPS = 60
TITLE = "Minesweeper"

NUMBER_COLORS = {
    1: (0, 0, 255),  # Blue
    2: (0, 255, 0),  # Green
    3: (255, 0, 0),  # Red
    4: (0, 0, 139),  # Dark Blue
    5: (139, 0, 0),  # Dark Red
    6: (0, 255, 255),  # Cyan
    7: (128, 0, 128),  # Purple
    8: (0, 0, 0)  # Black
}

class Minesweeper(gym.Env):
    def __init__(self):
        super(Minesweeper, self).__init__()

        # Initialize the environment variables
        self.true_state = self.generate_board()  # The actual game board (mines + numbers)
        self.revealed = np.zeros((ROWS, COLS), dtype=bool)  # Track which tiles are revealed
        self.flags = np.zeros((ROWS, COLS), dtype=bool)  # Track flagged tiles

        # State is the board of opened tiles (-1 = unopened, 0 = opened, >0 = number, -2 = bomb clicked)
        self.state = np.full((ROWS, COLS), -1, dtype=int)

        # Action space is a single tile click (rows x cols)
        self.action_space = spaces.Discrete(ROWS * COLS)

        # Observation space is the state of the game
        self.observation_space = spaces.Box(low=-1, high=9, shape=(ROWS, COLS), dtype=np.int32)

        # Pygame setup
        pygame.init()  # Initialize Pygame
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create a window for the game
        pygame.display.set_caption(TITLE)  # Set the window title
        self.clock = pygame.time.Clock()  # Set up the clock for FPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.true_state = self.generate_board()  # Reset the board with new mines and numbers
        self.revealed.fill(False)
        self.flags.fill(False)
        self.state = np.full((ROWS, COLS), -1, dtype=int)  # Reset state to all unopened

        print("Environment reset.")  # Debugging line to confirm reset
        return self.state, {}


    def step(self, action):
        x, y = divmod(action, COLS)  # Convert action to grid coordinates

        reward = 0
        done = False

        # Check if the action has already been taken (cell revealed or flagged)
        if self.state[x, y] != -1:  # -1 means it's unopened
            return self.state, -1000, done, False, {}  # Don't do anything if already revealed or flagged

        if self.true_state[x, y] == -1:  # Bomb clicked, game over
            self.state[x, y] = -2  # Mark bomb as clicked
            reward = -10  # Large negative reward for clicking a bomb
            done = True
        else:  # Reveal the tile and check if it's empty
            self.reveal(x, y)
            reward = 1  # Positive reward for revealing a safe tile
            
            # If the agent revealed an empty tile (no number), give it a better reward
            if self.true_state[x, y] == 0:
                reward = 2  # Give higher reward for uncovering a "safe" empty tile

        # Check if the game has ended (all non-mine tiles revealed)
        if np.all(self.revealed[self.true_state != -1]):
            reward = 1000  # Larger positive reward for winning the game
            done = True

        self.state = self.get_state()  # Update the state
        return self.state, reward, done, False, {}






    def render(self):
        self.screen.fill((255, 255, 255))  # Fill the screen with a white background

        # Draw the grid
        for x in range(ROWS):
            for y in range(COLS):
                rect = pygame.Rect(y * TILESIZE, x * TILESIZE, TILESIZE, TILESIZE)

                if self.state[x, y] == -1:  # Unrevealed tile
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                elif self.state[x, y] == -2:  # Bomb clicked
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                elif self.true_state[x, y] == 0:  # Empty tile revealed
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)
                else:  # Numbered tile revealed
                    pygame.draw.rect(self.screen, (180, 180, 180), rect)

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Grid borders

                # Draw numbers on revealed tiles with custom colors
                if self.state[x, y] >= 0:
                    font = pygame.font.SysFont("Arial", 20)
                    text = font.render(str(self.true_state[x, y]) if self.true_state[x, y] > 0 else "", True, 
                                       NUMBER_COLORS.get(self.true_state[x, y], (0, 0, 0)))
                    self.screen.blit(text, rect.move(10, 10))

        pygame.display.flip()  # Update the display after drawing everything
        return self.screen  # Return the screen surface to capture it for the GIF

    def close(self):
        pygame.quit()  # Ensure Pygame quits properly

    def generate_board(self):
        board = np.zeros((ROWS, COLS), dtype=int)
        # Place bombs randomly
        mines = random.sample(range(ROWS * COLS), AMOUNT_MINES)
        for mine in mines:
            x, y = divmod(mine, COLS)
            board[x, y] = -1  # -1 for mines

        # Set numbers for the surrounding tiles
        for x in range(ROWS):
            for y in range(COLS):
                if board[x, y] == -1:  # Skip mines
                    continue
                board[x, y] = self.count_adjacent_mines(x, y, board)

        return board

    def count_adjacent_mines(self, x, y, board):
        adjacent_mines = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < ROWS and 0 <= ny < COLS and board[nx, ny] == -1:
                    adjacent_mines += 1
        return adjacent_mines

    def reveal(self, x, y):
        if not (0 <= x < ROWS and 0 <= y < COLS) or self.revealed[x, y]:
            return  # Do nothing if out of bounds or already revealed

        self.revealed[x, y] = True
        if self.true_state[x, y] == 0:  # If it's an empty tile, reveal adjacent tiles
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    self.reveal(x + dx, y + dy)

    def get_state(self):
        # Return the state representation: -1 for unopened, 0 for opened empty, >0 for opened numbered tile, -2 for clicked bomb
        state = np.full((ROWS, COLS), -1, dtype=int)
        state[self.revealed] = self.true_state[self.revealed]
        state[self.state == -2] = -2  # Mark bombs clicked
        return state
