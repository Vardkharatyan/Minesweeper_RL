# Minesweeper_RL
[Copy of Gaming Design Portfolio by Slidesgo.pptx](https://github.com/user-attachments/files/18619074/Copy.of.Gaming.Design.Portfolio.by.Slidesgo.pptx)
![minesweeper_game](https://github.com/user-attachments/assets/75f099b8-0c40-4067-9382-27f385e8980e)
![MineSweeper (1)](https://github.com/user-attachments/assets/a8b4ee88-1b61-4bf6-9c8c-0ff546042775)

MINESWEEPER

Made by 
Vardansuh Kharatyan & Nare Khalatyan

Minesweeper Environment 


Minesweeper is a partially observable, grid-based environment where an agent interacts with a minefield by revealing tiles while avoiding hidden mines. The goal is to uncover all non-mine tiles while strategically marking the mines.The main goal is to open all cells, except for the ones containing mines.

 

self.action_space = spaces.Discrete(ROWS * COLS)

self.observation_space = spaces.Box(low=-1, high=9, shape=(ROWS, COLS), dtype=np.int32)



Used technologies & Libraries

Gymnasium 
Creating environments


Stable Baselines
RL Algorithm Implementations


Pygame 
Rendering

Matplotlib
Plotting
RL terms

State-The current state of the game board, which includes revealed and hidden tiles, flagged mines, and numbers indicating nearby mines.

Action-The feedback received based on the action—uncovering an empty tile gives a neutral or positive reward, while hitting a mine results in a negative reward.

Reward-The move taken by the player (or agent), such as clicking on a tile to reveal it or flagging a suspected mine.
