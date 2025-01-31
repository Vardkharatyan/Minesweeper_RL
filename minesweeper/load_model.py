from stable_baselines3 import DQN
from env import Minesweeper  # Import the Minesweeper environment
import pygame
import time

env = Minesweeper()
obs, _ = env.reset()

model = DQN.load("minesweeper_dqn_model.zip", env=env)

# List to store frames for creating the GIF
frames = []

done = False
# Run the trained model and capture frames for the GIF
while not done:
    action, _ = model.predict(obs, deterministic=True)  # Get the action from the model
    print(f"Action taken: {action}")  # Log the action taken

    # Step in the environment with the action
    obs, reward, done, _, _ = env.step(action)

    # Render the current game state to the screen
    screen = env.render()  # Return the pygame surface representing the current game state

    # Convert the pygame surface to a numpy array for creating the GIF
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)  # Swap axes for correct orientation
    frames.append(frame)  # Add the frame to the list

    time.sleep(0.7)  # Add a small delay to visualize the gameplay in real-time

    if done:
        print("Game Over!")  # Print when the game finishes
        break



# Save the frames as a GIF
gif_path = "minesweeper_game.gif"
imageio.mimsave(gif_path, frames, duration=1/30)  # 30 FPS for the GIF
print(f"GIF saved to {gif_path}.")

# Close the environment
env.close()