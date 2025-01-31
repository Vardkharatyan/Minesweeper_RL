import gymnasium as gym
from env import Minesweeper  # Assuming your Minesweeper environment is in the `env.py` file
from stable_baselines3 import DQN
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt

# Callback to log rewards during training
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Log the reward and print the step number to see progress
        self.rewards.append(self.locals["rewards"])
        print(f"Step {self.num_timesteps} Reward: {self.locals['rewards']}")
        return True

# Initialize the Minesweeper environment
env = Minesweeper()

# Adjust exploration parameters
exploration_kwargs = {
    'exploration_fraction': 0.5,  # Increase the fraction of training time spent exploring
    'exploration_initial_eps': 1.0,  # Start with full exploration
    'exploration_final_eps': 0.05,   # Allow a small exploration rate at the end
    'train_freq': 1,               # frequency of training (default: 1)
}

# Initialize DQN model with adjusted exploration settings
model = DQN("MlpPolicy", env, verbose=2, learning_rate=0.0005,  # Reduced learning rate
            exploration_fraction=exploration_kwargs['exploration_fraction'],
            exploration_initial_eps=exploration_kwargs['exploration_initial_eps'],
            exploration_final_eps=exploration_kwargs['exploration_final_eps'],
            train_freq=exploration_kwargs['train_freq'])

# Training for a larger number of timesteps
timesteps = 1000000 # Increase timesteps for more complete training

# Reward logging callback
reward_callback = RewardLoggerCallback()

# Start training
print("Starting training...")
try:
    model.learn(total_timesteps=timesteps, callback=reward_callback, log_interval=100)
except Exception as e:
    print(f"Training encountered an error: {e}")

# Save the trained model
model.save("minesweeper_dqn_model")
print("Model saved.")

# Testing phase
print("Starting testing phase...")

# Initialize the Minesweeper environment after training
env = Minesweeper()
obs, _ = env.reset()  # Reset the environment and get the initial state
done = False

# List to store frames for creating the GIF
frames = []

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

# Optionally, visualize rewards (to check the learning progress)
rewards = np.array(reward_callback.rewards)
aggregated_rewards = rewards.reshape(-1, 500).mean(axis=1)  # Aggregate rewards for smoother plot
plt.plot(aggregated_rewards)
plt.show()


