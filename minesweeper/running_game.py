from env import Minesweeper  # Import the Minesweeper environment
import pygame
import time
import imageio

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 450  # Adjusted height for intro screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper")
frames=[]
# Load background image
background = pygame.image.load("images/bg.jpg")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100,165,200)

# Fonts
font = pygame.font.Font(None, 36)
button_font = pygame.font.Font(None, 30)

def draw_intro():
    screen.blit(background, (0, 0))  # Set background image
    title_text = font.render("Welcome to Minesweeper", True, BLACK)
    play_text = button_font.render("Play", True, WHITE)
    
    # Button dimensions
    button_rect = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2, 100, 40)
    pygame.draw.rect(screen, BLUE, button_rect)
    
    # Draw text
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 3))
    screen.blit(play_text, (button_rect.x + 30, button_rect.y + 10))
    
    pygame.display.flip()
    return button_rect

# Show intro screen
button_rect = draw_intro()
intro = True
while intro:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                intro = False  # Start game when button is clicked

# Initialize the environment
env = Minesweeper()
obs, _ = env.reset()
start_time=time.time()
max_duration=120
# Main game loop
running = True
while running:
    elapsed_time=time.time()-start_time

    if elapsed_time >= max_duration:
        break
    should_exit = False
    done = False  # Define done at the beginning of the loop

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_exit = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            mx //= 40  # Get x-coordinate in grid units (tile size)
            my //= 40  # Get y-coordinate in grid units (tile size)

            # Ensure the click is within the game grid bounds
            if 0 <= mx < 9 and 0 <= my < 9:
                action = mx + my * 9  # Convert 2D to 1D
                _, reward, done, _, _ = env.step(action)

    env.render()
    frame=pygame.surfarray.array3d(screen)
    frame=frame.swapaxes(0,1)
    frames.append(frame)
    if done or should_exit:
        break
gif_path="MineSweeper.gif"
imageio.mimsave(gif_path,frames,duration=1/30)
env.close()
