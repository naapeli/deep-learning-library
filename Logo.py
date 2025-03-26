import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Deep Learning Logo")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the nodes (neurons) positions
nodes_layer_1 = [(150, 250), (150, 350), (150, 450)]  # First layer (1 fewer neuron)
nodes_layer_2 = [(300, 200), (300, 300), (300, 400), (300, 500)]  # Second layer
nodes_layer_3 = [(500, 200), (500, 300), (500, 400), (500, 500)]  # Third layer
nodes_layer_4 = [(650, 250), (650, 350), (650, 450)]  # Last layer (1 fewer neuron)

# Function to draw a line between two points
def draw_line(start, end):
    pygame.draw.line(screen, WHITE, start, end, 2)

# Function to draw a node (circle)
def draw_node(position, radius=15):
    pygame.draw.circle(screen, WHITE, position, radius)

# Main loop
running = True
while running:
    screen.fill(BLACK)  # Fill the screen with black

    # Draw the nodes (neurons)
    for node in nodes_layer_1:
        draw_node(node)
    for node in nodes_layer_2:
        draw_node(node)
    for node in nodes_layer_3:
        draw_node(node)
    for node in nodes_layer_4:
        draw_node(node)

    # Draw the lines (connections between the layers)
    for node_1 in nodes_layer_1:
        for node_2 in nodes_layer_2:
            draw_line(node_1, node_2)
    for node_2 in nodes_layer_2:
        for node_3 in nodes_layer_3:
            draw_line(node_2, node_3)
    for node_3 in nodes_layer_3:
        for node_4 in nodes_layer_4:
            draw_line(node_3, node_4)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the screen
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
