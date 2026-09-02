import pygame
import sys
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from pygame.math import Vector2

pygame.init()

# Fonts & Colors matching the Retro UI style
title_font = pygame.font.Font(None, 60)
score_font = pygame.font.Font(None, 40)

GREEN = (173, 204, 96)
DARK_GREEN = (43, 51, 24)

# Grid Layout Settings
CELL_SIZE = 30
NUMBER_OF_CELLS = 20
OFFSET = 75
BLOCK_SIZE = CELL_SIZE  # Maintained for compatibility with Agent logic
SPEED = 60  # Execution tick speed for AI agent loop

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:

    def __init__(self, w=NUMBER_OF_CELLS*CELL_SIZE, h=NUMBER_OF_CELLS*CELL_SIZE):
        self.w = w
        self.h = h

        # Initialize screen with header and border padding
        self.screen = pygame.display.set_mode((2 * OFFSET + self.w, 2 * OFFSET + self.h))
        pygame.display.set_caption("Retro Snake - AI Agent")
        self.clock = pygame.time.Clock()

        # Load audio assets safely
        try:
            self.eat_sound = pygame.mixer.Sound("Snake/Sounds/eat.mp3")
            self.wall_hit_sound = pygame.mixer.Sound("Snake/Sounds/wall.mp3")
        except pygame.error:
            self.eat_sound = None
            self.wall_hit_sound = None

        # Load graphic assets safely
        try:
            self.food_surface = pygame.image.load("Snake/Graphics/food.png")
            self.food_surface = pygame.transform.scale(self.food_surface, (CELL_SIZE, CELL_SIZE))
        except pygame.error:
            self.food_surface = None

        self.reset()

    def reset(self):
        # Reset movement and orientation
        self.direction = Direction.RIGHT

        # Initialize head position at center grid block
        center_x = (NUMBER_OF_CELLS // 2) * CELL_SIZE
        center_y = (NUMBER_OF_CELLS // 2) * CELL_SIZE
        self.head = Point(center_x, center_y)

        # Build initial snake segments
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, NUMBER_OF_CELLS - 1) * BLOCK_SIZE
        y = random.randint(0, NUMBER_OF_CELLS - 1) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Process Pygame events (keep window responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Measure distance to food BEFORE moving
        old_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        # 2. Update direction and advance head position
        self._move(action)
        self.snake.insert(0, self.head)

        # Measure distance to food AFTER moving
        new_dist = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        # 3. Evaluate terminal loss states (wall collision, self-collision, or execution loops)
        reward = 0
        game_over = False

        # Tighter loop limit: 50 steps per snake segment prevents endless circling
        if self.is_collision() or self.frame_iteration > 50 * len(self.snake):
            game_over = True
            reward = -10
            if self.wall_hit_sound:
                self.wall_hit_sound.play()
            return reward, game_over, self.score

        # 4. Reward shaping to prevent tail-chasing
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.frame_iteration = 0  # Reset loop counter on food capture
            if self.eat_sound:
                self.eat_sound.play()
            self._place_food()
        else:
            self.snake.pop()
            # Distance guidance + step penalty to stop aimless looping
            if new_dist < old_dist:
                reward = 0.1   # Reward for moving closer to food
            else:
                reward = -0.25 # Stronger penalty for moving away from food

        # 5. Redraw game interface
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Expose step status output to agent
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Check arena boundaries
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True

        # Check self-intersection
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        # Draw background and retro outer line frame
        self.screen.fill(GREEN)
        pygame.draw.rect(
            self.screen,
            DARK_GREEN,
            (OFFSET - 5, OFFSET - 5, self.w + 10, self.h + 10),
            5
        )

        # Render snake elements with rounded rectangles
        for pt in self.snake:
            segment_rect = pygame.Rect(OFFSET + pt.x, OFFSET + pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, DARK_GREEN, segment_rect, 0, 7)

        # Render food item
        food_rect = pygame.Rect(OFFSET + self.food.x, OFFSET + self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        if self.food_surface:
            self.screen.blit(self.food_surface, food_rect)
        else:
            pygame.draw.rect(self.screen, DARK_GREEN, food_rect, 0, 7)

        # Render header titles and real-time score indicators
        title_surface = title_font.render("Retro Snake AI", True, DARK_GREEN)
        score_surface = score_font.render(f"Score: {self.score}", True, DARK_GREEN)
        self.screen.blit(title_surface, (OFFSET - 5, 20))
        self.screen.blit(score_surface, (OFFSET - 5, OFFSET + self.h + 10))

        pygame.display.flip()

    def _move(self, action):
        # Direction translation map for clockwise orientation
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Keep current direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Turn 90 deg right
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Turn 90 deg left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)