# Retro Snake AI — Deep Q-Learning (DQN)

An intelligent Snake game powered by **Deep Q-Learning (DQN)** using **PyTorch** and **Pygame**.

The AI learns to navigate the board, find food, avoid walls and its own body, and improve its performance through reinforcement learning and experience replay.

## Features

* **16-Feature State Representation**
  Encodes immediate danger, movement direction, food location, snake length, and tail position to give the agent spatial awareness of its environment.

* **Tail & Self-Collision Awareness**
  Tracks the relative position of the snake's tail and normalized body length, helping the agent recognize situations that can lead to self-trapping and tail-chasing loops.

* **Distance-Based Reward Shaping**
  Rewards movement toward food while penalizing movement away from food, idle behavior, collisions, and inefficient navigation.

* **Experience Replay**
  Stores previous experiences in a replay buffer and samples batches during training to improve learning stability and reduce correlation between consecutive experiences.

* **Automatic Model Checkpointing**
  Saves the neural network weights to `./model/model.pth` whenever a new high score is achieved, allowing training to continue across restarts.

* **Real-Time Training Visualization**
  Uses Matplotlib to display current scores and running mean performance while the agent trains.

## Project Structure

```text
.
├── agent.py          # AI agent, state representation, memory, and training loop
├── snake.py          # Pygame environment, rendering, and game logic
├── model.py          # PyTorch neural network and Q-learning trainer
├── helper.py         # Real-time training visualization
├── model/
│   └── model.pth     # Saved neural network weights
└── Snake/
    ├── Graphics/
    │   └── food.png
    └── Sounds/
        ├── eat.mp3
        └── wall.mp3
```

## Installation

### Prerequisites

* Python 3.8+
* pip

### Dependencies

Install the required packages:

```bash
pip install pygame torch numpy matplotlib
```

## Running the AI

Start the training process with:

```bash
python agent.py
```

When launched, the system:

1. Loads an existing `./model/model.pth` checkpoint if available.
2. Starts the Snake environment.
3. Observes the current game state.
4. Selects an action using an epsilon-greedy policy.
5. Receives a reward from the environment.
6. Trains on the most recent experience.
7. Stores the experience in replay memory.
8. Performs long-term training after each game.
9. Updates the live performance graph.
10. Saves the model whenever a new record score is achieved.

This allows the agent to **train → save → restart → continue training** without losing its learned weights.

## Deep Reinforcement Learning

### State Representation

The agent receives a **16-element state vector** describing the current environment.

| Feature Category   | Inputs | Description                                        |
| ------------------ | -----: | -------------------------------------------------- |
| Immediate Danger   |      3 | Collision checks for straight, right, and left     |
| Movement Direction |      4 | Current Left, Right, Up, or Down direction         |
| Food Location      |      4 | Food position relative to the snake's head         |
| Body Length        |      1 | Snake length normalized relative to board capacity |
| Tail Location      |      4 | Tail position relative to the snake's head         |

**Total: 16 input features**

### Neural Network Architecture

```text
Input
16 Features
    │
    ▼
┌───────────────┐
│ Linear Layer  │
│ 16 → 256      │
└───────┬───────┘
        │
       ReLU
        │
        ▼
┌───────────────┐
│ Linear Layer  │
│ 256 → 128     │
└───────┬───────┘
        │
       ReLU
        │
        ▼
┌───────────────┐
│ Linear Layer  │
│ 128 → 3       │
└───────┬───────┘
        │
        ▼
  Q-Values
```

### Actions

The network predicts Q-values for three possible actions:

```text
[Straight, Turn Right, Turn Left]
```

The action with the highest predicted Q-value is selected when the agent is exploiting its learned policy.

During exploration, the agent randomly selects an action according to its epsilon-greedy exploration rate.

## Reward Structure

| Event                 |  Reward |
| --------------------- | ------: |
| Eating food           | `+10.0` |
| Moving closer to food |  `+0.1` |
| Moving away from food | `-0.25` |
| Collision / timeout   | `-10.0` |

Reward shaping gives the agent a learning signal not only for successfully eating food, but also for making progress toward its objective.

## Training & Persistence

The model is checkpointed whenever the agent achieves a new record:

```python
agent.model.save('model.pth')
```

At startup, the checkpoint is loaded:

```python
self.model.load(file_name='model.pth')
```

This means the AI does not have to relearn everything from the beginning after the program is closed.

```text
             TRAINING
                 │
                 ▼
          New High Score?
            /         \
          No           Yes
          │             │
          │             ▼
          │       Save model.pth
          │             │
          └──────┬──────┘
                 ▼
            Next Game
                 │
                 ▼
          Continue Training
```

## Real-Time Metrics

During training, the project tracks:

* Current game score
* Highest score achieved
* Running mean score
* Number of games played

The live graph makes it possible to observe whether the agent's performance is improving over time.

## Customization

### Training Speed

The game speed can be adjusted in `snake.py`.

For example:

```python
SPEED = 60
```

Increasing the value can significantly accelerate training, although extremely high speeds may make visual monitoring less useful.

### Exploration

The exploration rate is controlled in `agent.py`:

```python
self.epsilon = max(1, 80 - self.n_games)
```

This controls the balance between:

* **Exploration** — trying random actions
* **Exploitation** — choosing the action predicted to have the highest Q-value

As training progresses, exploration gradually decreases.

## Technologies

* **Python** — Core programming language
* **PyTorch** — Neural network and Deep Q-Learning
* **Pygame** — Game environment and rendering
* **NumPy** — Numerical operations and state representation
* **Matplotlib** — Training visualization

## Learning Objectives

This project was built to explore practical concepts in reinforcement learning and neural networks, including:

* Deep Q-Learning
* Reinforcement learning
* Q-values
* Epsilon-greedy exploration
* Experience replay
* Reward shaping
* Neural network training
* Model checkpointing
* Persistent AI training
* Game-state representation

## Roadmap

* [x] Build Snake environment
* [x] Implement AI agent
* [x] Create 16-feature state representation
* [x] Implement Deep Q-Learning
* [x] Add experience replay
* [x] Add reward shaping
* [x] Add real-time training visualization
* [x] Add persistent model checkpoints
* [ ] Improve training efficiency
* [ ] Experiment with different network architectures
* [ ] Compare reward strategies
* [ ] Evaluate long-term training performance

## Author

**Soala Amachree**

Mechatronics Engineering Student | AI & Software Developer

Exploring **Artificial Intelligence, Robotics, Systems Programming, and Machine Learning**.
