import numpy as np
import pylab as plt
import networkx as nx

points_list = [(0, 2), (1, 5), (4, 6), (5, 4), (1, 2), (2, 3), (2, 7), (3, 9), (6, 10),
               (5, 9), (5, 6), (8, 9), (9, 10), (0, 8)]

goal = 10
prize = [4]
down = [9]
MATRIX_SIZE = 11
np.random.seed(2022)

# Visualization
G = nx.Graph()
G.add_edges_from(points_list)
mapping = {0: '0/Start', 1: '1', 2: '2', 3: '3', 4: '4/prize', 5: '5',
           6: '6', 7: '7', 8: '8', 9: '9/lose', 10: '10/Fin'}
G = nx.relabel_nodes(G, mapping)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G, pos, node_size=[3000, 90, 90, 90, 1500, 90, 90, 90, 1500, 3000, 90], node_color='green')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.title('Environment Map', fontsize=16)
plt.show()

# Initialize matrices (R, Q, Enviro)

R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE))) * -1

for point in points_list:
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0
    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0

R[goal, goal] = 100

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
enviro_matrix = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Hyperparameters
gamma = 0.85
epsilon = 0.3
epochs = 2000

#  Helper functions

def available_actions_with_enviro_help(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]

    # Filter: if we know a path leads to a heavy penalty, avoid it during exploration
    env_row = enviro_matrix[state, av_act]
    if np.any(env_row < -100):
        safe_traj = av_act[np.array(env_row)[0] >= 0]
        if len(safe_traj) > 0:
            av_act = safe_traj
    return av_act


def update(current_state, action, gamma):
    act_int = int(action)

    # Find max Q for the next state
    max_index = np.where(Q[act_int,] == np.max(Q[act_int,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    # Environmental rewards logic
    env_reward = 0
    if act_int in prize:
        enviro_matrix[current_state, act_int] = 150  # High Prize
        env_reward = 150
    elif act_int in down:
        enviro_matrix[current_state, act_int] = -200  # Heavy Penalty
        env_reward = -200

    # Total reward = static R + dynamic enviro
    total_reward = R[current_state, act_int] + env_reward

    # Q-Learning update (Bellman Equation)
    Q[current_state, act_int] = total_reward + gamma * Q[act_int, max_index]

    if np.max(Q) > 0:
        return np.sum(Q / np.max(Q) * 100)
    return 0

# Training phase

scores = []
for i in range(epochs):
    # Balanced start: 50% start at beginning, 50% random
    current_state = 0 if np.random.rand() < 0.5 else np.random.randint(0, MATRIX_SIZE)

    # Choose action using epsilon-greedy
    available_act = available_actions_with_enviro_help(current_state)

    if np.random.rand() < epsilon:
        action = int(np.random.choice(available_act, 1))
    else:
        # Exploit the best known Q-value for available actions
        state_actions = Q[current_state, available_act]
        action = available_act[np.argmax(state_actions)]

    score = update(current_state, action, gamma)
    scores.append(score)

# Testing phase

current_state = 0
steps = [current_state]
total_raw_points = 0
total_discounted_points = 0
k = 0  # Step counter for discounting

while current_state != goal and len(steps) < 20:
    next_step = int(np.argmax(Q[current_state,]))

    # Identify reward for this specific move
    move_reward = R[current_state, next_step]
    if next_step in prize:
        move_reward += 150
    elif next_step in down:
        move_reward -= 200

    # Raw sum
    total_raw_points += move_reward

    # Discounted sum: (gamma^step) * reward
    total_discounted_points += (gamma ** k) * move_reward

    steps.append(next_step)
    current_state = next_step
    k += 1

print("\n--- RESULTS ---")
print(f"Most efficient path: {steps}")
print(f"Total raw points won: {total_raw_points}")
print(f"Total discounted value: {total_discounted_points:.2f}")

# Plotting the convergence
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title('Q-Learning Convergence')
plt.xlabel('Iterations')
plt.ylabel('Normalized Score')
plt.grid(True)
plt.show()