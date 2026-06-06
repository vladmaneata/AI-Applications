import random
import numpy as np

# Define the environment for Tic-Tac-Toe
class TicTacToe:
    def __init__(self):
        # 0: empty, 1: 'X', -1: 'O'
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Start with player 'X'
        self.status = "playing" # "playing", "win", "draw", "loss"

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.status = "playing"
        return self.get_state()

    def is_valid_action(self, action):
        row, col = action
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def step(self, action):
        if not self.is_valid_action(action):
            # Invalid move should ideally not happen with a smart agent,
            # but we can penalize it heavily if it does.
            return self.get_state(), -10, True # Severe penalty for invalid move

        row, col = action
        self.board[row, col] = self.current_player

        reward = 0
        done = False

        if self.check_win(self.current_player):
            reward = 10  # Reward for winning
            self.status = "win"
            done = True
        elif self.is_draw():
            reward = 0   # Neutral reward for a draw
            self.status = "draw"
            done = True
        else:
            self.current_player *= -1 # Switch player

        # Penalize the opponent for winning (from the current player's perspective)
        if done and self.status == "win" and self.current_player == -1:
             reward = -10 # Penalty for opponent winning

        return self.get_state(), reward, done

    def check_win(self, player):
        # Check rows
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True
        # Check columns
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True
        # Check diagonals
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != 0) and not self.check_win(1) and not self.check_win(-1)

    def get_state(self):
        # Return a tuple representation of the board for hashing
        return tuple(self.board.flatten())

    def get_valid_actions(self):
        valid_actions = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    valid_actions.append((row, col))
        return valid_actions

    def render(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("-------------")
        for row in range(3):
            print("|", symbols[self.board[row, 0]], "|", symbols[self.board[row, 1]], "|", symbols[self.board[row, 2]], "|")
            print("-------------")

# Define the Q-learning agent
class QlearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {} # Stores Q-values: {(state, action): value}
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate

    def get_state_key(self, state):
        # Convert the state tuple to a string or other hashable format if needed,
        # but the tuple itself is hashable and works as a dictionary key.
        return state

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)

        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random valid action
            return random.choice(valid_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = {}
            for action in valid_actions:
                action_key = (state_key, action)
                # Get Q-value, default to 0 if never seen
                q_values[action] = self.q_table.get(action_key, 0.0)

            # Choose action with the maximum Q-value
            max_q = -float('inf')
            best_action = None
            for action, q_value in q_values.items():
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
                # Handle ties randomly to encourage exploration among equally good actions
                elif q_value == max_q and random.uniform(0, 1) < 0.5:
                     best_action = action

            if best_action is None: # Fallback in case of no valid actions (shouldn't happen in a valid state)
                 return random.choice(valid_actions)

            return best_action

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        action_key = (state_key, action)

        current_q = self.q_table.get(action_key, 0.0)

        if done:
            # If the episode is finished, there is no future reward
            new_q = current_q + self.alpha * (reward - current_q)
        else:
            # Estimate maximum future reward from the next state
            next_state_key = self.get_state_key(next_state)
            max_future_q = 0.0

            # Corrected: Iterate through the flattened next_state tuple
            next_valid_actions = []
            for k in range(9): # There are 9 positions on the 3x3 board
                if next_state[k] == 0: # If the position is empty
                    row = k // 3
                    col = k % 3
                    next_valid_actions.append((row, col))

            if next_valid_actions: # Only consider future Q if there are valid moves
                 max_future_q = max([self.q_table.get((next_state_key, next_action), 0.0) for next_action in next_valid_actions])

            # Q-learning update rule
            new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        self.q_table[action_key] = new_q
# Training the agent
if __name__ == "__main__":
    env = TicTacToe()
    agent_x = QlearningAgent(epsilon=0.1) # Agent for 'X'
    agent_o = QlearningAgent(epsilon=0.1) # Agent for 'O' (can also train two agents against each other)

    num_episodes = 50000
    print("\nTraining ... Just wait!")
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            if env.current_player == 1: # 'X's turn
                action = agent_x.choose_action(state, valid_actions)
                next_state, reward, done = env.step(action)
                agent_x.learn(state, action, reward, next_state, done) # Agent X learns based on its move's outcome
                state = next_state # Move to the next state

            if not done: # Check if game ended after X's move
                valid_actions = env.get_valid_actions()
                if valid_actions: # Check if there are still valid moves for O
                     if env.current_player == -1: # 'O's turn
                        action = agent_o.choose_action(state, valid_actions)
                        next_state, reward, done = env.step(action)
                        # We are training agent_x from the perspective of 'X'.
                        # The reward for agent_o's move affects agent_o's learning,
                        # but we can simplify here by having agent_x learn from the game's outcome
                        # regardless of who made the final move.
                        agent_o.learn(state, action, reward * -1, next_state, done) # Agent O learns (reward is inverted from O's perspective)
                        state = next_state # Move to the next state
                else: # No valid moves for O, check for draw
                    if env.is_draw():
                         reward = 0
                         done = True
                         # No explicit learning for draw state here in this simple example,
                         # the lack of win/loss reward implicitly teaches avoidance.


        # if episode % 1000 == 0:
        #    print(f"Episode {episode}/{num_episodes} finished. Status: {env.status}")

    print("\nTraining finished.")
    print(f"Size of Q-table for Agent X: {len(agent_x.q_table)}")
    print(f"Size of Q-table for Agent O: {len(agent_o.q_table)}")

    # Optional: Play a game against the trained agent
    print("\nPlaying a game against the trained Agent X ('X'):")
    env.reset()
    env.current_player = 1 # Human plays as 'O'
    human_player = -1
    agent_player = 1

    while env.status == "playing":
        env.render()
        if env.current_player == human_player:
            valid_actions = env.get_valid_actions()
            while True:
                try:
                    row, col = map(int, input("Enter your move (row, col): ").split())
                    if (row, col) in valid_actions:
                        action = (row, col)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input format. Please enter row and column as numbers separated by a space.")
            state, reward, done = env.step(action)
        else: # Agent's turn
             valid_actions = env.get_valid_actions()
             state_key = agent_x.get_state_key(env.get_state())
             # Agent plays greedily after training (epsilon = 0 for evaluation)
             q_values = {}
             for action in valid_actions:
                 action_key = (state_key, action)
                 q_values[action] = agent_x.q_table.get(action_key, 0.0)

             if q_values: # Ensure there are possible moves before finding max
                 max_q = -float('inf')
                 best_action = None
                 for action, q_value in q_values.items():
                     if q_value > max_q:
                         max_q = q_value
                         best_action = action
                 action = best_action if best_action is not None else random.choice(valid_actions) # Fallback
             else:
                 # This case should ideally not be reached if game status is playing but no valid moves
                 action = random.choice([(0,0)]) # Placeholder, should handle game end before this

             print(f"Agent plays: {action}")
             state, reward, done = env.step(action)

        if done:
            env.render()
            if env.status == "win" and env.current_player == agent_player:
                print("Agent wins!")
            elif env.status == "win" and env.current_player == human_player:
                 print("You win!")
            elif env.status == "draw":
                 print("It's a draw!")
            break