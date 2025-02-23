import numpy as np
import random
import time
from generateInstance import load_environment

# Dynamic Epsilon-Greedy Strategy Epsilon
class EGreedyPolicy:
    def __init__(self):
        # Counters for exploration and exploitation
        self.exploration = 0
        self.exploitation = 0
        # Initial epsilon value
        self.epsilon = 0.95
    def compute_action(self, q_table, state, allowed_actions):
        # With probability epsilon, explore; otherwise, exploit
        if random.randint(0, 1) < self.epsilon:
            # Exploration: choose a random action
            action = random.choice(allowed_actions[state])
            self.exploration += 1
        else:
            # Exploitation: choose the action with the maximum Q-value
            q_max = max(q_table[state, new_action] for new_action in allowed_actions[state])
            best_actions = [action for action in allowed_actions[state] if q_table[state, action] == q_max]
            action = np.random.choice(best_actions)
            self.exploitation += 1
        return action

class WartungsEnv():
    def __init__(self, environment_data={}):
        # Initialize the state variables (current state)
        self.current_step = 0
        self.state = 0
        # Hyper-Parameters
        self.learning_rate = 1
        # Set parameter values
        self.action_space = np.array(environment_data['action_space'])
        self.observation_space = np.array(environment_data['observation_space'])
        self.allowed_action = {int(state): np.array(actions) for state, actions in environment_data['allowed_action'].items()}
        self.transition_probabilities = {int(state): {int(action): probs for action, probs in actions.items()}
                                         for state, actions in environment_data['transition_probabilities'].items()}
        self.action_costs = {int(state): {int(action): cost for action, cost in costs.items()}
                             for state, costs in environment_data['action_costs'].items()}
        self.q_table = np.array(environment_data['q_table'])
        self.max_steps = environment_data['max_step'] ** 2
        # Choose a policy
        self.policy = EGreedyPolicy()
        # Choose a reference state: randomly from all states
        self.state_0 = np.random.randint(0, len(self.observation_space) - 1)

    def get_action(self, state):
        # Get the action based on the policy
        action = self.policy.compute_action(self.q_table, state, self.allowed_action)
        return action

    def reset(self):
        # Reset the environment to the initial state
        self.current_step = 0
        self.state = 0
        return self.state

    def learn(self, state, action, reward, new_state, done):
        if done:
            # Terminal state, max(Q(s',a')) does not exist
            max_q = 0
            max_q_0 = 0
        else:
            # max(Q(s',a'))
            max_q = max(self.q_table[new_state, new_action] for new_action in self.allowed_action[new_state])
            # max(Q(s_0,a_0))
            max_q_0 = max(self.q_table[self.state_0, new_action] for new_action in self.allowed_action[self.state_0])
        # Q(s,a)
        actual_q_value = self.q_table[state, action]
        # (1-alpha)*Q(s,a) + alpha * [R + max(Q(s',a') - max(Q(s_0,a_0)))]
        q_value = (1 - self.learning_rate) * actual_q_value + self.learning_rate * (reward + max_q - max_q_0)
        # Update Q-value in the Q-table
        self.q_table[state, action] = q_value

    def step(self, action):
        # Execute chosen action
        cost = self.action_costs[self.state][action]
        # Move one step forward
        self.current_step += 1
        # Check if the maximum number of steps has been reached
        done = self.current_step >= self.max_steps
        # Determine the next state
        next_state_probs = self.transition_probabilities[self.state][action]
        next_state = np.random.choice(len(self.observation_space), p=next_state_probs)
        # Calculate the reward based on the current state and action
        reward = cost
        # Update Q-value
        self.learn(self.state, action, reward, next_state, done)
        # Update the state
        self.state = next_state
        return self.state, reward, done, {}

# Training process
time_pr = []
C_alpha = 1  # Startwert für Lernrate
t_0 = 10  # Offset zur Stabilisierung
p = 0.8  # Steuert die Abnahmegeschwindigkeit

C_epsilon = 0.95  # Startwert für epsilon
lambda_epsilon = 0.005  # Steuert, wie schnell epsilon abnimmt

for count in range(1, 11):
    environment_data = load_environment(f"Instanz5_4/instanz{count}.txt")
    env = WartungsEnv(environment_data=environment_data)

    episodes = len(env.observation_space) * len(env.action_space) * 100
    start_time = time.process_time()
    # Langfristig bleibt die Lernrate groß genug, um weiter zu lernen.
    # Aber die Lernrate wird klein genug, sodass das Lernen stabilisiert wird.

    for episode in range(1, episodes + 1):
        state = env.reset()
        score = 0
        done = False
        step = 0  # Schrittzähler in der Episode
        while not done:
            # Dynamische Lernrate pro Schritt in der Episode
            env.learning_rate = C_alpha / ((step + t_0) ** p)

            # Dynamische Epsilon-Anpassung pro Schritt
            env.policy.epsilon = C_epsilon * np.exp(-lambda_epsilon * step)

            action = env.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            step += 1  # Schrittzähler erhöhen

    end_time = time.process_time() - start_time
    time_pr.append(end_time)

    # Entries with 0 in the q-table must be replaced by the minimum negative value,
    # because otherwise, they might be considered as maximum q-values.
    for i in range(len(env.q_table)):
        for j in range(len(env.q_table[0])):
            if env.q_table[i][j] == 0 and j not in env.allowed_action[i]:
                env.q_table[i][j] = float("-inf")
    # Derive the policy from the Q-table; policy now contains the preferred action for each state
    policy = np.argmax(env.q_table, axis=1).tolist()
    print(f"total number of exploration {env.policy.exploration}")
    print(f"total number of exploitation {env.policy.exploitation}\n")
    print(f"CPU Zeit {end_time}")
    # Output the optimal stationary strategy
    file_path = f"ErgebnisQlearning/ergebnis{count}.txt"
    with open(file_path, 'w') as file:
        for state, action in enumerate(policy, start=0):
            file.write(f"sigma*({state})={action}\n")
print(f"Durschnittliche CPU-Zeit: {sum(time_pr)/10}")