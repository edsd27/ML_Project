import numpy as np
import json

def load_environment(filename):
        """
    Load environment information from a JSON file.

    Args:
    - filename (str): The name of the JSON file containing environment data.

    Returns:
    - environment_data (dict): Dictionary containing environment information.
    """
        with open(filename, 'r') as file:
            environment_data = json.load(file)        
        return environment_data
 
def generate_transition_probabilities(state_size, action_size):
    """
    Generate transition probabilities for a given state and action size.

    Args:
    - state_size (int): Number of states in the environment.
    - action_size (int): Number of possible actions.

    Returns:
    - allowed_action (dict): Dictionary mapping each state to its allowed actions.
    - transition_probabilities (dict): Dictionary representing transition probabilities for each state and action.
"""
    # Define allowed actions for each state
    allowed_action = {}
    for state in range(state_size):
        if state == state_size - 1:
           # Only allow "Renewal" for the worst state
            allowed_action[state] = np.array([action_size - 1])
        else:
            # Randomly choose actions for other states
            allowed_action[state] = np.random.choice(np.arange(action_size - 1), size=np.random.randint(1, action_size), replace=False)

    # Create transition probabilities: Ensure positive recurrence and irreducibility
    transition_probabilities = {}
    for state in np.arange(state_size):
        transition_probabilities[state] = {}
        for action in allowed_action[state]:
            if state == state_size - 1:
                # Allow only "Renewal" in the last state
                probs = [1] + [0] * (state_size - 1)
                transition_probabilities[state][action] = probs
            else:
                # Create a random distribution for other states
                probs = np.random.dirichlet(np.ones(state_size), size=1)[0]
                # Normalize probabilities to sum to 1
                probs /= np.sum(probs)

                transition_probabilities[state][action] = probs.tolist()

    return allowed_action, transition_probabilities
def generate_environment(filename , action_size, state_size):
        """
    Generate environment data and save it to a JSON file.

    Args:
    - filename (str): The name of the JSON file to store environment data.
    - action_size (int): Number of possible actions.
    - state_size (int): Number of states in the environment.
    """
        # Define action and state spaces
        action_space = np.arange(action_size)
        observation_space = np.arange(state_size)
        # Generate allowed actions and transition probabilities
        allowed_action , transition_probabilities = generate_transition_probabilities(state_size, action_size) 
        system_cost = np.arange(len(observation_space)) * 0.25
         # Initialize Q-table
        action_costs = {state: {action: -1*(system_cost[state] +  action_space[action]) for action in allowed_action[state]}
                        for state in observation_space}       
        # Q-Tabelle initialisieren
        q_table = np.zeros((state_size, action_size))
        # Set the length of an episode
        max_steps = state_size
      
        # Save environment information to a JSON file
        environment_data = {
            'action_space': action_space.tolist(),
            'observation_space': observation_space.tolist(),
            'allowed_action': {str(state): actions.tolist() for state, actions in allowed_action.items()},
            'transition_probabilities': {str(state): {str(action): probs for action, probs in actions.items()}
                                         for state, actions in transition_probabilities.items()},
            'action_costs': {str(state): {str(action): cost for action, cost in costs.items()}
                             for state, costs in action_costs.items()},
            'q_table': q_table.tolist(),
            'max_step': max_steps
        }

        with open(filename, 'w') as file:
            json.dump(environment_data, file, indent=4)

state_size = 10
action_size = 8
#for i in range(1,10+1):
#      generate_environment(f"Instanz10_8/instanz{i}.txt",action_size,state_size)
