import gym
import numpy as np
import json
from gym import spaces

class MDPEnvironment(gym.Env):
    def __init__(self, state_size=15, action_size=8, load_from=None):
        super(MDPEnvironment, self).__init__()

        if load_from:
            self.load_environment(load_from)
        else:
            self.state_size = state_size
            self.action_size = action_size
            
            # Zustands- und Aktionsraum definieren
            self.observation_space = spaces.Discrete(state_size)
            self.action_space = spaces.Discrete(action_size)

            # Generiere zufällige Übergangswahrscheinlichkeiten und Kosten
            self.allowed_action, self.transition_probabilities = self.generate_transition_probabilities()
            self.action_costs = {state: {action: np.random.uniform(-3, 0) for action in self.allowed_action[state]}
                                 for state in range(state_size)}

            # Startzustand
            self.state = np.random.choice(state_size)
            self.max_steps = state_size  # Episodenlänge
            self.current_step = 0

    def generate_transition_probabilities(self):
        """Erstellt zufällige Übergangswahrscheinlichkeiten für das MDP."""
        allowed_action = {state: np.random.choice(np.arange(self.action_size - 1), 
                                                  size=np.random.randint(1, self.action_size), 
                                                  replace=False)
                          for state in range(self.state_size)}
        
        # Der letzte Zustand erlaubt nur eine Aktion (Erneuerung)
        allowed_action[self.state_size - 1] = np.array([self.action_size - 1])

        transition_probabilities = {}
        for state in range(self.state_size):
            transition_probabilities[state] = {}
            for action in allowed_action[state]:
                if state == self.state_size - 1:
                    probs = np.zeros(self.state_size)
                    probs[0] = 1  # Geht immer zurück zum ersten Zustand
                else:
                    probs = np.random.dirichlet(np.ones(self.state_size))
                
                transition_probabilities[state][action] = probs.tolist()
        
        return allowed_action, transition_probabilities

    def step(self, action):
        """Führt eine Aktion aus und gibt (next_state, reward, done, info) zurück."""
        if action not in self.allowed_action[self.state]:
            return self.state, -10, False, {"message": "Ungültige Aktion"}

        # Ziehe den nächsten Zustand basierend auf den Übergangswahrscheinlichkeiten
        next_state = np.random.choice(self.state_size, p=self.transition_probabilities[self.state][action])
        
        # Belohnung aus der Kostenfunktion
        reward = self.action_costs[self.state][action]
        
        self.state = next_state
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Episode beendet?

        return self.state, reward, done, {}

    def reset(self):
        """Setzt die Umgebung zurück und gibt den Startzustand zurück."""
        self.state = np.random.choice(self.state_size)
        self.current_step = 0
        return self.state

    def render(self, mode='human'):
        """Visualisierung der aktuellen Umgebung."""
        print(f"Schritt: {self.current_step}, Zustand: {self.state}")

    def save_environment(self, filename):
        """Speichert die Umgebungsinformationen in einer JSON-Datei."""
        environment_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'allowed_action': {str(state): actions.tolist() for state, actions in self.allowed_action.items()},
            'transition_probabilities': {str(state): {str(action): probs for action, probs in actions.items()}
                                         for state, actions in self.transition_probabilities.items()},
            'action_costs': {str(state): {str(action): cost for action, cost in costs.items()}
                             for state, costs in self.action_costs.items()},
            'max_step': self.max_steps
        }

        with open(filename, 'w') as file:
            json.dump(environment_data, file, indent=4)

    def load_environment(self, filename):
        """Lädt die Umgebungsinformationen aus einer JSON-Datei."""
        with open(filename, 'r') as file:
            environment_data = json.load(file)

        self.state_size = environment_data['state_size']
        self.action_size = environment_data['action_size']
        self.max_steps = environment_data['max_step']
        self.observation_space = spaces.Discrete(self.state_size)
        self.action_space = spaces.Discrete(self.action_size)

        self.allowed_action = {int(state): np.array(actions) for state, actions in environment_data['allowed_action'].items()}
        self.transition_probabilities = {int(state): {int(action): np.array(probs) for action, probs in actions.items()}
                                         for state, actions in environment_data['transition_probabilities'].items()}
        self.action_costs = {int(state): {int(action): cost for action, cost in costs.items()}
                             for state, costs in environment_data['action_costs'].items()}

        self.state = np.random.choice(self.state_size)
        self.current_step = 0
        print(f"✅ Umgebung aus '{filename}' geladen!")
