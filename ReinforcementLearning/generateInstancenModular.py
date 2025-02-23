import numpy as np
import json
from residual_graph_generator import*

def load_environment(filename):
        # Laden der Umgebungsinformationen aus einer JSON-Datei
        with open(filename, 'r') as file:
            environment_data = json.load(file)        
        return environment_data
#Generierung von Übergangnswahrscheinlichkeit   
#Übergangswahrscheinlichkeit: Dimemsion num_component x D(I) x I
def generate_transition_probabilities(num_component , state_size, action_size):
    # Zulässige Aktionen für jeden Zustand
    allowed_action = {}
    for component in range( num_component):
        allowed_action[component] = {}
        for state in range(state_size):
            if state == 0:
            # Zustand 0 hat die Aktionen 0 und action_size - 1
                 allowed_action[component][state] = np.array([action_size - 1])
            elif state == state_size -1:
                 allowed_action[component][state] = np.array([0, action_size - 1])
             # Andere Zustände haben alle Aktionen von 0 bis action_size - 1
            else :
                 allowed_action[component][state] = np.array(range(action_size))
    # Stelle sicher, dass mindestens ein Zustand von jedem anderen Zustand erreichbar ist
    transition_probabilities = {}
    for component in range( num_component):
        transition_probabilities[component] = {}
        for state in np.arange(state_size):
            transition_probabilities[component][state] = {}
            for action in allowed_action[component][state]:
                if state == 0:
                    probs = [0]*(state_size - 1) + [1]
                    transition_probabilities[component][state][action] = probs 
                else:
                    # Für andere Zustände Zufallsverteilung erstellen
                    probs = np.random.dirichlet(np.ones(state_size), size=1)[0]
                    # Normalisiere die Wahrscheinlichkeiten, damit die Summe 1 ergibt
                    probs /= np.sum(probs)
                    transition_probabilities[component][state][action] = probs.tolist()
    return allowed_action, transition_probabilities

def generate_environment(filename , component_size, num_parallel_paths,action_size, state_size):
        G =  generate_risidual_graph(component_size,num_parallel_paths)
        remove_edge_between_nodes_with_min_neighbors(G)
        # Action: Es gibt 'action_size' mögliche Aktionen
        action_space = np.arange(action_size)
        # State: Die Anzahl der Zustände ist state_size
        observation_space = np.arange(state_size)
        # Zulässige Aktionen für jeden Zustand und Übergangswahrscheinlichkeiten
        allowed_action , transition_probabilities = generate_transition_probabilities(component_size,state_size, action_size) 
        # Kostenfunktion
        action_costs = {
                        component:  {action: 0 if action == 0 else -1.0*action_space[action]
                             for action in range(action_size)}
                             for component in range(component_size)
                        }
        #Erneureung ist die Teuerste Aktion
        for component in range(component_size):
             action_costs[component][action_size-1] = -2.0*len(action_space)
        # Speichern der Umgebungsinformationen in einer JSON-Datei
        environment_data = {
            'action_space': action_space.tolist(),
            'observation_space': observation_space.tolist(),
            'allowed_action': {str(component): {str(state): actions.tolist() for state, actions in allowed_action[component].items()} for component in range(component_size)},
            'transition_probabilities': {str(component): {str(state): {str(action): probs for action, probs in actions.items()}
                                         for state, actions in transition_probabilities[component].items()}
                                         for component in range(component_size)},
            'action_costs':{str(component):{str(action): cost for action, cost in costs.items()} 
                             for component, costs in action_costs.items()},
            'edges' : list(G.edges()),
            'num_components': component_size
        }

        with open(filename, 'w') as file:
            json.dump(environment_data, file, indent=4)
state_size = 3
action_size = 3
component_size = 5
num_parallel_paths =2
#for i in range(1,5+1):
#  generate_environment(f"InstanceModularSystem/serien_system/instance{i}.txt",component_size,num_parallel_paths,action_size,state_size)   
   #generate_environment(f"instance{i}.txt",component_size,num_parallel_paths,action_size,state_size)
