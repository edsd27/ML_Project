from pulp import *
import time
from generateInstancenModular import load_environment
from utility import *
from residual_graph_generator  import*

for count in range(1,5+1):
  # Load environment data from file
  environment_data = load_environment(f'InstanceModularSystem/serien_system/instance{count}.txt')
  # Extract relevant information from environment_data
  num_components =  environment_data['num_components']
  edges = environment_data['edges']
  A = np.array(environment_data['action_space'])
  I = np.array(environment_data['observation_space'])
  # Extract allowed actions and transition probabilities
  num_states = len(I)
  num_actions = len(A)
  allowed_actions = {int(component): {int(state): [int(val) for val in value] 
                                      for state, value in inner_dict.items()}
                                      for component, inner_dict in environment_data['allowed_action'].items()}                          
  transition_probabilities = {int(component): {int(state): {int(action): probs for action, probs in actions.items()}
                                              for state, actions in inner_dict.items()}
                                              for component, inner_dict in environment_data['transition_probabilities'].items()}
  # Initialize q-values based on transition_probabilities
  q = {i: {int(state): [[0.0] * num_states for _ in range(num_actions)] for state in transition_probabilities[i]} for i in range(num_components) }
  for i in range(num_components):
    for state, actions in transition_probabilities[i].items():
      for action, probabilities in actions.items():
          q[i][int(state)][int(action)] = probabilities
  # Mapping of state and action combinations to numbers
  states = range(num_states)
  actions = range(num_actions)
  state_to_number = {state_combination: i for i, state_combination in enumerate(product(states, repeat=num_components))}
  action_to_number = {action_combination: i for i, action_combination in enumerate(product(actions, repeat=num_components))}
  number_to_state = {i : state_combination  for state_combination ,i  in state_to_number.items()}
  number_to_action = {i : action_combination for  action_combination , i in action_to_number.items()}
  # Define action costs
  action_costs = {int(component): {int(action): float(cost) for action, cost in costs.items()} 
                      for component, costs in environment_data['action_costs'].items()}
  #allowed action combination
  allowed_action_combination = D(allowed_actions,number_to_state, action_to_number ,range(num_components)) 
  # Calculate total costs for each state-action combination
  num_state_combinations = len(state_to_number)
  num_action_combinations = len(action_to_number)
  c = np.zeros((num_state_combinations, num_action_combinations))
  c = calculate_cost_system(edges,num_components,allowed_action_combination,action_to_number,state_to_number,action_costs,c)
  # Linear Programming formulation using Pulp
  prob = LpProblem(sense=LpMaximize)
  # Decision variables
  x =[[LpVariable(f"x_{j}_{k}",lowBound=0.0) for k in number_to_action] for j in number_to_state]
  # Define the objective function
  prob+= lpSum([[c[s,a]*x[s][a] for a in  allowed_action_combination[s]] for s in number_to_state])
  # Define constraints
  for s in number_to_state :
    prob+= lpSum( x[s][a] for a in allowed_action_combination[s]) == lpSum([pro_sys(q,sprime,number_to_state,number_to_action,a,s,range(num_components))
                                                        *x[sprime][a] for a in number_to_action for sprime in number_to_state]) 

  prob+=lpSum([[x[s][a] for a in  allowed_action_combination[s]] for s in number_to_state]) == 1.0
  
  # Solve the LP problem with CBC solver
  prob.solve(solver=getSolver("PULP_CBC_CMD"))
  print("Statut:",LpStatus[prob.status])
  file_path = f"LPErgebnisModularSystem/serien/ergebnis{count}.txt"
  #Save results to a text file
  with open(file_path, 'w') as file:
      for s in number_to_state:
          for a in allowed_action_combination[s]:
              if (x[s][a].varValue>0):
                file.write(f"sigma*({number_to_state[s]})={number_to_action[a]}\n")