from pulp import *
from math import *
import numpy as np
import time
from generateInstanzen import load_environment

for count in range(1, 20+1):
    # Erstelle eine Instanz für ein Maximierungsproblem
    prob = LpProblem(sense=LpMinimize)
    d_max = LpVariable("d_max", lowBound=0)
    #Entscheidungsvariablen (x,y) : koordinat eines standorts
    environment_data = load_environment(f"Instanz11_4/instanz{count}.txt")
    #environment_data = load_environment(f"QSIBeispiel.txt")
    A = np.array(environment_data['action_space'])
    I = np.array(environment_data['observation_space'])
    x =[[LpVariable(f"x_{i}_{j}",lowBound=0.0) for j in range(len(A))] for i in range(len(I))]
    #abbbilduung von eine Zustand i auf  Aktionen a  
    D =  {int(state): np.array(actions) for state, actions in environment_data['allowed_action'].items()}
    #Übergangswahrscheinlichkeit: Dimemsion D(I) x I
    # Bestimme die Anzahl der Zustände und Aktionen
    num_states = len(I)
    num_actions = len(A) 
    # Initialisiere das q-Dictionary
    transition_probabilities = {int(state): {int(action): probs for action, probs in actions.items()}
                                            for state, actions in environment_data['transition_probabilities'].items()}
    q = {int(state): [[0.0] * num_states for _ in range(num_actions)] for state in transition_probabilities}
    # Fülle das q-Dictionary mit den Werten aus transition_probabilities_data
    for state, actions in transition_probabilities.items():
        for action, probabilities in actions.items():
            q[int(state)][int(action)] = probabilities
    # Belohnungsfunktion : Dimension I x A  und r_sa = -(c_s + c_a)
    # Erstelle eine Liste der Kostenwerte
    action_costs_data = {int(state): {int(action): cost for action, cost in costs.items()}
                                for state, costs in environment_data['action_costs'].items()}
    cost_values = []
    for state, actions in action_costs_data.items():
        for action, cost in actions.items():
            cost_values.append(cost)
    # Erstelle ein NumPy-Array und fülle es mit den Kostenwerten
    action_costs_matrix = np.zeros((num_states, num_actions))
    for state in I:
        for action in A :
            if action in action_costs_data[state]:
                action_costs_matrix[state, action] = -action_costs_data [state][action]
            else :
                action_costs_matrix[state, action] =10000000000
    c = action_costs_matrix
    # Definiere die Zielfunktion
    #for s in I:
    prob+= lpSum([c[s][a]*x[s][a] for a in A for s in I])
    for s in I :
        prob+= lpSum( x[s][a] for a in D[s]) == lpSum([q[sprime][a][s] * x[sprime][a] for a in A for sprime in I])
    prob+=lpSum([x[s][a] for a in A for s in I]) == 1.0
    # Löse die Probleminstanz mit CBCCbc
    start_time=time.time()
    prob.solve(solver=getSolver("PULP_CBC_CMD"))
    end_time = time.time()-start_time
    #print(f"Dauer der Simulation: {end_time}")
    #print("Statut:",LpStatus[prob.status])
    # Gib die Werte aller Variablen aus
    #aktion= ["k","w","e"]
    #for s in I:
    #    for a in A:
    #     if (x[s][a].varValue>0):
    #        print(f"Warhscheinlichkeit x_{s}{a} =",x[s][a].varValue)
    #        print(f"sigma({s})={a}")
    file_path = f"ErgebenisLinearOptimierung/ergebnis{count}.txt"
        # Schreiben Sie die Ergebnisse in die Textdatei
    with open(file_path, 'w') as file:
            file.write("Solver: Linear optimization \n")
            file.write(f"Statut: {LpStatus[prob.status]}\n")
            file.write(f"Dauer der Simulation :{end_time}\n")
            for s in I:
                for a in A:
                    if (x[s][a].varValue>0):
                        file.write(f"sigma*({s})={a}\n")
                        file.write(f"Warhscheinlichkeit x_{s}{a}= {x[s][a].varValue}\n")