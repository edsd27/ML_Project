import networkx as nx
import matplotlib.pyplot as plt
import random
from math import *
import numpy as np
from itertools import product

#Function to draw a graph with labels
def draw_graph_with_labels(graph):
    # Layout for graph visualization
    pos = nx.spring_layout(graph)
    # Edge labels with weights
    edge_labels = {(u, v): d.get(0, {}).get('weight', '') for u, v, d in graph.edges(data=True)}
    # Draw the graph using NetworkX
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', font_size=8,
            font_color='black', edge_color='gray', width=2) 
    # Draw edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    # Show the graph
    plt.show()

# Function to determine allowed actions based on component states
def D(allowed_actions,number_to_state, action_to_number , C):
    allowed_action_combination = {}
    for state in number_to_state:
        D_s = []
        state_combination= number_to_state[state]
        # Iterate over components
        for i in C:
            D_s.append(allowed_actions[i][state_combination[i]])
        # Generate all possible action combinations
        actions = list(product(*D_s))
        result = []
        for a in actions:
            result.append(action_to_number[a])
        allowed_action_combination[state] = result
    return allowed_action_combination
# Function to calculate the total system probability
def pro_sys(q,sprime , number_to_state , number_to_action ,a ,s ,C):
    p=1
    for i in C:
        p*=q[i][number_to_state[sprime][i]][number_to_action[a][i]][number_to_state[s][i]]
    return p

# Function to calculate costs for each component, state, and action com
def calculate_cost_system(edges,num_components,allowed_actions, action_to_number,state_to_number, action_costs,c):
    num_state_combinations = len(state_to_number)
    num_action_combinations = len(action_to_number)
    c = np.zeros((num_state_combinations+1, num_action_combinations+1))
    for state_combination in state_to_number:
            state_number = state_to_number[state_combination]
            state =0
            G = nx.MultiGraph()
            G.add_edges_from(edges)
            # Assign component states to edge weights
            for e in G.edges():
                    G[e[0]][e[1]][0]['weight']= state_combination[state]
                    state +=1
            start_edge = list(G.edges())[0]
            # Calculate cost of the structural function
            system_cost =  -5.0*(np.exp(-reduction_method(G,start_edge)/10.0))
            for action_combination in action_to_number:
                action_number = action_to_number[action_combination]
                # Check if the action is allowed for the current state
                if action_number in allowed_actions[state_number]:
                    total_action_cost = sum(action_costs[i][action_combination[i]] for i in range(num_components))
                    total_cost = system_cost + total_action_cost
                else:
                    total_cost = -100000000 # A high value to represent infeasible actions
               
                c[state_number, action_number] = total_cost
    return c

# Function to check if two given edges are in parallel
def are_edges_in_parallel(edge1, edge2):
    common_start = edge1[0] == edge2[0] 
    common_end = edge1[1] == edge2[1]
    return  (edge1[1] == edge2[0] and (edge1[1]>= edge2[1])) \
            or common_start or (edge1[0] == edge2[1] and (edge1[0]<= edge2[1])) \
            or (common_end and (edge2[0]<= edge2[1])) or (edge1[0] == edge2[0] and (edge1[1]== edge2[1]))
# Function to get neighbor edges of an edge in the graph
def get_neighbor_edges(graph, edge, typ_connection_parallel):
    node1, node2 = edge
    neighbor_edges = []
    # Find edges that share a node with the current edge
    for e in graph.edges():
        if (e != edge and (node1 in e or node2 in e)) or len([i for i in  graph.edges() if i == edge])>=2:
            neighbor_edges.append(e)
    # Filter parallel or in-series edges
    neighbor_edges_parallel = [e for e in  neighbor_edges if are_edges_in_parallel(edge, e)]
    if  neighbor_edges_parallel:
        # If parallel edges exist, choose one randomly
        typ_connection_parallel = True
        return random.choice( neighbor_edges_parallel),typ_connection_parallel
    else:
        typ_connection_parallel = False
        # If no parallel edges, choose one randomly
        return random.choice(neighbor_edges) if neighbor_edges else None ,typ_connection_parallel
# Reduction method for determining the quality of the entire system
def reduction_method(graph, current_edge):
    # If only one edge is present, return the weight of that edge
    if graph.number_of_edges() == 1:
        value = graph[current_edge[0]][current_edge[1]].get(0, {}).get('weight', 0)
        graph.remove_edge(*current_edge)
        return value
    typ_connection_parallel = False
    # Choose a neighbor edge of current_edge
    neighbor_edge,typ_connection_parallel = get_neighbor_edges(graph, current_edge,typ_connection_parallel)
    if neighbor_edge == None :
         edges_without_current_edge =[e for e in graph.edges() if e!= current_edge ]
         neighbor_edge =edges_without_current_edge[0]
         typ_connection_parallel = True
    # Remove the edge and call the function recursively
    part_graph = graph.copy()
    part_graph.remove_edge(*current_edge)
    # Check the relationship between the edges
    if typ_connection_parallel:
            return max(graph[current_edge[0]][current_edge[1]].get(0, {}).get('weight', 0),reduction_method(part_graph, neighbor_edge))
    else :
            return min(graph[current_edge[0]][current_edge[1]].get(0, {}).get('weight', 0),reduction_method(part_graph, neighbor_edge))

#Test Method Reduktionsverfahren
#G = nx.MultiGraph()
#G.add_edges_from([(0,1),(0,2),(1,5),(2,5),(0,3),(3,4),(4,5)])
#G[0][1][0]['weight']= 1
#G[0][2][0]['weight']= 1
#G[0][3][0]['weight']= 2
#G[1][5][0]['weight']= 1
#G[2][5][0]['weight']= 4
#G[3][4][0]['weight']= 2
#G[4][5][0]['weight']= 0
#print(reduction_method(G,(0,1)))
#print(50*(2.71828**(-5*reduction_method(G,(0,1)))))