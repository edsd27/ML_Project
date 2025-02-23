import networkx as nx
import random

#Entferne Brueckenkante im Resisualgraph
def remove_edge_between_nodes_with_min_neighbors(graph,  min_neighbors =3):
    # Find nodes with at least min_neighbors neighbors excluding the first and last nodes
    eligible_nodes = [node for node, degree in graph.degree() if degree >= min_neighbors and node not in [0, graph.number_of_nodes()-1]]
    removed_edges = []
    saved_brige = []
    # Check if there is an edge between eligible nodes
    for node1 in eligible_nodes:
        for node2 in eligible_nodes:
            if node1 != node2 and graph.has_edge(node1, node2):
                # Ensure that neither node1 nor node2 is the start or end node of the edge
                if node1 not in (graph.edges[node1,node2,0], graph.edges[node2,node1,0]) and node1 not in removed_edges and node2 not in removed_edges:
                    # Remove the edge between eligible nodes
                    removed_edges.append(node1)
                    removed_edges.append(node2)
                    saved_brige.append((node1,node2))
                    graph.remove_edge(node1, node2)
def update_end_nodes(graph , end):
    start_nodes = set()
    for edge in graph.edges:
        start_nodes.add(edge[0])
    e =  list(graph.edges()).copy()
    for edge in e:
        end_node = edge[1]
        if end_node not in start_nodes:
           graph.remove_edge(edge[0],edge[1])
           graph.add_edge(edge[0],end)

def generate_risidual_graph(num_edges, num_parallel_paths):
    if num_edges < num_parallel_paths:
        print("Anzahl der Kanten muss größer als die Anzahl der Parallel Pfaden")
        return 
    if num_parallel_paths ==1 :
        edges=[(0,1)]
        for i in range(1 , num_edges):
            edges.append((i ,i+1))
        G = nx.MultiGraph()
        G.add_edges_from(edges)
    else :
        edges = [(0, i) for i in range(1, num_parallel_paths + 1)]
        remaining_edges = num_edges - num_parallel_paths
        end_nodes_in_parallel_paths = set(range(1, num_parallel_paths + 1))
        start_nodes_in_parallel_paths =list(end_nodes_in_parallel_paths)[:]
        i=0
        for count in range(remaining_edges):
                if i == num_parallel_paths -1 and num_parallel_paths !=1:
                    i = 0
                start_node = list(start_nodes_in_parallel_paths)[i]
                end_node = list(end_nodes_in_parallel_paths)[-1]
                if count< remaining_edges -1:
                         end_node += 1
                edges.append((start_node, end_node))
                end_nodes_in_parallel_paths.add(end_node)
                end_nodes_in_parallel_paths.remove(start_node)
                start_nodes_in_parallel_paths.remove(start_node)
                start_nodes_in_parallel_paths.append(end_node) 
                i+=1
        G = nx.MultiGraph()
        G.add_edges_from(edges)
        max_end_node = max(node for node in list(end_nodes_in_parallel_paths))
        update_end_nodes(G , max_end_node)
    return G


