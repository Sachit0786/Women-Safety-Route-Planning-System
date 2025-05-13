# Import required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import folium
import webbrowser
import osmnx as ox
import networkx as nx
from datetime import datetime
from generate_safety_data import NITJ_LOCATIONS

# Step 1: Load CSV Data
def load_data():
    """
    Load the safety data from a CSV file.
    """
    data = pd.read_csv("processed_safety_data.csv")
    return data

# Step 2: Normalize the Data
def normalize_data(data):
    """
    Normalize the safety parameters.
    """
    # Select only the safety-related columns for normalization
    safety_columns = ['crime_rate', 'accidents', 'police_stations', 'street_lights', 
                     'cameras', 'public_transport', 'shops', 'crowd_density', 
                     'construction_zones', 'residential_area']
    
    scaler = MinMaxScaler()
    data_normalized = data.copy()
    data_normalized[safety_columns] = scaler.fit_transform(data[safety_columns])
    return data_normalized

# Step 3: Calculate Safety Index
def calculate_safety_index(data_normalized, time_of_travel):
    """
    Calculate safety index based on the normalized data and time of travel.
    """
    # Define weights for different safety parameters
    weights = {
        'crime_rate': -0.2,  # Negative weight as higher crime rate means lower safety
        'accidents': -0.15,
        'police_stations': 0.15,
        'street_lights': 0.1,
        'cameras': 0.1,
        'public_transport': 0.1,
        'shops': 0.1,
        'crowd_density': 0.05,
        'construction_zones': -0.1,
        'residential_area': 0.05
    }
    
    # Adjust weights based on time of travel
    if time_of_travel.hour >= 18 or time_of_travel.hour <= 6:  # Night time
        weights['street_lights'] *= 1.5
        weights['cameras'] *= 1.5
        weights['crowd_density'] *= 0.5  # Less crowded might be safer at night
    
    # Calculate weighted safety index
    safety_index = np.zeros(len(data_normalized))
    for param, weight in weights.items():
        safety_index += data_normalized[param] * weight
    
    # Normalize safety index to [0, 1] range
    safety_index = (safety_index - safety_index.min()) / (safety_index.max() - safety_index.min())
    
    return safety_index

# Step 4: Fetch Road Network Data using OSMnx
def fetch_road_network(location_point, dist=1000):
    """
    Fetch road network data for a given location point.
    """
    G = ox.graph_from_point(location_point, dist=dist, network_type="drive")
    return G

# Step 5: Compute Safety Indices for Road Network Nodes
def compute_safety_indices(G, data_normalized, safety_index, time_of_travel):
    """
    Compute safety indices for all nodes in the road network.
    """
    node_safety_indices = {}
    for node in G.nodes:
        # Find the nearest data point to the node
        node_lat, node_lon = G.nodes[node]["y"], G.nodes[node]["x"]
        distances = np.sqrt((data_normalized["latitude"] - node_lat) ** 2 + 
                          (data_normalized["longitude"] - node_lon) ** 2)
        nearest_index = distances.idxmin()
        node_safety_indices[node] = safety_index[nearest_index]
    
    return node_safety_indices

def compute_safety_indices_for_path(G, data_normalized, safety_index, time_of_travel, path):
    """
    Compute safety indices for nodes in the path.
    """
    path_safety_indices = {}
    for node in path:
        node_lat, node_lon = G.nodes[node]["y"], G.nodes[node]["x"]
        distances = np.sqrt((data_normalized["latitude"] - node_lat) ** 2 + 
                          (data_normalized["longitude"] - node_lon) ** 2)
        nearest_index = distances.idxmin()
        path_safety_indices[node] = safety_index[nearest_index]
    
    return path_safety_indices

def normalize_safety_indices(safety_indices):
    """
    Normalize the safety indices to ensure values are between 0 and 1.
    """
    values = np.array(list(safety_indices.values()))
    normalized = (values - values.min()) / (values.max() - values.min())
    return dict(zip(safety_indices.keys(), normalized))

# Step 6: Find Safest Path using Floyd-Warshall Algorithm
def find_safest_path(G, safety_indices, start_node, end_node):
    """
    Find the safest path using the Floyd-Warshall algorithm.
    """
    num_nodes = len(G.nodes)
    node_list = list(G.nodes)
    safety_matrix = np.full((num_nodes, num_nodes), np.inf)
    next_matrix = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(safety_matrix, 0)

    # Initialize safety matrix and next matrix
    for u, v, data in G.edges(data=True):
        u_idx = node_list.index(u)
        v_idx = node_list.index(v)
        # Use inverse of average safety index as edge weight
        safety_matrix[u_idx][v_idx] = 1 - (safety_indices[u] + safety_indices[v]) / 2
        next_matrix[u_idx][v_idx] = v_idx

    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if safety_matrix[i][j] > safety_matrix[i][k] + safety_matrix[k][j]:
                    safety_matrix[i][j] = safety_matrix[i][k] + safety_matrix[k][j]
                    next_matrix[i][j] = next_matrix[i][k]

    # Reconstruct the safest path
    start_idx = node_list.index(start_node)
    end_idx = node_list.index(end_node)

    if next_matrix[start_idx][end_idx] == -1:
        raise ValueError(f"No path exists from {start_node} to {end_node}.")

    path = []
    current = start_idx
    while current != end_idx:
        path.append(node_list[current])
        current = next_matrix[current][end_idx]
    path.append(node_list[end_idx])

    return path

# Step 7: Find Shortest Path using Dijkstra's Algorithm
def find_shortest_path(G, start_node, end_node):
    """
    Find the shortest path using Dijkstra's algorithm.
    """
    shortest_path = nx.shortest_path(G, start_node, end_node, weight="length")
    return shortest_path

# Step 8: Calculate Path Length
def calculate_path_length(G, path):
    """
    Calculate the total length of a path in kilometers.
    """
    total_length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                for key, data in edge_data.items():
                    total_length += data.get("length", 0)
            else:
                raise ValueError(f"No edge found between {u} and {v}.")
        else:
            raise ValueError(f"No edge found between {u} and {v}.")
    return total_length / 1000  # Convert meters to kilometers

# Step 9: Visualize Paths on Map
def display_paths_on_map(G, safest_path, shortest_path, safety_indices):
    """
    Display the safest and shortest paths on a Folium map.
    """
    # Create a map centered at the start node
    start_lat, start_lon = G.nodes[safest_path[0]]["y"], G.nodes[safest_path[0]]["x"]
    map = folium.Map(location=[start_lat, start_lon], zoom_start=16)

    # Add the safest path
    safest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in safest_path]
    folium.PolyLine(safest_coords, color="green", weight=5, opacity=1, popup="Safest Path").add_to(map)

    # Add the shortest path
    shortest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in shortest_path]
    folium.PolyLine(shortest_coords, color="blue", weight=5, opacity=1, popup="Shortest Path").add_to(map)

    # Add markers for start and end points
    folium.Marker(safest_coords[0], popup=f"Start | Safety Index: {safety_indices[safest_path[0]]:.2f}").add_to(map)
    folium.Marker(safest_coords[-1], popup=f"End | Safety Index: {safety_indices[safest_path[-1]]:.2f}").add_to(map)

    # Save and open the map
    map.save("safest_and_shortest_paths.html")
    webbrowser.open("safest_and_shortest_paths.html")

# Step 10: Main Function
def main():
    # Load and normalize the data
    data = load_data()
    data_normalized = normalize_data(data)

    # Calculate safety indices
    time_of_travel = datetime.now()  # Use current time or allow user input
    safety_index = calculate_safety_index(data_normalized, time_of_travel)

    # Fetch road network data for NITJ campus
    location_point = (31.39611324675054, 75.53603682537147)  # Center of NITJ campus
    G = fetch_road_network(location_point, dist=2000)  # Increased radius to cover campus

    # Get user input for start and end locations
    print("\nAvailable locations:")
    for i, (name, (lat, lon)) in enumerate(NITJ_LOCATIONS.items(), 1):
        print(f"{i}. {name}")
    
    start_idx = int(input("\nEnter the number for start location: ")) - 1
    end_idx = int(input("Enter the number for end location: ")) - 1
    
    start_name = list(NITJ_LOCATIONS.keys())[start_idx]
    end_name = list(NITJ_LOCATIONS.keys())[end_idx]
    
    start_lat, start_lon = NITJ_LOCATIONS[start_name]
    end_lat, end_lon = NITJ_LOCATIONS[end_name]

    # Compute safety indices for road network nodes
    safety_indices = compute_safety_indices(G, data_normalized, safety_index, time_of_travel)

    # Find the nearest nodes to the user's input
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    # Find the safest and shortest paths
    safest_path = find_safest_path(G, safety_indices, start_node, end_node)
    shortest_path = find_shortest_path(G, start_node, end_node)

    # Calculate path lengths
    safest_path_length = calculate_path_length(G, safest_path)
    shortest_path_length = calculate_path_length(G, shortest_path)

    # Calculate safety indices for paths
    safety_indices_safest = compute_safety_indices_for_path(G, data_normalized, safety_index, time_of_travel, safest_path)
    safety_indices_shortest = compute_safety_indices_for_path(G, data_normalized, safety_index, time_of_travel, shortest_path)

    # Normalize the safety indices
    normalized_safety_indices_safest = normalize_safety_indices(safety_indices_safest)
    normalized_safety_indices_shortest = normalize_safety_indices(safety_indices_shortest)

    mean_safety_index = sum(normalized_safety_indices_safest.values()) / len(normalized_safety_indices_safest)
    mean_shortest_index = sum(normalized_safety_indices_shortest.values()) / len(normalized_safety_indices_shortest)

    # Display the path lengths and safety indices
    print(f"\nRoute from {start_name} to {end_name}:")
    print(f"Safest Path Length: {safest_path_length:.2f} km")
    print(f"Shortest Path Length: {shortest_path_length:.2f} km")
    print(f"Safest Path Safety Index: {mean_safety_index:.2f}")
    print(f"Shortest Path Safety Index: {mean_shortest_index:.2f}")

    # Display the paths on the map
    display_paths_on_map(G, safest_path, shortest_path, safety_indices)

# Run the program
if __name__ == "__main__":
    main()