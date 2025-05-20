import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import folium
import webbrowser
import osmnx as ox
import networkx as nx
import webbrowser
from datetime import datetime

from models.safety_model import model_details

# Feature importance weights
FEATURE_WEIGHTS = {
    "cameras": 0.90,
    "street_lights": 0.75,
    "police_stations": 0.80,
    "public_transport": 0.65,
    "crime_rate": -0.85,
    "accidents": -0.60,
    "shops": 0.55,
    "construction_zones": -0.40,
    "crowd_density": -0.60,
    "parks_recreation": 0.10,
    "population_density": 0.30,
    "traffic_density": 0.25,
    "market_areas": 0.40,
    "time_of_day": -0.35,
    "sidewalk_presence": 0.50,
    "is_festival": 0.20,
    "hospitality_venues": 0.10,
    "emergency_services": 0.85,
    "is_holiday": -0.25,
    "is_night": -0.85,
    "day_of_week": 0.15,
    "religious_places": 0.10,
    "educational_institutions": 0.20
}

# Step 1: Load CSV Data
def load_data():
    """
    Load the safety data from a CSV file.
    """
    data = pd.read_csv("data/processed_safety_data.csv")
    return data

# Step 2: Normalize the Data
def normalize_data(data):
    """
    Normalize the safety parameters.
    """
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.iloc[:, :])
    data_normalized = pd.DataFrame(data_normalized, columns=data.columns[:])
    data_normalized["latitude"] = data["latitude"]
    data_normalized["longitude"] = data["longitude"]
    return data_normalized

def generate_weighted_labels(data_normalized):
    """
    Generate safety index labels using weighted features.
    """
    safety_scores = np.zeros(len(data_normalized))
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature in data_normalized.columns:
            safety_scores += data_normalized[feature].values * weight
    # Normalize to 0-1
    scaler = MinMaxScaler()
    return scaler.fit_transform(safety_scores.reshape(-1, 1)).flatten()

def adjust_features_for_context(features, time_of_travel):
    """
    Modify feature values based on real-time context (time of day, night, holidays).
    """
    hour = time_of_travel.hour
    is_night = hour >= 18 or hour <= 6

    # Find column names based on index
    feature_names = list(FEATURE_WEIGHTS.keys())
    feature_dict = dict(zip(feature_names, features.flatten()))

    # Apply contextual modifiers
    if is_night:
        feature_dict["is_night"] = 1
        feature_dict["cameras"] *= 1.5
        feature_dict["street_lights"] *= 1.5
    else:
        feature_dict["is_night"] = 0

    feature_dict["time_of_day"] = hour / 24.0
    feature_dict["day_of_week"] = time_of_travel.weekday() / 6.0  # normalize 0-1

    # Convert back to numpy array
    return np.array([feature_dict.get(f, 0) for f in feature_names]).reshape(1, -1)

# Step 3: Train ANN Model
def train_ANN_model(data_normalized):
    """
    Train an ANN model to predict the safety index.
    """
    # Generate synthetic labels for training (safety index)
    # Replace synthetic labels with weighted safety index
    safety_labels = generate_weighted_labels(data_normalized)


    # Define the ANN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="sigmoid", input_shape=(data_normalized.shape[1] - 2,)),  # Exclude lat/lon
        tf.keras.layers.Dense(64, activation="sigmoid"),
        tf.keras.layers.Dense(32, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Output safety index between 0 and 1
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(data_normalized.iloc[:, :-2], safety_labels, epochs=50, batch_size=10, verbose=0)

    return model

# Step 4: Fetch Road Network Data using OSMnx
def fetch_road_network(location_point, dist=1000):
    """
    Fetch road network data for a given location point.
    """
    G = ox.graph_from_point(location_point, dist=dist, network_type="drive")
    return G

# Step 5: Compute Safety Indices for Road Network Nodes
def compute_safety_indices(G, model, data_normalized, time_of_travel):
    """
    Compute safety indices for all nodes in the road network using the ANN model.
    """
    safety_indices = {}
    for node in G.nodes:
        # Find the nearest data point to the node
        node_lat, node_lon = G.nodes[node]["y"], G.nodes[node]["x"]
        distances = np.sqrt((data_normalized["latitude"] - node_lat) ** 2 + (data_normalized["longitude"] - node_lon) ** 2)
        nearest_index = distances.idxmin()
        
        # Extract features
        features_raw = data_normalized.iloc[nearest_index, :-2].values.reshape(1, -1)
        features = adjust_features_for_context(features_raw, time_of_travel)


        # Adjust features based on time of travel (e.g., street lights and cameras are more important at night)
        if time_of_travel.hour >= 18 or time_of_travel.hour <= 6:  # Night time
            features[0][4] *= 1.5  # Increase weight for street lights
            features[0][5] *= 1.5  # Increase weight for cameras

        # Predict safety index using the ANN model
        safety_index = model.predict(features)[0][0]
        safety_indices[node] = safety_index

    return safety_indices


def compute_safety_indices_for_path(G, model, data_normalized, time_of_travel, path):
    """
    Compute safety indices for nodes in the shortest path using the ANN model.
    """
    safety_indices = {}
    for node in path:
        # Find the nearest data point to the node
        node_lat, node_lon = G.nodes[node]["y"], G.nodes[node]["x"]
        distances = np.sqrt((data_normalized["latitude"] - node_lat) ** 2 + (data_normalized["longitude"] - node_lon) ** 2)
        nearest_index = distances.idxmin()
        
        # Extract features
        features_raw = data_normalized.iloc[nearest_index, :-2].values.reshape(1, -1)
        features = adjust_features_for_context(features_raw, time_of_travel)


        # Adjust features based on time of travel (e.g., street lights and cameras are more important at night)
        if time_of_travel.hour >= 18 or time_of_travel.hour <= 6:  # Night time
            features[0][4] *= 1.5  # Increase weight for street lights
            features[0][5] *= 1.5  # Increase weight for cameras

        # Predict safety index using the ANN model
        safety_index = model.predict(features)[0][0]
        safety_indices[node] = safety_index

    return safety_indices

def normalize_safety_indices(safety_indices):
    """
    Normalize the safety indices cumulatively to ensure values are between 0 and 1.
    """
    # Ensure all safety indices are positive
    min_safety = min(safety_indices.values())
    if min_safety < 0:
        # Shift all values to make them positive
        safety_indices = {node: safety_index - min_safety for node, safety_index in safety_indices.items()}
    
    # Normalize to [0, 1] range
    total_safety = sum(safety_indices.values())
    if total_safety == 0:
        # If total_safety is zero, assign equal normalized values
        normalized_safety_indices = {node: 1.0 / len(safety_indices) for node in safety_indices}
    else:
        normalized_safety_indices = {node: safety_index / total_safety for node, safety_index in safety_indices.items()}
    
    return normalized_safety_indices



# Step 6: Find Safest Path using Floyd-Warshall Algorithm
def find_safest_path(G, safety_indices, start_node, end_node):
    """
    Find the safest path using the Floyd-Warshall algorithm.
    """
    # Create a safety matrix where higher safety index is better
    num_nodes = len(G.nodes)
    node_list = list(G.nodes)
    safety_matrix = np.full((num_nodes, num_nodes), np.inf)
    next_matrix = np.full((num_nodes, num_nodes), -1, dtype=int)  # To reconstruct paths
    np.fill_diagonal(safety_matrix, 0)

    # Initialize safety matrix and next matrix
    for u, v, data in G.edges(data=True):
        u_idx = node_list.index(u)
        v_idx = node_list.index(v)
        # Use inverse of average safety index as edge weight (higher safety = lower weight)
        safety_matrix[u_idx][v_idx] = 1 - (safety_indices[u] + safety_indices[v]) / 2
        next_matrix[u_idx][v_idx] = v_idx  # Direct edge from u to v

    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if safety_matrix[i][j] > safety_matrix[i][k] + safety_matrix[k][j]:
                    safety_matrix[i][j] = safety_matrix[i][k] + safety_matrix[k][j]
                    next_matrix[i][j] = next_matrix[i][k]  # Update path through k

    # Reconstruct the safest path
    start_idx = node_list.index(start_node)
    end_idx = node_list.index(end_node)

    if next_matrix[start_idx][end_idx] == -1:
        raise ValueError(f"No path exists from {start_node} to {end_node}.")

    # Reconstruct the path
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
        # Access edge data using G[u][v]
        if G.has_edge(u, v):
            # If the graph has multi-edges, iterate over all edges between u and v
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Sum the lengths of all edges between u and v
                for key, data in edge_data.items():
                    total_length += data.get("length", 0)
            else:
                raise ValueError(f"No edge found between {u} and {v}.")
        else:
            raise ValueError(f"No edge found between {u} and {v}.")
    return total_length / 1000  # Convert meters to kilometers

# Step 9: Visualize Paths on Map
def display_paths_on_map(G, safest_path, shortest_path,  safety_indices_safe, safety_indices_short, safest_path_length, shortest_path_length):
    """
    Display the safest and shortest paths on a Folium map.
    """
    # Create a map centered at the start node
    start_lat, start_lon = G.nodes[safest_path[0]]["y"], G.nodes[safest_path[0]]["x"]
    map = folium.Map(location=[start_lat, start_lon], zoom_start=14)

    # Add the safest path
    safest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in safest_path]
    folium.PolyLine(safest_coords, color="green", weight=5, opacity=1, popup="Safest Path").add_to(map)

    # Add the shortest path
    shortest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in shortest_path]
    folium.PolyLine(shortest_coords, color="blue", weight=5, opacity=1, popup="Shortest Path").add_to(map)

    # Add markers for start and end points
    mid_idx_safe = len(safest_coords) // 2
    mid_idx_short = len(shortest_coords) // 2
    folium.Marker(location=safest_coords[mid_idx_safe], popup=f"Distance: {safest_path_length:.2f} | Safety Index: {safety_indices_safe:.4f}").add_to(map)
    folium.Marker(location=shortest_coords[mid_idx_short], popup=f"Distance: {shortest_path_length:.2f} | Safety Index: {safety_indices_short:.4f}").add_to(map)

    folium.Marker(safest_coords[0], popup=f"Start").add_to(map)
    folium.Marker(safest_coords[-1], popup=f"End").add_to(map)
    
    # Save and open the map
    map.save("safest_and_shortest_paths.html")
    webbrowser.open("safest_and_shortest_paths.html")

def open_path_in_google_maps(G, path, label="Safest"):
    """
    Open the path in Google Maps using waypoints.
    """
    coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in path]
    
    if len(coords) < 2:
        print("❌ Path is too short to visualize.")
        return

    origin = f"{coords[0][0]},{coords[0][1]}"
    destination = f"{coords[-1][0]},{coords[-1][1]}"
    waypoints = "|".join([f"{lat},{lon}" for lat, lon in coords[1:-1]])

    base_url = "https://www.google.com/maps/dir/?api=1"
    url = f"{base_url}&origin={origin}&destination={destination}"
    if waypoints:
        url += f"&waypoints={waypoints}"

    print(f"🌐 Opening {label} path in Google Maps...")
    webbrowser.open(url)

FAMOUS_LANDMARKS = {
    "Empire State Building": (40.74824320024852, -73.98607893014898),
    "Madison Square Garden": (40.75057700646688, -73.99346190022933),
    "Macy's": (40.75083909907612, -73.98913901873878),
    "Times Square": (40.75812095390535, -73.98561770524518),
    "New York University": (40.731360678299, -73.99713542107983),
    "Union Square": (40.73539785604513, -73.98990297150544),
    "Rockefeller Center": (40.75856590351697, -73.9809954973572),
    "Bryant Park": (40.75401815815159, -73.98405918662925),
    "High Line": (40.75310369185266, -74.00043461322687),
    "The Nicole": (40.76676842118233, -73.98728343772832),  # Known residential landmark near Columbus Circle
    "Osborne Apt House": (40.76572916976822, -73.98002229028022),  # Historic pre-war landmark building
    "Playwright Celtic Pub": (40.75934261074114, -73.98794348883031),  # Known Times Square pub
    "Pennsylvania Plaza": (40.74985509583679, -73.99288704294781),
    "Carpenters Union": (40.72919313866907, -74.00781050744203)  # Institutional landmark
}

def get_user_location_choice(data_normalized):
    """
    Display numbered location options to the user and get start/end choices.
    """
    print("\Choose Starting and Ending location from the available location choices :")
    # Convert to list for indexing
    landmark_names = list(FAMOUS_LANDMARKS.keys())

    # Display choices
    print("Available Landmarks:")
    for i, name in enumerate(landmark_names):
        print(f"{i}: {name}")

    # Get user selections
    start_idx = int(input("\nEnter the number of your START location: "))
    end_idx = int(input("Enter the number of your END location: "))

    # Get coordinates
    start_lat, start_lon = FAMOUS_LANDMARKS[landmark_names[start_idx]]
    end_lat, end_lon = FAMOUS_LANDMARKS[landmark_names[end_idx]]

    # Confirm selections
    print(f"\nStart Location: {landmark_names[start_idx]} ({start_lat}, {start_lon})")
    print(f"End Location: {landmark_names[end_idx]} ({end_lat}, {end_lon})")

    # for i, row in data_normalized[["latitude", "longitude"]].head(50).iterrows():
    #     print(f"{i}: Latitude: {row['latitude']:.5f}, Longitude: {row['longitude']:.5f}")
    
    # start_index = int(input("\nEnter START location index: "))
    # end_index = int(input("Enter END location index: "))

    # start_lat = data_normalized.iloc[start_index]["latitude"]
    # start_lon = data_normalized.iloc[start_index]["longitude"]
    # end_lat = data_normalized.iloc[end_index]["latitude"]
    # end_lon = data_normalized.iloc[end_index]["longitude"]

    return (start_lat, start_lon), (end_lat, end_lon)


# Step 10: Main Function
def main():
    # Load and normalize the data
    data = load_data()
    data_normalized = normalize_data(data)

    model_details(data_normalized)
    # Train the ANN model
    model = train_ANN_model(data_normalized)

    # Fetch road network data
    location_point = (40.74597019293782, -73.9906438091723)  # Central Manhattan
    G = fetch_road_network(location_point)

    # Let user choose from known locations
    (start_lat, start_lon), (end_lat, end_lon) = get_user_location_choice(data_normalized)

    # Get current time (or modify to user-input datetime if needed)
    time_of_travel = datetime.now()

    # Compute safety indices for all graph nodes
    print("Computing safety indices across network...")
    safety_indices = compute_safety_indices(G, model, data_normalized, time_of_travel)

    # Get nearest graph nodes to selected coordinates
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    # Find safest and shortest paths
    print("Computing safest path...")
    safest_path = find_safest_path(G, safety_indices, start_node, end_node)
    shortest_path = find_shortest_path(G, start_node, end_node)

    # Compute safety indices along paths
    safety_indices_safest = compute_safety_indices_for_path(G, model, data_normalized, time_of_travel, safest_path)
    safety_indices_shortest = compute_safety_indices_for_path(G, model, data_normalized, time_of_travel, shortest_path)

    # Normalize and compare path safety
    norm_safety_safest = normalize_safety_indices(safety_indices_safest)
    norm_safety_shortest = normalize_safety_indices(safety_indices_shortest)

    mean_safety_index = sum(norm_safety_safest.values()) / len(norm_safety_safest)
    mean_shortest_index = sum(norm_safety_shortest.values()) / len(norm_safety_shortest)

    safest_path_length = calculate_path_length(G, safest_path)
    shortest_path_length = calculate_path_length(G, shortest_path)

    # Print comparison
    print(f"\n--- Route Analysis ---")
    print(f"Safest Path Length: {safest_path_length:.2f} km")
    print(f"Shortest Path Length: {shortest_path_length:.2f} km")
    print(f"Mean Safety (Safest Path): {mean_safety_index:.3f}")
    print(f"Mean Safety (Shortest Path): {mean_shortest_index:.3f}")

    # Visualize
    display_paths_on_map(G, safest_path, shortest_path, mean_safety_index, mean_shortest_index, safest_path_length, shortest_path_length)
    open_path_in_google_maps(G, safest_path, label="Safest")
    open_path_in_google_maps(G, shortest_path, label="Shortest")


# Run the program
if __name__ == "__main__":
    main()