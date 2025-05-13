import streamlit as st
import pydeck as pdk
import pandas as pd
import csv

# Title
st.title("Safest Path Finder")

# Predefined locations
locations = {}

# Read the CSV file
with open('locations_coordinates.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        location_name = row['Location']
        latitude = float(row['Latitude'])
        longitude = float(row['Longitude'])
        locations[location_name] = [latitude, longitude]

# Predefined path coordinates
path_coordinates = []

with open('path.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        latitude = float(row[0])
        longitude = float(row[1])
        path_coordinates.append((latitude, longitude))

# Select starting and ending locations
start = st.selectbox("Select Starting Location", list(locations.keys()))
end = st.selectbox("Select Ending Location", list(locations.keys()))

start_coord = locations[start]
end_coord = locations[end]

# Determine sub-path between start and end in path_coordinates
def get_sub_path(path_coords, start_pt, end_pt):
    try:
        start_idx = path_coords.index(start_pt)
        end_idx = path_coords.index(end_pt)
        if start_idx <= end_idx:
            return path_coords[start_idx:end_idx + 1]
        else:
            return path_coords[end_idx:start_idx + 1][::-1]
    except ValueError:
        return []

# Calculate sub-path
path_segment = get_sub_path(path_coordinates, start_coord, end_coord)

# Build dataframe for path
df = pd.DataFrame(path_segment, columns=['lat', 'lon'])

# Map view settings
view_state = pdk.ViewState(
    latitude=df['lat'].mean(),
    longitude=df['lon'].mean(),
    zoom=16,
    pitch=0,
)

# Define the path layer
path_layer = pdk.Layer(
    'PathLayer',
    data=[{"path": path_segment, "name": "Route"}],
    get_path='path',
    get_color=[255, 0, 0],
    width_scale=10,
    width_min_pixels=3,
)

# Define start and end markers
marker_layer = pdk.Layer(
    'ScatterplotLayer',
    data=[{"position": start_coord}, {"position": end_coord}],
    get_position='position',
    get_color=[0, 0, 255],
    get_radius=10,
)

# Display map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/streets-v11',
    initial_view_state=view_state,
    layers=[path_layer, marker_layer],
))
