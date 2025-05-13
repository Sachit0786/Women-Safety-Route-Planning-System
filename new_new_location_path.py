import folium
from flask import Flask, render_template
import pandas as pd
import csv

app = Flask(__name__)

# Predefined locations
locations = {}

# Read the CSV file for locations
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

# Function to determine sub-path between start and end
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

# Example coordinates for start and end
start_coord = locations.get('NITJ Temple')
end_coord = locations.get('Guest House')

# Calculate sub-path
path_segment = get_sub_path(path_coordinates, start_coord, end_coord)

# Ensure path_segment is not empty before trying to calculate the map center
if path_segment:
    map_center = [sum([point[0] for point in path_segment]) / len(path_segment),
                  sum([point[1] for point in path_segment]) / len(path_segment)]
else:
    # Handle case where path_segment is empty (fallback to start location)
    print("No valid path found between the selected locations.")
    map_center = [start_coord[0], start_coord[1]]  # You can also fallback to end_coord

# Now use map_center for further map operations
# Create a map centered at the midpoint of the path
map_center = [sum([point[0] for point in path_segment]) / len(path_segment), 
              sum([point[1] for point in path_segment]) / len(path_segment)]

mymap = folium.Map(location=map_center, zoom_start=16)

# Add path to the map
folium.PolyLine(path_segment, color="red", weight=3, opacity=1).add_to(mymap)

# Add markers for start and end locations
folium.Marker(start_coord, popup='Start Location', icon=folium.Icon(color='green')).add_to(mymap)
folium.Marker(end_coord, popup='End Location', icon=folium.Icon(color='blue')).add_to(mymap)

# Save map as an HTML file
mymap.save("safest_path_map.html")

@app.route('/')
def index():
    # Display map in the web browser
    return render_template('index.html', map_file="safest_path_map.html")

if __name__ == '__main__':
    app.run(debug=True)
