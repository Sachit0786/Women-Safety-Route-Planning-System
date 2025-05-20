import random

# Original points
latitudes = [
    40.74824320024852,
    40.750577006466884,
    40.750839099076124,
    40.75812095390535,
    40.731360678299005,
    40.735397856045125
]

longitudes = [
    -73.98607893014898, 
    -73.99346190022933, 
    -73.98913901873878, 
    -73.98561770524518, 
    -73.99713542107983, 
    -73.98990297150544 
]

# Function to jitter coordinates (within ~100–300m)
def jitter_coords(lat, lon, max_offset_m=1000):
    # Convert meters to degrees roughly (latitude and longitude)
    lat_offset = random.uniform(-1, 1) * max_offset_m / 111_000  # 1 deg ≈ 111km
    lon_offset = random.uniform(-1, 1) * max_offset_m / (111_000 * abs(math.cos(math.radians(lat))))
    return lat + lat_offset, lon + lon_offset

import math

new_latitudes = []
new_longitudes = []

while len(new_latitudes) < 94:
    base_idx = random.randint(0, len(latitudes) - 1)
    base_lat = latitudes[base_idx]
    base_lon = longitudes[base_idx]

    new_lat, new_lon = jitter_coords(base_lat, base_lon)
    new_latitudes.append(new_lat)
    new_longitudes.append(new_lon)

# Combine with original
all_latitudes = latitudes + new_latitudes
all_longitudes = longitudes + new_longitudes

# Print all points
print("📍 Generated 15 locations:")
for i, (lat, lon) in enumerate(zip(all_latitudes, all_longitudes), 1):
    print(f"{i:2}: Latitude: {lat:.14f}, Longitude: {lon:.14f}")
