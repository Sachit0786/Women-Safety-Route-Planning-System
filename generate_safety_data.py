import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
import time

# NITJ locations data
NITJ_LOCATIONS = {
    "NITJ Temple": (31.391959993572907, 75.53602672921012),
    "Community Centre": (31.393117955541946, 75.536217201339),
    "Children's park": (31.3931841788518, 75.5387161665682),
    "Director Bungalow": (31.393302226399296, 75.53729275998296),
    "Guest House": (31.393649344276298, 75.53671119001272),
    "SBI ATM": (31.393948083883764, 75.53739719936297),
    "NITJ Depatmental Store": (31.394028461730645, 75.53733055098061),
    "NITJ Book Shop": (31.394084787969113, 75.53740361311607),
    "Juice Shop": (31.39408530140987, 75.53728874002925),
    "NITJ Post Office": (31.39409058882013, 75.5375969022263),
    "Canara Bank ATM": (31.39410274073723, 75.53729344626146),
    "Girls Hostel 2": (31.394151637413113, 75.53899681128486),
    "NITJ Main Gate": (31.39415791790809, 75.53297552271245),
    "Dispensary NITJ": (31.394209555472536, 75.53757522247375),
    "Nescafe": (31.394212454423144, 75.53707045450096),
    "Mega Girls Hostel Phase 2": (31.39433710258572, 75.53986666560951),
    "PNB Bank ATM": (31.394424815951577, 75.53299978178094),
    "Girls Hostel 1 Basketball Court": (31.394492183854005, 75.53862430835488),
    "IT Building Parking Lot": (31.394499478728015, 75.53535415714909),
    "Lawn Tennis Ground": (31.394608179910602, 75.53705725037588),
    "Girls Hostel 1": (31.394662988229424, 75.53852493041805),
    "OAT NITJ": (31.39471947180314, 75.53365841221073),
    "IT Building": (31.394909304732256, 75.53574844185675),
    "Parking Lot NITJ": (31.395171674027598, 75.5330250839097),
    "IPE Department": (31.395263035801754, 75.53644581618946),
    "Department of Textile Engineering": (31.39527292699817, 75.53750300629967),
    "Badminton Court NITJ": (31.395390780763808, 75.53253988217514),
    "Gymnasium": (31.395390780763808, 75.53280596054317),
    "Central Lawn NITJ": (31.395412483166492, 75.53500608064398),
    "IPE Department Parking Space": (31.395540066973886, 75.53589059893815),
    "Department of Chemical Engineering": (31.3955610867787, 75.53743332138073),
    "Department of Civil Engineering": (31.39567476425579, 75.53698578932996),
    "Sub Station C": (31.395704910288785, 75.53833025524492),
    "Administrative Block": (31.39583253318301, 75.53561724456935),
    "Academic Block": (31.39596479487298, 75.5375728870548),
    "CSH": (31.395991962291532, 75.53449495138571),
    "Department of Electronics and Communication Engineeering": (31.396056772422902, 75.53686035646894),
    "Admin Block Parking": (31.39620077582308, 75.53538071039237),
    "Campus Cafe": (31.39645860632206, 75.53672098664221),
    "Old Lecture Theatre 2": (31.396514987935433, 75.53749189428869),
    "LT side sitting left": (31.39654452453418, 75.53729240295566),
    "Drawing Hall": (31.396583862887343, 75.53596710316745),
    "Basketball Court NITJ": (31.396595858638882, 75.5332942926124),
    "NITJ Library": (31.39668029211444, 75.53519255819442),
    "NITJ Reading Room": (31.39669176376774, 75.53524631596655),
    "NITJ Reading Hall": (31.396724754440527, 75.53483625085988),
    "Lecture theatre": (31.396796991425134, 75.53716697010462),
    "New Lecture Hall": (31.396838268529827, 75.53490184523471),
    "Snackers": (31.396848778657098, 75.53406252923737),
    "Volleyball Court NIT": (31.396861775517003, 75.53335502078833),
    "LT side sitting right": (31.396948999964813, 75.53691300728593),
    "Central AC Point": (31.397028603649822, 75.53790661911712),
    "Department of Chemistry": (31.39703209171694, 75.53459146327725),
    "Instrumentation Centre": (31.397100855883362, 75.53453775664008),
    "Old Lecture Theatre 1": (31.397102625132167, 75.53701943651595),
    "Atletics Ground Store Room": (31.397177564466386, 75.53059917525076),
    "Department of Mechanical Engineering": (31.397189569494817, 75.53639424176042),
    "Athletics Ground": (31.39720536915965, 75.53153794182006),
    "Atletics Ground Toilet": (31.397254550566366, 75.53062089490207),
    "Department of Mathematics and Computing": (31.397262268961004, 75.53584450517957),
    "Peacock Spot": (31.397413725692104, 75.53747502079298),
    "BH1": (31.39743795261974, 75.53289679739795),
    "Department of Physics": (31.39744138777314, 75.53446496696046),
    "BH1 Badminton Court": (31.397445207465783, 75.53267581365303),
    "BH2 Volletball Court": (31.397695703612026, 75.53238757664974),
    "Yadav Canteen": (31.39772886598103, 75.53661878205655),
    "BH1 Volletball Court": (31.39776716084096, 75.53318035504702),
    "manufacturing Workshop": (31.397804068161754, 75.53473412284632),
    "Perianth Racing Workshop": (31.397870068932498, 75.53582682757501),
    "BH2": (31.397873221626412, 75.5325638276788),
    "Department of Biotechnology": (31.39803168605705, 75.53558372395207),
    "BH2 Badminton Court": (31.398095479003498, 75.53290119700391),
    "Water Tank": (31.39817227631624, 75.53733288791285),
    "Domino's": (31.398179915007372, 75.53403054423983),
    "Cricket Ground": (31.398184616839725, 75.53064439108017),
    "BH3": (31.398385577207552, 75.53347430630245),
    "BH6": (31.3984873320106, 75.53633449398527),
    "Athletics Ground Side Road": (31.3985338075491, 75.53156375886535),
    "BH5": (31.398585062598006, 75.53200643786425),
    "Night Canteen": (31.398784858646923, 75.53535121007981),
    "BH6 Volleyball Ground": (31.39886913486576, 75.5357101308705),
    "BH4": (31.398869566083988, 75.5328738254875),
    "BH7": (31.39896299450123, 75.53702408910564),
    "MBH Ground": (31.39897655164169, 75.53441534729825),
    "BH6 Badminton Ground": (31.399055263194573, 75.53612947923116),
    "Mega Boys Hostel B badminton Ground": (31.399134072807446, 75.53601029058575),
    "Mega Boys Hostel A": (31.399181379743496, 75.53532984392156),
    "Mega Boys Hostel A badminton Ground": (31.399273727496038, 75.53555163282373),
    "BH7E": (31.399336841251674, 75.5373073822204),
    "BH7 Badminton Ground": (31.399381384595017, 75.53650596090891),
    "Mega Boys Hostel B": (31.399388188287038, 75.53594861572485),
    "Institute Stage": (31.39956273942574, 75.53390972849074),
    "Mega Guest House": (31.3996143878155, 75.53517988551279),
    "Mega Boys Hostel F": (31.399874981994873, 75.53476512143155)
}

# Function to fetch live data from OpenStreetMap and other APIs
def fetch_live_data(latitudes, longitudes):
    data = {
        "crime_rate": [],
        "accidents": [],
        "police_stations": [],
        "street_lights": [],
        "cameras": [],
        "public_transport": [],
        "shops": [],
        "crowd_density": [],
        "construction_zones": [],
        "residential_area": [],
    }

    # List of Overpass API servers to try
    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ]

    for lat, lon in zip(latitudes, longitudes):
        # Try each Overpass server until one succeeds
        for server in overpass_servers:
            try:
                # Query OpenStreetMap for police stations near the coordinates
                police_stations_url = f"{server}?data=[out:json];node[amenity=police](around:1000,{lat},{lon});out;"
                response = requests.get(police_stations_url, timeout=10)
                police_stations_data = response.json()
                police_stations_count = len(police_stations_data.get("elements", []))

                # Query OpenStreetMap for street lights near the coordinates
                street_lights_url = f"{server}?data=[out:json];node[highway=street_lamp](around:1000,{lat},{lon});out;"
                response = requests.get(street_lights_url, timeout=10)
                street_lights_data = response.json()
                street_lights_count = len(street_lights_data.get("elements", []))

                # Query OpenStreetMap for shops near the coordinates
                shops_url = f"{server}?data=[out:json];node[shop](around:1000,{lat},{lon});out;"
                response = requests.get(shops_url, timeout=10)
                shops_data = response.json()
                shops_count = len(shops_data.get("elements", []))

                # Query OpenStreetMap for public transport stops near the coordinates
                public_transport_url = f"{server}?data=[out:json];node[public_transport](around:1000,{lat},{lon});out;"
                response = requests.get(public_transport_url, timeout=10)
                public_transport_data = response.json()
                public_transport_count = len(public_transport_data.get("elements", []))

                # Simulate crime rate and accidents (replace with actual API calls if available)
                crime_rate = np.random.uniform(0, 1)  # Simulated crime rate
                accidents = np.random.uniform(0, 1)  # Simulated accident rate

                # Simulate other parameters (replace with actual API calls if available)
                cameras = np.random.uniform(0, 1)  # Simulated CCTV camera density
                crowd_density = np.random.uniform(0, 1)  # Simulated crowd density
                construction_zones = np.random.uniform(0, 1)  # Simulated construction zones
                residential_area = np.random.uniform(0, 1)  # Simulated residential area density

                # Append data to the dictionary
                data["crime_rate"].append(crime_rate)
                data["accidents"].append(accidents)
                data["police_stations"].append(police_stations_count)
                data["street_lights"].append(street_lights_count)
                data["cameras"].append(cameras)
                data["public_transport"].append(public_transport_count)
                data["shops"].append(shops_count)
                data["crowd_density"].append(crowd_density)
                data["construction_zones"].append(construction_zones)
                data["residential_area"].append(residential_area)

                # Break out of the retry loop if successful
                break

            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"Error with server {server}: {e}. Retrying with next server...")
                time.sleep(2)  # Wait before retrying
                continue

    return pd.DataFrame(data)

# Function to apply KDE and normalize data
def apply_kde_and_normalize(data, bandwidth=0.02):
    """
    Apply Kernel Density Estimation (KDE) to each parameter and normalize the results.
    """
    scaler = MinMaxScaler()
    kde_data = pd.DataFrame()

    for column in data.columns:
        # Reshape data for KDE
        values = data[column].values.reshape(-1, 1)

        # Fit KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)

        # Evaluate KDE
        log_density = kde.score_samples(values)
        density = np.exp(log_density)  # Convert log density to density

        # Normalize density to [0, 1]
        normalized_density = scaler.fit_transform(density.reshape(-1, 1)).flatten()

        # Add to DataFrame
        kde_data[column] = normalized_density

    return kde_data

# Main function to generate and process data
def generate_and_process_data():
    # Extract latitudes and longitudes from NITJ locations
    latitudes = [coord[0] for coord in NITJ_LOCATIONS.values()]
    longitudes = [coord[1] for coord in NITJ_LOCATIONS.values()]

    # Fetch live data
    print("Fetching live data...")
    data = fetch_live_data(latitudes, longitudes)

    # Apply KDE and normalize
    print("Applying KDE and normalizing data...")
    processed_data = apply_kde_and_normalize(data)

    # Add latitudes and longitudes to the processed data
    processed_data["latitude"] = latitudes
    processed_data["longitude"] = longitudes
    processed_data["location_name"] = list(NITJ_LOCATIONS.keys())

    # Save to CSV
    processed_data.to_csv("processed_safety_data.csv", index=False)
    print("Processed data saved to 'processed_safety_data.csv'.")

# Generate and process data
generate_and_process_data()