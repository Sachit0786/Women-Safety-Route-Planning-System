# generate_safety_data.py
import numpy as np
import pandas as pd
import requests
import time
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

# Define all expected features in final dataset
FEATURES = [
    "cameras", "street_lights", "police_stations", "public_transport", "crime_rate", "accidents",
    "shops", "construction_zones", "crowd_density", "parks_recreation", "population_density",
    "traffic_density", "market_areas", "time_of_day", "sidewalk_presence", "is_festival",
    "hospitality_venues", "emergency_services", "is_holiday", "is_night", "day_of_week",
    "religious_places", "educational_institutions"
]

def query_with_timeout(server, tag, lat, lon):
    try:
        url = f"{server}?data=[out:json];node[{tag}](around:1000,{lat},{lon});out;"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        elements = response.json().get("elements", [])
        return len(elements)
    except Exception as e:
        print(f"❌ FAIL: {tag} @ ({lat:.4f},{lon:.4f}) | {e}")
        return 0

def fetch_live_data(latitudes, longitudes):
    data = {key: [] for key in FEATURES}
    data["latitude"] = []
    data["longitude"] = []

    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ]

    tags_mapping = {
        "amenity=police": "police_stations",
        "highway=street_lamp": "street_lights",
        "shop": "shops",
        "public_transport": "public_transport",
        "amenity=marketplace": "market_areas",
        "amenity~\"restaurant|bar|pub|cafe|hotel\"": "hospitality_venues",
        "emergency": "emergency_services",
        "amenity=place_of_worship": "religious_places",
        "amenity~\"school|college|university\"": "educational_institutions",
        "leisure~\"park|recreation_ground\"": "parks_recreation"
    }

    for lat, lon in zip(latitudes, longitudes):
        print(f"\n📍 Processing ({lat:.4f}, {lon:.4f})")
        result = {feature: 0 for feature in FEATURES}
        result["latitude"] = lat
        result["longitude"] = lon

        for tag, label in tags_mapping.items():
            count = 0
            for server in overpass_servers:
                count = query_with_timeout(server, tag, lat, lon)
                if count >= 0:
                    break
            result[label] = count

        # Simulated features
        result.update({
            "cameras": np.random.uniform(0, 1),
            "crime_rate": np.random.uniform(0, 1),
            "accidents": np.random.uniform(0, 1),
            "construction_zones": np.random.uniform(0, 1),
            "crowd_density": np.random.uniform(0, 1),
            "population_density": np.random.uniform(0, 1),
            "traffic_density": np.random.uniform(0, 1),
            "sidewalk_presence": np.random.uniform(0, 1),
            "time_of_day": np.random.uniform(0, 1),
            "day_of_week": np.random.uniform(0, 1),
            "is_night": np.random.randint(0, 2),
            "is_holiday": np.random.randint(0, 2),
            "is_festival": np.random.randint(0, 2)
        })

        # Append each location only once after building result dict
        for key in data:
            data[key].append(result.get(key, 0))  # No double-appending

    return pd.DataFrame(data)

def apply_kde_and_normalize(data, bandwidth=0.02):
    kde_data = pd.DataFrame(index=data.index)
    scaler = MinMaxScaler()

    # Normalize all features except lat/lon
    for column in FEATURES:
        values = data[column].values.reshape(-1, 1)
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
            log_density = kde.score_samples(values)
            density = np.exp(log_density)
            normalized = scaler.fit_transform(density.reshape(-1, 1)).flatten()
            kde_data[column] = normalized
        except Exception as e:
            print(f"⚠️ KDE failed on {column}, using raw normalization instead. {e}")
            kde_data[column] = scaler.fit_transform(values).flatten()

    # Add coordinates as-is
    kde_data["latitude"] = data["latitude"]
    kde_data["longitude"] = data["longitude"]

    return kde_data

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


def generate_and_process_data():
    np.random.seed(42)

    latitudes = {40.74824320024852, 40.75057700646688, 40.75083909907612, 40.75812095390535, 40.731360678299, 40.73539785604513, 40.75856590351697, 40.74985509583679, 40.73125759597995, 40.72496926168325, 40.73710522892537, 40.74978066831234, 40.73674121186265, 40.74482611056035, 40.74387603006983, 40.75724925252236, 40.76572916976822, 40.72919313866907, 40.75567317858561, 40.74623419830452, 40.73360439112273, 40.74807070576344, 40.75934261074114, 40.72841661476453, 40.73637953733218, 40.7609542073833, 40.75331767793492, 40.72518170759591, 40.7400461140816, 40.74209620040558, 40.75503947044712, 40.75305275090884, 40.73284828131155, 40.75038137082496, 40.74746008960899, 40.759683205204, 40.75032816512953, 40.74077696416769, 40.72791922747654, 40.74191596898865, 40.74743511998196, 40.75243062610118, 40.75401815815159, 40.75461924534481, 40.75583792497979, 40.74204117359164, 40.7293636535487, 40.76676842118233, 40.75310369185266, 40.76108234344288, 40.73929902794672, 40.74184894994014, 40.75853537556232, 40.75694633803501, 40.73503135541941, 40.74422332238863, 40.75622361929059, 40.74921750281805, 40.75842080400336, 40.7385616144058, 40.7323163986276, 40.75036495067326, 40.75946884453123, 40.75172261995026, 40.73163119242256, 40.75165313631373, 40.74330371813281, 40.72799548038213, 40.74847820189744, 40.75934509725126, 40.73456631644027, 40.75431809929528, 40.75920724143178, 40.76058690407451, 40.72344485575784, 40.75608225896129, 40.74720113080608, 40.74217284168541, 40.75595399630801, 40.72730104936672, 40.74432706509946, 40.75101122291183, 40.74410415034006, 40.76387982840509, 40.75277235927258, 40.75774338869491, 40.74627864314523, 40.75429739605477, 40.74992923088533, 40.72746956240574, 40.74792282882824, 40.75038954863334, 40.75731444016323, 40.74799568144429, 40.74429666620373, 40.75454733242822, 40.72747216265145, 40.75252100508699, 40.75494455974496, 40.74805272787287}
    longitudes = {-73.98607893014898, -73.99346190022933, -73.98913901873878, -73.98561770524518, -73.99713542107983, -73.98990297150544, -73.9809954973572, -73.99288704294781, -73.98072544118014, -73.98604391177852, -73.98538736545574, -73.99466481261811, -74.00825288360392, -73.98409222762554, -73.99074148825785, -73.99164882103334, -73.98002229028022, -74.00781050744203, -73.98349914713086, -73.97782328504871, -73.98089782488906, -73.99191690340106, -73.98794348883031, -74.00222269509491, -73.99860558544212, -73.99253182506301, -74.00332072433278, -73.99385880852559, -73.97795379957618, -73.99635036134964, -73.99102552778774, -73.99688679049922, -73.98773008302507, -73.99684974116647, -73.9967899738108, -73.99652770016128, -73.9768619870684, -73.98390030077559, -74.00530878745433, -73.99787712056585, -74.00220743994672, -73.99425450456809, -73.98405918662925, -73.9883282296435, -73.98144754488963, -73.98569958054858, -73.99554292567912, -73.98728343772832, -74.00043461322687, -73.97842692839936, -73.9977525770807, -73.99073882193122, -73.99744765509614, -73.98958977682634, -73.98703442752861, -73.9970858614371, -73.98460369655572, -73.99471038143123, -74.00018565701012, -74.00098977142927, -73.99594494054489, -73.97834855773253, -73.99103814369437, -73.98475869822512, -73.99536325961205, -73.98605116617793, -73.98548458003778, -74.00311651899278, -73.9777927647502, -73.99230597912181, -73.97807894479733, -73.97762644224184, -73.98339440270664, -73.99342683362718, -73.99934072092897, -73.97432810617171, -73.99139381822823, -73.99994230618059, -73.99169387048391, -74.00259367579869, -73.98642080794409, -73.99154067131235, -73.9959047505731, -73.98459682808685, -73.99906545051503, -73.99454735564147, -73.9911406203838, -73.98677854785423, -73.98045286137994, -73.98718155660633, -73.97788928534648, -73.9839595891738, -73.97575834654228, -73.98949214472643, -73.98549089194785, -73.98763188196551, -73.99609326262133, -74.00024274142906, -73.99366822000084, -74.00529681144728}
    num_locations = len(latitudes)
    print("\n📡 Fetching live/simulated data...")
    raw_data = fetch_live_data(latitudes, longitudes)
    raw_data.to_csv("data/raw_safety_data.csv", index=False)
    print("✅ Raw data shape:", raw_data.shape)

    print("\n📈 Applying KDE and normalizing...")
    processed = apply_kde_and_normalize(raw_data)
    print("✅ Final processed shape:", processed.shape)
    assert len(processed) == num_locations, (
    f"❌ Row mismatch! Expected {num_locations}, got {len(processed)}."
    )

    print("\n✅ Validating output consistency...")
    assert processed.shape[0] == num_locations, f"❌ Mismatch: {processed.shape[0]} rows vs {num_locations} locations"
    assert not processed.isnull().any().any(), "❌ Missing values detected in final data!"

    processed.to_csv("data/processed_safety_data.csv", index=False)
    print("✅ Processed safety data saved to 'processed_safety_data.csv'")


if __name__ == "__main__":
    generate_and_process_data()
