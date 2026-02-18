from shapely.geometry import Point
from pathlib import Path
import geopandas as gpd
from math import sqrt
from sqlalchemy.orm import Session
from service.entities import Adress
from service.requestforms import LocationRequest
import re
import unicodedata

swiss_shp = Path("../resources/map/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp")
gdf = None
cantons_gdf = { 'Genève': 'GE', 'Thurgau': 'TG', 'Valais': 'VS', 'Aargau': 'AG', 'Schwyz': 'SZ', 'Zürich': 'ZH', 'Obwalden': 'OW',
                'Fribourg': 'FR', 'Glarus': 'GL', 'Uri' : 'UR', 'Nidwalden' : 'NW', 'Solothurn' : 'SO', 'Appenzell Ausserrhoden' : 'AR',
                'Jura' : 'JU', 'Graubünden' : 'GR', 'Vaud' : 'VD', 'Luzern' : 'LU', 'Ticino' : 'TI', 'Zug' : 'ZG', 'Basel-Landschaft' : 'BL',
                'St. Gallen' : 'SG', 'Schaffhausen' : 'SH', 'Bern' : 'BE', 'Basel-Stadt' : 'BS', 'Neuchâtel': 'NE', 'Appenzell Innerrhoden': 'AI'
                }

def init_locations():
    global gdf
    gdf = gpd.read_file(swiss_shp)
    gdf = gdf.to_crs(epsg=4326)

def get_canton(lat, lon):
    global gdf
    point = Point(lon, lat)
    match = gdf[gdf.contains(point)]
    if not match.empty:
        return cantons_gdf[match.iloc[0]['NAME']]
    return None

def get_city(lat: float, long: float, db: Session):
    try:
        lat = float(lat)
        long = float(long)
        lower_lat = lat - float(0.01)
        upper_lat = lat + float(0.01) # approx 1km
        lower_long = long - float(0.005)
        upper_long = long + float(0.005)

        results = db\
            .query(Adress.coord_x, Adress.coord_y, Adress.name)\
            .where(Adress.coord_x.between(lower_lat, upper_lat),
                   Adress.coord_y.between(lower_long, upper_long)
            )\
            .all()
        closest_city = None
        min_distance = float('inf')

        for coord_x, coord_y, name in results:
            distance = sqrt((float(coord_x) - lat) ** 2 + (float(coord_y) - long) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_city = name
        return closest_city
    except Exception:
        raise Exception()

def get_location(request: LocationRequest, db: Session):
    zips = request.preprocessed.zip_codes if request.preprocessed else None
    streets = request.preprocessed.streets if request.preprocessed else None
    try:
        #return return_coords(2892956, msg="Testing")
        raw = request.raw_text.lower()
        if zips: # I think this should work even for the case where we have multiple zips. We are comparing them wtih cities and raw_text. This should hold
            print(zips)
            zip_list = list(zips)
            # decide which zip fits. I assume that everytime a zip is on a poster, I have city somewhere too. (Nobody advertises by "Come join us in 4056" wtf)
            zip_and_cities = db\
                .query(Adress.zip, Adress.name)\
                .where(Adress.zip.in_(zip_list))\
                .all()
            matching = []
            for zip_code, city in zip_and_cities:
                cleaned_city = remove_brackets(city)
                if cleaned_city.lower() in raw and (zip_code, city) not in matching: # We need to find the correct zip
                    # consider adding fuzzing here for matching typos
                    matching.append((zip_code, city))
            # Now we have a list of zip codes with their matching cities, where the city names
            # are part of the raw text given by the client. This list is most likely going to be rather small
            print("Matching zip/city pairs:", matching)
            if len(matching) == 1:
                streets_in_zip = db\
                    .query(Adress.street)\
                    .where(Adress.zip == matching[0][0], Adress.name == matching[0][1])\
                    .all()
                zip_street = None # determine the street
                normalized_raw = normalize(raw)
                for (street,) in streets_in_zip:
                    norm_street = normalize(street)
                    if norm_street in normalized_raw:
                        zip_street = street
                        break
                if zip_street is not None:
                    print(f"Processing zip_street: {zip_street}")
                    number_results = db\
                        .query(Adress.id, Adress.number)\
                        .where(Adress.zip == matching[0][0], Adress.name == matching[0][1], Adress.street == zip_street)\
                        .all()
                    potential_number = None
                    if " " in zip_street:
                        potential_number = find_number_after(zip_street.lower().replace(' ', ''), normalized_raw.replace(' ', ''))
                    else:
                        potential_number = word_after(raw, zip_street)
                    if potential_number and contains_digit(potential_number):
                        # Get closest number
                        closest_number_id, closest_number = find_best_match(potential_number, number_results)
                        if closest_number == potential_number:
                            return return_coords_content(closest_number_id, f"This is the location of {zip_street} {closest_number}")
                        else:
                            return return_coords_content(closest_number_id, f"This is the next closest known location: {zip_street} {closest_number}")
                    else:
                        print("Potential_number is None")
                        # return default of this or closest
                        id_and_lowest_number = sorted(number_results, key=sort_key)
                        return return_coords_content(id_and_lowest_number[0][0], f"Location originating from zip {zip_street} without potential number.")
            else:
                print("Tie-Breaking or no match has been found!")

        print("Trying to get location using streets...")
        if streets:
            print("Beginning in streets")
            # Deduplicate and filter out empty/null strings
            streets = list({s.strip().lower() for s in streets if s and s.strip()})
            # We have zip as well as some streets
            # We might want to run a query which verifies,if there is a match between City and Zip (any word in raw_text is city name for zip)
            # We have a lot of possible street names, as we can check the legality pretty good
            print(f"Streets: {streets}")
            name_street = db\
                .query(Adress.name, Adress.street, Adress.normalized_street)\
                .where(Adress.normalized_street.in_(streets))\
                .all()
            if len(name_street) == 0:
                print("No street was found.")
                return 404, {"error": "No streets found"}

            main_city, main_street = "", ""
            for city, street, normalized_street in name_street: # We need to decide which city the correct one is
                cleaned_city = remove_brackets(city)
                if cleaned_city.lower() in raw: # Might want to add similarity check here (Levenshtein)
                    # We very likely found our city
                    main_city = city
                    main_street = street
                    break

            potential_number = None
            if " " in main_street:
                print(f"find_number_after {raw.replace(' ' , '')} {main_street.lower().replace(' ', '')}")
                potential_number = find_number_after(main_street.lower().replace(" ", ""), raw.replace(" ", ""))
            else:
                print("word_after")
                potential_number = word_after(raw, main_street)
            print(f"Potential number is: {potential_number}, main_city: {main_city} and main_street: {main_street}")
            number_results = db\
                .query(Adress.id, Adress.number)\
                .where(Adress.name == main_city, Adress.street == main_street)\
                .all()
            if potential_number and contains_digit(potential_number):
                closest_number_id, _ = find_best_match(potential_number, number_results)
                if closest_number_id:
                    return return_coords_content(closest_number_id, "Location retrieved by using addresses as startpoint.")
            else:
                # We do not have a number, therefore we want the lowest number there is an entry for
                id_and_lowest_number = sorted(number_results, key=sort_key)
                return return_coords_content(id_and_lowest_number[0][0], "Location retrieved by using addresses as startpoint without potential number.")
        return 404, {"error": "No location has been found!"}
    except IndexError as e:
        print(f"Exception in get_location: {str(e)}")
        return 404, {"error": "No location has been found!"}
    except KeyError as e:
        print(f"KeyError in get_location: {str(e)}")
        return 505, {"error": str(e)}
    except Exception as e:
        print(f"Exception in get_location: {str(e)}")
        return 500, {"error": str(e)}

def remove_brackets(text):
    return re.sub(r'\s*\(.*?\)', '', text).strip()


def find_number_after(sequence, text):
    pattern = rf'\b{re.escape(sequence)}(\d+)'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def normalize(text):
    text = text.lower()
    text = text.replace("-", " ").replace(".", " ")
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", text).strip()

def word_after(text, target):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() == target.lower() and i + 1 < len(words):
            return words[i + 1]
    return None  # Not found or no word after

def contains_digit(word):
    return any(char.isdigit() for char in word)

def extract_leading_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else None

def remove_special_characters(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s)

# Number list is (id, number) of the street we have retrieved from the database
def find_best_match(potential_number, number_list):
    try:
        if not potential_number:
            return None, None

        # Normalize
        potential_number = potential_number.lower()
        entries = [(str(num).lower(), id) for (id, num) in number_list]

        # 1. Exact match
        for num, id in entries:
            if num == remove_special_characters(potential_number): # could have commas or some stuff
                return id, num

        # 2. Prefix match (e.g., "3" -> "3a")
        for num, id in entries:
            if num.startswith(potential_number):
                return id, num

        # 3. Closest numeric match
        potential_num_val = extract_leading_number(potential_number)
        if potential_num_val is None:
            return None, None

        numeric_matches = [(id, num, extract_leading_number(num)) for num, id in entries]
        numeric_matches = [(id, num, val) for id, num, val in numeric_matches if val is not None]

        if not numeric_matches:
            return None, None

        closest = min(numeric_matches, key=lambda x: abs(x[2] - potential_num_val))
        return closest[0], closest[1]
    except Exception as e:
        print(f"Exception in find_best_match: {str(e)}")
        return None, None

def sort_key(val):
    number = val[1]
    if number is None:
        return (0, 0, '')
    match = re.match(r"(\d+)([a-zA-Z]*)", number)
    if match:
        num_part = int(match.group(1))
        alpha_part = match.group(2)
        return (1, num_part, alpha_part)
    else:
        return (2, float('inf'), number)

def return_coords_content(id, msg, db: Session):
    try:
        if not id:
            raise Exception(message='ID is not defined!', argument=id)
        result = db\
            .query(Adress.street, Adress.number, Adress.name, Adress.coord_x, Adress.coord_y)\
            .where(Adress.id == id)\
            .first()
        return 202, {
                "address": f"{result[0]} {result[1]}",
                "name": result[2],
                "x": float(result[3]),
                "y": float(result[4]),
                "message": msg
            }
    except Exception as e:
        print(f"Exception in return_coords: {str(e)}")
        return 505, {"error": str(e)}