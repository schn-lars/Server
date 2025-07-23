from argparse import ArgumentError
from contextlib import asynccontextmanager
from datetime import datetime
import subprocess
from pathlib import Path
import base64
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, BackgroundTasks
import psycopg2
from matplotlib import gridspec
from matplotlib.streamplot import OutOfBounds
from mpmath import degree
from pydantic import constr, BaseModel
import re
import asyncio
from sqlmodel import SQLModel, Field
import io
import numpy as np
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from typing import Optional, List, Dict, Any
import os
from shapely.geometry import Point
from sympy.physics.mechanics import potential_energy
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from PIL import Image
import torch
from EcoNameTranslator import to_species, to_scientific
import threading
import time
import requests
from typing import Optional
import unicodedata
import geopandas as gpd
from math import sqrt

known_species = set() # Those are all the species which are present in our database

db_config = {
    "dbname": os.getenv("POSTGRES_DB", "default_db"),
    "user": os.getenv("POSTGRES_USER", "default_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "default_password"),
    "host": os.getenv("API_HOST", "default_host"),
    "port": os.getenv("DB_PORT", "5432")
}

general_config = {
    "reverse_search_limit": os.getenv("REVERSE_SEARCH_LIMIT", 100),
    "github_key": os.getenv("GITHUB_KEY", "DUMMY_KEY"),
    "api_key": os.getenv("SERP_API_KEY", "DUMMY_KEY")
}

image_deletion_dict = []
deletor_lock = threading.Lock()
DELETE_TIME = 900

def deletor():
    global image_deletion_dict
    global deletor_lock
    time_to_sleep = DELETE_TIME
    while True:
        with deletor_lock:
            if len(image_deletion_dict) == 0:
                time_to_sleep = DELETE_TIME
            else:
                id, timestamp = image_deletion_dict[0]
                if time.time() > timestamp:
                    # remove this item
                    image_deletion_dict.pop(0)
                    remove_image(id)
                    if len(image_deletion_dict) == 0:
                        time_to_sleep = DELETE_TIME
                    else:
                        next_id, next_timestamp = image_deletion_dict[0]
                        time_to_sleep = min(DELETE_TIME, max(next_timestamp - time.time(), 0))
        time.sleep(time_to_sleep)

#Startup method which is being called when you start up application
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting...")
    startup()
    yield
    print("Terminating...")
    clear_images()

app = FastAPI(lifespan=lifespan) # init FastAPI-Application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def startup():
    global refresh_repo
    global gdf
    global image_deletion_dict
    print("Starting with startup...")
    puller_thread = threading.Thread(target=puller, daemon=True)
    puller_thread.start()

    # Clear all images before the re-launch
    clear_images()
    deletor_thread = threading.Thread(target=deletor, daemon=True)
    deletor_thread.start()

    gdf = gpd.read_file(swiss_shp)
    gdf = gdf.to_crs(epsg=4326)

def get_db_conn():
    try:
        connection = psycopg2.connect(**db_config)
        return connection
    except Exception as e:
        print(f"get_db_connection: {e}")
        return None


def refresh_known_species():
    connection = get_db_conn()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT species_name FROM birds;")
    results = cursor.fetchall()
    known_species.update(result[0] for result in results)
    cursor.close()
    connection.close()

def setup_cache():
    connection = get_db_conn()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT id, label FROM labels;")
    labels = cursor.fetchall()
    counter = 0
    batch = []
    query = '''INSERT INTO synonyms (label_id, synonym) VALUES (%s, %s);'''
    print(f"We have {len(labels)} labels retrived.")
    for (id, label) in labels:
        new_label = None
        steps = 0
        while new_label is None and steps < 50:
            new_label = retrieve_known_species(label)
            steps = steps + 1
            if steps == 100 and new_label is None:
                print(f"Did not find label for {label}.")
        counter = counter + 1
        if new_label is not None:
            batch.append((id, new_label))

def retrieve_known_species(label):
    split_label = label.split(" ")
    split_label.append(label)
    split_label = list(set(split_label))
    print(f"Split_label = {split_label}")
    label_synonyms = {}
    for split in split_label:
        try:
            label_synonyms[split] = to_scientific([split])[split][1]
            #print(f"Synonyms for {split}: {label_synonyms[split]}")
        except Exception as e:
            #print(f"Error synonyms extraction: {e}")
            continue
    if not bool(known_species):
        refresh_known_species()
    #print("Sorting keys now")
    sorted_keys = sorted(label_synonyms.keys(), key=lambda k: -k.count(" "))
    for key in sorted_keys:
        for val in label_synonyms[key]:
            if val in known_species:
                return val
    return None


from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from PIL import Image
import torch
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")


# Bird

@app.post("/classify")
async def classify_bird(
        file: UploadFile = File(...)
    ):
    try:
        print("Classify_bird")
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = preprocessor(img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        print(f"Label: {label}")
        normalized_label = label.strip().lower()

        query = '''
            SELECT s.synonym
            FROM labels l
            JOIN synonyms s ON l.id = s.label_id
            WHERE l.label = %s
            LIMIT 1;
        ''' 
        connection = get_db_conn()
        if connection is None:
            raise Exception("Database could not be reached!")
        cursor = connection.cursor()
        cursor.execute(query,(label,))
        result = cursor.fetchall()
        print(result)
        try:
            return JSONResponse(content={"birdName": result[0][0]}, status_code=200)
        except Exception as e:
            print(str(e))
            return JSONResponse(content={"birdName": normalized_label}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/birdplot")
async def get_bird_plot(
        name: str = Form(...),
        latitude: float = Form(...),
        longitude: float = Form(...),
        language: str = Form(...)
    ):
    try:
        location = get_canton(latitude, longitude)
        image_stream = create_bird_plot(location, name, language)
        if image_stream is None:
            return JSONResponse(content={"error": f"The species {name} has never been reported, yet!"}, status_code=404)
        return StreamingResponse(image_stream, media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

swiss_shp = Path("./map/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp")
gdf = None
cantons_gdf = { 'Genève': 'GE', 'Thurgau': 'TG', 'Valais': 'VS', 'Aargau': 'AG', 'Schwyz': 'SZ', 'Zürich': 'ZH', 'Obwalden': 'OW',
                'Fribourg': 'FR', 'Glarus': 'GL', 'Uri' : 'UR', 'Nidwalden' : 'NW', 'Solothurn' : 'SO', 'Appenzell Ausserrhoden' : 'AR',
                'Jura' : 'JU', 'Graubünden' : 'GR', 'Vaud' : 'VD', 'Luzern' : 'LU', 'Ticino' : 'TI', 'Zug' : 'ZG', 'Basel-Landschaft' : 'BL',
                'St. Gallen' : 'SG', 'Schaffhausen' : 'SH', 'Bern' : 'BE', 'Basel-Stadt' : 'BS', 'Neuchâtel': 'NE', 'Appenzell Innerrhoden': 'AI'
                }

def get_canton(lat, lon):
    global gdf
    point = Point(lon, lat)
    match = gdf[gdf.contains(point)]
    if not match.empty:
        return cantons_gdf[match.iloc[0]['NAME']]
    return None


def create_bird_plot(canton, bird_name, language):
    try:
        result = None
        connection = get_db_conn()
        cursor = connection.cursor()
        if canton is None:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            cursor.execute(command, (bird_name,))
            result = cursor.fetchall()
        else:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE canton = %s
                AND species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            cursor.execute(command, (canton, bird_name,))
            result = cursor.fetchall()

        if not result:
            command = '''
                SELECT year_number, SUM(total_count)
                FROM birds_materialized
                WHERE species_name = %s
                GROUP BY year_number
                LIMIT 10;
            '''
            cursor.execute(command, (bird_name,))
            result = cursor.fetchall()
            if result:
                year_to_count = {year: count for year, count in result}
                max_year = max(year_to_count.keys())
                years = list(range(max_year - 9, max_year + 1))
                filled_results = [(year, year_to_count.get(year, 0)) for year in years]
                years, counts = zip(*filled_results)

                fig, ax = plt.subplots()
                ax.plot(years, counts, marker='o')
                ax.set_title(
                    f"Occurrences of {bird_name} in Switzerland" if language == "ENG" else f"Sichtungen von {bird_name} in der Schweiz")
                ax.set_xlabel("Year" if language == "ENG" else "Jahr")
                ax.set_ylabel("Count" if language == "ENG" else "Anzahl")
                fig.autofmt_xdate()

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plt.close(fig)
                return buf
            else:
                print(f"No data found for {bird_name} and {canton}.")
                return None
        else:
            year_to_count = {year: count for year, count in result}
            max_year = max(year_to_count.keys())
            years = list(range(max_year - 9, max_year + 1))
            filled_results = [(year, year_to_count.get(year, 0)) for year in years]
            years, counts = zip(*filled_results)

            fig, ax = plt.subplots()
            ax.plot(years, counts, marker='o')
            ax.set_title(f"Occurrences of {bird_name} in {canton}" if language == "ENG" else f"Sichtungen von {bird_name} in {canton}")
            ax.set_xlabel("Year" if language == "ENG" else "Jahr")
            ax.set_ylabel("Count" if language == "ENG" else "Anzahl")
            fig.autofmt_xdate()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            return buf
    except Exception as e:
        print(f"Error in create_bird_plot: {e}")
        raise

import pyjokes

# Person

@app.post("/joke")
async def get_joke(
        language: str = Form(...),
        category: str = Form(...)
    ):
    try:
        print("get_joke ....")
        return JSONResponse(content={"joke": pyjokes.get_joke(language=language, category=category)}, status_code=200)
    except Exception as e:
        print(f"get_joke() error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# General

LOG_FILE = Path("./commits.txt")
refresh_repo = True
lock = threading.Lock()

def load_existing_logs():
    if LOG_FILE.exists():
        return set(LOG_FILE.read_text().splitlines())
    return set()

def append_new_logs(new_logs, existing_logs):
    existing_shas = set(log.strip().split("|", 2)[1] for log in existing_logs)
    new_ordered_logs = list(reversed(new_logs))
    logs_to_append = [log for log in new_ordered_logs if log['hash'] not in existing_shas]
    if not logs_to_append:
        print("No new logs to append")
        return
    with LOG_FILE.open("a") as f:
        for entry in logs_to_append:
            line = f"{entry['date']}|{entry['hash']}|{entry['message']}"
            f.write(line + "\n")

def puller():
    global refresh_repo
    while True:
        with lock:
            refresh_repo = True
        time.sleep(43200)

def try_pull():
    global refresh_repo
    try:
        with lock:
            if refresh_repo:
                print("Starting auto-pulling...")
                new_logs = fetch_commits_from_github()
                existing_logs = load_existing_logs()
                append_new_logs(new_logs, existing_logs)
                refresh_repo = False
                #print("Finish auto-pull")
    except Exception as e:
        print(f"Error doing auto-pull: {str(e)}")

def fetch_commits_from_github():
    #print("Fetching commits")
    url = f"https://api.github.com/repos/schn-lars/MrIntenso/commits?sha=main&per_page=100"
    headers = {"Accept": "application/vnd.github.v3+json"}
    headers['Authorization'] = f"token {general_config['github_key']}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    commit_logs = []
    for item in response.json():
        date = item['commit']['author']['date']
        dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        formatted_date = dt.strftime("%d.%m.%y")
        sha = item['sha']
        message = item['commit']['message'].replace('\n', ' ')
        commit_logs.append({
            "date": formatted_date,
            "hash": sha,
            "message": message
        })
    return commit_logs


@app.get("/patchnotes/")
@app.get("/patchnotes/{commit_hash}")
async def get_patch_notes(commit_hash: Optional[str] = ""):
    global refresh_repo # good for debugging
    try:
        refresh_repo = True # good for debugging
        try_pull()
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        commits = []
        last_hash = ""
        found = commit_hash == ""
        for i, line in enumerate(lines):
            if not found:
                if f"|{commit_hash}|" in line:
                    found = True
                contents = line.strip().split("|", 2)
                last_hash = contents[1]
                continue
            date, hash, message = line.strip().split("|", 2)
            last_hash = hash
            commits.append({
                "date": date,
                "hash": hash,
                "message": message
            })
        if last_hash == commit_hash:
             return JSONResponse(content={"message": "You are up to date."}, status_code=200)

        if len(commits) == 0 and commit_hash != "":
            return JSONResponse(content={"error": "Commits not found!"}, status_code=404)
 
        return JSONResponse(content={"commits": commits}, status_code=200)
    except Exception as e:
        print("ERROR in /patchnotes/:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# SERP-API
@app.get("/apikey")
async def get_apikey():
    try:
        return JSONResponse(content={"apikey": general_config['api_key']}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

img_folder = Path("images")
img_folder.mkdir(parents=True, exist_ok=True)

class ReverseRequest(BaseModel):
    apikey: str
    object: str
    id: str

@app.get("/images/{uuid}.jpg")
@app.head("/images/{uuid}.jpg")
async def get_image(uuid: str, request: Request):
    print(f"GET image {uuid}.jpg from {request.client.host} or {request.headers}")

    try:
        file_path = img_folder / f"{uuid}.jpg"
        if file_path.exists():
            response = FileResponse(file_path, media_type="image/jpeg")
            response.headers["Content-Type"] = "image/jpeg"
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Cache-Control"] = "public, max-age=900, immutable"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Strict-Transport-Security"] = "max-age=300"
            response.headers["Accept-Ranges"] = "bytes"
            return response
        return JSONResponse(content={"message": "Image not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

class UploadRequest(BaseModel):
    img: str
    id: str

@app.get("/test")
async def get_test():
    response = {
        "products_page_token": "8P7KinicTZBJsqNGFEXDO_ECrLJEIyAq...",
        "serpapi_products_link": "https://serpapi.com/search.json?...",
        "serpapi_exact_matches_link": "https://serpapi.com/search.json?...",
        "object": "laptop",
        "visual_matches": [
            {
                "image": "http://c.tutti.ch/big/0614252591.jpg",
                "image_height": 430,
                "image_width": 768,
                "link": "https://www.tutti.ch/de/vi/st-gallen/computer-zubehoer/computer/macbook-pro-13-2019-tb-qc-i5-2-4ghz-16gb-512gb/71370093",
                "position": 1,
                "source": "tutti.ch",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d035c8d1e308f76c597a9517f041014bd3d2d7b4df3979a003be.png",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOhoNO0hQ...",
                "thumbnail_height": 168,
                "thumbnail_width": 300,
                "title": "MacBook Pro 13\" 2019 TB QC i5 2,4GHz 16GB 512GB im Kanton St. Gallen - tutti.ch"
            },
            {
                "image": "https://preview.redd.it/my-macbook-pro-comes-up-with-an-error-when-i-try-to-connect-v0-e8pcf8fdnqne1.jpeg?width=640&crop=smart&auto=webp&s=3325683f1c1bbbf0dd726efa05215eccae0bc029",
                "image_height": 640,
                "image_width": 853,
                "link": "https://www.reddit.com/r/applehelp/comments/1j7j844/...",
                "position": 2,
                "source": "Reddit",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d035d46dc2fbee449e59e3f78dc5977f90f9fe12be5c3ea459e0.png",
                "thumbnail": "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcR1iKSbi7-nNrgaAvKcVS0VSbWjgb3bFHs_K2cOVFHGyoindGgL",
                "thumbnail_height": 194,
                "thumbnail_width": 259,
                "title": "My MacBook Pro comes up with an error when I try to connect to WiFi : r/applehelp"
            },
            {
                "image": "https://external-preview.redd.it/flashing-pink-before-turning-off-v0-NzY1eWtqczhwcjViMf3lzKkurh-TXBuL8fkUC-9sMG-2nsMiIvElkNOfel9D.png?width=640&crop=smart&format=pjpg&auto=webp&s=ae3dff2f8133da7619a672d472361d3d5c3b0451",
                "image_height": 640,
                "image_width": 1137,
                "link": "https://www.reddit.com/r/macbookrepair/comments/148csy0/...",
                "position": 3,
                "source": "Reddit",
                "source_icon": "https://serpapi.com/searches/6842e3a603b1360912090d62/images/c5e251cd5315d0354aee44d98390809bdf8d0c1325c590c9903220c414a0b0b9.png",
                "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQRJ7kmVaDtbffyz04nMHLLS_Nd1hDi2JBT2gGlfp-txz0ADsqn",
                "thumbnail_height": 168,
                "thumbnail_width": 299,
                "title": "Flashing Pink Before Turning Off : r/macbookrepair"
            },
            {
                "condition": "Used",
                "image": "https://i.ebayimg.com/images/g/AnQAAOSw1OllqaL4/s-l1200.jpg",
                "image_height": 900,
                "image_width": 1200,
                "link": "https://www.ebay.com/itm/315105573570",
                "position": 18,
                "price": {
                    "currency": "$",
                    "extracted_value": 300,
                    "value": "$300*"
                },
                "rating": "3.9",
                "reviews": 1920,
                "source": "eBay",
                "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn:ANd9GcT5NXsW5qpQoNKtvnEC0sNL88H54opWmBBYIh2gQ3U_SGUU-yc8xV_BfeECVq4HYfwroQsx3k4lpMvjDByZM4KvONyK63j7aI6RPQrDFwRa9lo",
                "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxi-QaJcOVLnezkhk4rSCD4gz52Ze42__klJQTCEa26j3gdiK-",
                "thumbnail_height": 194,
                "thumbnail_width": 259,
                "title": "Apple Macbook Pro 13” (256GB SSD, Intel Core i5 Dual-Core, 3.1GHz, 8GB RAM) 2017 | eBay"
            },
            {
                "image": "https://pisces.bbystatic.com/image2/BestBuy_US/ugc/photos/thumbnail/7500865a3d202ecc6ae59d9f10262d9d.jpg;maxHeight=256;maxWidth=256?format=webp",
                "image_height": 192,
                "image_width": 256,
                "in_stock": 0,
                "link": "https://www.bestbuy.com/site/...",
                "position": 56,
                "price": {
                    "currency": "$",
                    "extracted_value": 670,
                    "value": "$670*"
                },
                "rating": "4.8",
                "reviews": 23286,
                "source": "Best Buy",
                "source_icon": "https://encrypted-tbn1.gstatic.com/favicon-tbn?q=tbn:ANd9GcQRElpIn0_ngPFmOJ5ZdW65M7G4nuI9MycoEeMitUDh35QpZN3krTU2tf03QRaenlaJn4hTIdv1gBd0yWlFCTuep64IeIo7wzb0RKajGmDwhoSbHbc",
                "thumbnail": "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcR3yQPwHtzZZNcFKmzEaEEeqUMtupmwG8MCsCtB74BkrkWRyBE2",
                "thumbnail_height": 192,
                "thumbnail_width": 256,
                "title": "Geek Squad Certified Refurbished MacBook Air 13.6\" Laptop Apple M2 chip 8GB Memory 256GB SSD Midnight GSRF MLY33LL/A - Best Buy"
            }
        ],
        "search_metadata": {
            "created_at": "2025-06-06 12:48:38 UTC",
            "google_lens_url": "https://lens.google.com/uploadbyurl?url=...",
            "id": "6842e3a603b1360912090d62",
            "json_endpoint": "https://serpapi.com/searches/3943d6e05970c134/6842e3a603b1360912090d62.json",
            "processed_at": "2025-06-06 12:48:38 UTC",
            "raw_html_file": "https://serpapi.com/searches/3943d6e05970c134/6842e3a603b1360912090d62.html",
            "status": "Success",
            "total_time_taken": "6.91"
        },
        "visual_matches_page_token": "NmO4anicTZBJsqNGFEXDO_...",
        "related_content": [
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=Apple+MacBook+Air+13-inch+Apple+M1+Chip+7-core+GPU+16GB+256GB+Space+Grey&kgmid=/g/11mw8j71m4&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAB6BQjtAhAB",
                "query" : "Apple MacBook Air 13-inch Apple M1 Chip 7-core GPU 16GB 256GB Space Grey",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fg%2F11mw8j71m4&q=Apple+MacBook+Air+13-inch+Apple+M1+Chip+7-core+GPU+16GB+256GB+Space+Grey",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe534b8f6aa3a86a3971fd702cd7f178ddc0.jpeg"
            },
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=Apple+MacBook+Pro&kgmid=/m/09tzfp&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAF6BQjtAhAD",
                "query" : "Apple MacBook Pro",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fm%2F09tzfp&q=Apple+MacBook+Pro",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe53baad5b23c6502719ec11b3ffd6b4d134.jpeg",
            },
            {
                "link" : "https://www.google.com/search?sca_esv=87b41ab4477c98ab&lns_surface=26&hl=en&q=MacBook+Pro&kgmid=/g/11t7nmnv1m&sa=X&ved=2ahUKEwi244u06tyNAxUhiP0HHbARA28Q9_gLKAJ6BQjtAhAF",
                "query" : "MacBook Pro",
                "serpapi_link" : "https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&hl=en&kgmid=%2Fg%2F11t7nmnv1m&q=MacBook+Pro",
                "thumbnail" : "https://serpapi.com/searches/6842e3a603b1360912090d62/images/5e7b1f0a649ea9f9a3cff9b018b4fe5317cb68559d45e152a1458cdcbd521002.jpeg"
            }
        ]
    }
    return response

@app.post("/upload")
async def upload_image(
        request: UploadRequest
    ):
    try:
        global image_deletion_dict
        global deletor_lock
        global DELETE_TIME
        image_data = base64.b64decode(request.img)
        filename = f"{request.id}.jpg"
        file_path = img_folder / filename
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.save(file_path, "JPEG")
        with deletor_lock:
            image_deletion_dict.append((request.id, time.time() + DELETE_TIME))
        return JSONResponse(content={"url": f"https://myurl.com/images/{request.id}.jpg"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def remove_image(id):
    file_path = img_folder / f"{id}.jpg"
    if file_path.exists():
        os.remove(file_path)
        jpg_count = sum(1 for f in img_folder.glob("*.jpg"))
        print(f"Removed image {id}. Now we have a count of {jpg_count} .jpg files")

def clear_images():
    for file in img_folder.iterdir():
        os.remove(file)

def delete_temp_file(path: Path, delay: int = 10):
    time.sleep(delay)
    try:
        if path.exists():
            os.remove(path)
            print(f"File deleted: {path}")
    except Exception as e:
        print(f"Error deleting file: {e}")
# DEBUGGING PURPOSES
@app.post("/log")
async def log(request: Request):
    body = await request.body()
    with open("log.txt", "a") as f:
        date = datetime.now().strftime("%d.%m.%y %H:%M:%S")
        f.write(f"{date}|{body.decode('utf-8')}" + "\n")
    return JSONResponse(content={}, status_code=200)

#
#
#  POSTERS
#
#

location_db_config = {
    "dbname": os.getenv("LOCATION_POSTGRES_DB", "default_db"),
    "user": os.getenv("POSTGRES_USER", "default_user"),
    "password": os.getenv("LOCATION_POSTGRES_PASSWORD", "default_password"),
    "host": os.getenv("LOCATION_API_HOST", "default_host"),
    "port": os.getenv("LOCATION_DB_PORT", "5432")
}

class PreprocessedInput(BaseModel):
    zip_codes: Optional[List[str]] = []
    streets: Optional[List[str]] = []
    numbers: Optional[List[str]] = []
    cities: Optional[List[str]] = []

class LocationRequest(BaseModel):
    raw_text: str
    preprocessed: Optional[PreprocessedInput] = None

@app.post("/shopping")
async def get_shopping_items(
        picture: UploadFile = File(...)
    ):
    try:
        print("Starting with shopping")
        response = {
            "shopping_results": [
            {
              "position": 1,
              "title": "Apple - MacBook Pro 14' Laptop - M3 chip - 8GB Memory - 10-core GPU - 512GB SSD ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-chip-8gb-memory-10-core-gpu-512gb-ssd-latest-model-space-gray/6534641.p?skuId=6534641&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:6210135998181032295",
              "product_id": "6210135998181032295",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=6210135998181032295",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$1,449.00",
              "extracted_price": 1449.0,
              "old_price": "$1,599.00",
              "extracted_old_price": 1599.0,
              "rating": 4.8,
              "reviews": 295,
              "extensions": [
                "Mac OS",
                "Octa Core",
                "USB-C",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQSBzh7sbe1Ya52-vjINxImUABAV6GkZiP1hRjLqtzbuzjrkMqCeOv_fePvs-2-wblGqi2V_Qlr&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/tyWlMHicu9mdUVJSUGylr5-al1xUWVCSmqJbkpRnrJdeXJJYkpmsl5yfq1-ckV9QkJmXbl9oC5SzcvRLsXRPDgx2qsowL05KNYxMNDXSLcvy9KvwzA11dHIMM3PPjsoMMMwIyvIpLKlKKq3KKsr2LXRO9S-LT0sNKCvWNdItT8pxL8w0CosPzClSKy0uLLB1dnQFAHuuMu4",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            },
            {
              "position": 2,
              "title": "Apple - MacBook Pro 14' Laptop - M3 chip - 8GB Memory - 10-core GPU - 512GB SSD ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-chip-8gb-memory-10-core-gpu-512gb-ssd-latest-model-silver/6534640.p?skuId=6534640&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:3059131534171723531",
              "product_id": "3059131534171723531",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=3059131534171723531",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$1,449.00",
              "extracted_price": 1449.0,
              "old_price": "$1,599.00",
              "extracted_old_price": 1599.0,
              "rating": 4.8,
              "reviews": 69,
              "extensions": [
                "Mac OS",
                "Octa Core",
                "USB-C",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcTopcSSvTfIe-JheQn8HrcAkgHN5yH_toz8fnJudGBf1v3N9cy1i-1tlPqYzeyISyBVdMnnEYI&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/gtpsuXicDcndCoIwFADgt-lOZUSggoSG-ANJoQReRZ1tKtbZdEdhPkJv2NvUd_t9Pz2RNqHnCYTZahLcoScytzP0oAFcUG_P9ErrAbvjFP0vjCseZNAoDXW9NrIQTtmLK_r5DPHY5dXB5ndSmy-xXHiWSLbuqwAsGxxGr8vUbsIWtU1u_IyYtsVuMZOOTnH6A17VMok",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            },
            {
              "position": 3,
              "title": "Apple - MacBook Pro 14' Laptop - M3 Pro chip - 18GB Memory - 18-core GPU - 1TB ...",
              "link": "https://www.bestbuy.com/site/apple-macbook-pro-14-laptop-m3-pro-chip-18gb-memory-18-core-gpu-1tb-ssd-latest-model-space-black/6534624.p?skuId=6534624&utm_source=feed",
              "product_link": "https://www.google.com/shopping/product/1?gl=us&prds=pid:3713303853686963414",
              "product_id": "3713303853686963414",
              "serpapi_product_api": "https://serpapi.com/search.json?device=desktop&engine=google_product&gl=us&google_domain=google.com&hl=en&product_id=3713303853686963414",
              "source": "Best Buy",
              "source_icon": "https://encrypted-tbn0.gstatic.com/favicon-tbn?q=tbn%3AANd9GcRJLrYt8ApvztGsW8TSy6-5HL7LwDNwH2emYmRabMUepMDXWE3LqD_Jltucg6NfE5z5MV57q9G1n_VVMyiUtZCVGXOuFlVA6g",
              "price": "$2,199.00",
              "extracted_price": 2199.0,
              "old_price": "$2,399.00",
              "extracted_old_price": 2399.0,
              "rating": 4.9,
              "reviews": 3813,
              "extensions": [
                "Mac OS",
                "USB-C",
                "HDMI",
                "120Hz",
                "SALE"
              ],
              "badge": "Top Quality Store",
              "thumbnail": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcSfcqIJ2kTmhAF2t_Pxs9HR2aPI6599y2wNrkwdi4vgnHSdUYuLGg7uBrFily2cibeqpBampas&usqp=CAE",
              "serpapi_thumbnail": "https://serpapi.com/images/url/191f4HicDclLDoIwFADA27gTTOMnJSEGjHyMIUR04YqUFkuDlFf6ELmCN_Q2Otv5fhpEsJ7r1poPM2AtlljplSMtMlTc4X3n2qYHUFrujf8_L8gEjXnx4CY9kfbaNUFEsMzfliYXwvJ0u6F0JlM2tJNQ65fUSSFu9_Ecy90YDpF6zoSrqjYQsg6YXYzWgH8Ijj9AhTKY",
              "tag": "SALE",
              "delivery": "Free delivery by Feb 13 & Free 15-day returns",
              "store_rating": 4.6,
              "store_reviews": 358
            }
          ]
        }

        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        print(f"Error in get_shopping_items: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/city")
def get_city(
        lat: float,
        long: float
    ):
    try:
        lat = float(lat)
        long = float(long)
        lower_lat = lat - float(0.01)
        upper_lat = lat + float(0.01) # approx 1km
        lower_long = long - float(0.005)
        upper_long = long + float(0.005)
        query = '''
            SELECT coord_x, coord_y, name
            FROM adresses a
            WHERE a.coord_x BETWEEN %s AND %s
            AND a.coord_y BETWEEN %s AND %s;
        '''
        connection = psycopg2.connect(**location_db_config)
        cursor = connection.cursor()

        cursor.execute(query, (lower_lat, upper_lat, lower_long, upper_long,))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        closest_city = None
        min_distance = float('inf')

        for coord_x, coord_y, name in results:
            distance = sqrt((float(coord_x) - lat) ** 2 + (float(coord_y) - long) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_city = name
        if closest_city:
            return JSONResponse(content={"city": closest_city}, status_code=200)
        else:
            return JSONResponse(content={"error": "City not found!"}, status_code=404)
    except Exception as e:
        print(f"Exception in get_city {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

class HourlyMeasurement(BaseModel):
    degrees: float
    millimeters: float
    chance: float
    hours: str

class HourlyRequest(BaseModel):
    slots: List[HourlyMeasurement]
    language: str

@app.post("/hourlyweather")
async def get_hourly_weather(
        request: HourlyRequest
    ):
    try:
        lang = request.language
        temperatures = [hour.degrees for hour in request.slots]
        rain = [hour.millimeters for hour in request.slots]
        rain_chance = [hour.chance for hour in request.slots]
        hours = [hour.hours for hour in request.slots]
        positions = list(range(len(hours)))

        fig, ax = plt.subplots(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.05, 0.95], wspace=0.05)
        cbar_ax = fig.add_axes([0.05, 0.15, 0.03, 0.7]) # color is the 3rd
        #cbar_ax = fig.add_subplot(gs[0])
        #ax = fig.add_subplot(gs[1])
        fig.subplots_adjust(left=0.2)
        ax.bar(positions, rain, color='lightskyblue', label='Precipitation' if lang == 'ENG' else 'Niederschlag')

        degree_ax = ax.twinx()
        degree_ax.plot(positions, temperatures, '-r', label='Temperature' if lang == 'ENG' else 'Temperatur')

        ax.set_xticks(positions)
        ax.set_xticklabels(hours)

        max_rain = max(rain)
        y_max = 10 if max_rain <= 9 else max_rain * 1.1
        ax.set_ylim(0, y_max)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap("Blues")

        for pos, chance in zip(positions, rain_chance):
            color = cmap(chance)
            ax.add_patch(plt.Rectangle(
                (pos - 0.4, y_max + 0.2),
                0.8,
                0.3,
                color=color,
                transform=ax.transData,
                clip_on=False
            ))

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Rain Chance (%)" if lang == "ENG" else "Regen Wahrscheinlichkeit (%)", rotation=270)
        cbar.set_ticks(np.linspace(0, 1, 11))
        cbar.set_ticklabels([f"{int(p * 100)}%" for p in np.linspace(0, 1, 11)])

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = degree_ax.get_legend_handles_labels()

        smallest_temp_index = temperatures.index(min(temperatures))
        if 0 <= smallest_temp_index < 9:
            ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        elif 9 <= smallest_temp_index < 17:
            ax.legend(handles1 + handles2, labels1 + labels2, loc='upper center')
        else:
            ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        ax.grid()

        ax.set_xlabel('Hours' if lang == "ENG" else "Stunden") # or better the actual day "Di, DD.MM."
        ax.set_ylabel("mm")
        degree_ax.set_ylabel("°C")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Exception in get_hourly_weather: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

class MinutelyMeasurement(BaseModel):
    chance: float
    minute: str

class MinutelyRequest(BaseModel):
    slots: List[MinutelyMeasurement]
    language: str
@app.post("/minutelyweather")
async def get_minutely_weather(
        request: MinutelyRequest
    ):
    try:
        lang = request.language
        rain_chance = [slot.chance for slot in request.slots]
        minutes = [slot.minute for slot in request.slots]

        fig, ax = plt.subplots()
        ax.plot(minutes, rain_chance, marker='o')
        ax.set_xlabel("Minutes" if lang == "ENG" else "Minuten")
        ax.set_ylabel("Rain Chance (%)" if lang == "ENG" else "Regen Wahrscheinlichkeit (%)")
        ax.set_xticks(minutes[::5])
        ax.set_ylim(0, 1)
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Exception in get_minutely_weather: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

def remove_brackets(text):
    return re.sub(r'\s*\(.*?\)', '', text).strip()


def find_number_after(sequence, text):
    pattern = rf'\b{re.escape(sequence)}(\d+)'
    match = re.search(pattern, text)
    return match.group(1) if match else None
@app.post("/location")
async def get_location(request: LocationRequest):
    print("Get_location: Raw text:", request.raw_text)
    zips = request.preprocessed.zip_codes if request.preprocessed else None
    streets = request.preprocessed.streets if request.preprocessed else None
    try:
        connection = psycopg2.connect(**location_db_config)
        cursor = connection.cursor()
        #return return_coords(2892956, msg="Testing")
        raw = request.raw_text.lower()
        if zips: # I think this should work even for the case where we have multiple zips. We are comparing them wtih cities and raw_text. This should hold
            print(zips)
            zips_query = ','.join(['%s'] * len(zips))
            # decide which zip fits. I assume that everytime a zip is on a poster, I have city somewhere too. (Nobody advertises by "Come join us in 4056" wtf)
            zip_query = f"SELECT zip, name FROM adresses WHERE zip IN ({zips_query});"
            zip_list = list(zips)
            cursor.execute(zip_query, zip_list)
            zip_and_cities = cursor.fetchall()
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
                streets_in_zip_query = f"SELECT street FROM adresses WHERE zip = '{matching[0][0]}' AND name = '{matching[0][1]}';"
                cursor.execute(streets_in_zip_query)
                streets_in_zip = cursor.fetchall()
                zip_street = None # determine the street
                normalized_raw = normalize(raw)
                for (street,) in streets_in_zip:
                    norm_street = normalize(street)
                    if norm_street in normalized_raw:
                        zip_street = street
                        break
                if zip_street is not None:
                    print(f"Processing zip_street: {zip_street}")
                    number_query = f"SELECT id,  number FROM adresses WHERE zip = '{matching[0][0]}' AND name = '{matching[0][1]}' AND street = '{zip_street}';"
                    cursor.execute(number_query)
                    number_results = cursor.fetchall()
                    potential_number = None
                    if " " in zip_street:
                        potential_number = find_number_after(zip_street.lower().replace(' ', ''), normalized_raw.replace(' ', ''))
                    else:
                        potential_number = word_after(raw, zip_street)
                    if potential_number and contains_digit(potential_number):
                        # Get closest number
                        closest_number_id, closest_number = find_best_match(potential_number, number_results)
                        if closest_number == potential_number:
                            return return_coords(closest_number_id, f"This is the location of {zip_street} {closest_number}")
                        else:
                            return return_coords(closest_number_id, f"This is the next closest known location: {zip_street} {closest_number}")
                    else:
                        print("Potential_number is None")
                        # return default of this or closest
                        id_and_lowest_number = sorted(number_results, key=sort_key)
                        return return_coords(id_and_lowest_number[0][0], f"Location originating from zip {zip_street} without potential number.")
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
            placeholders = ','.join(['%s'] * len(streets))
            print(f"Streets: {streets}")
            street_query = f"SELECT name, street, normalized_street FROM adresses WHERE normalized_street IN ({placeholders});"
            cursor.execute(street_query, streets)
            name_street = cursor.fetchall()

            if len(name_street) == 0:
                print("No street was found.")
                return JSONResponse(content={"error": "No streets found"}, status_code=404)

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
            number_query = f"SELECT id,  number FROM adresses WHERE name = '{main_city}' AND street = '{main_street}';"
            cursor.execute(number_query)
            number_results = cursor.fetchall()
            if potential_number and contains_digit(potential_number):
                closest_number_id, _ = find_best_match(potential_number, number_results)
                if closest_number_id:
                    return return_coords(closest_number_id, "Location retrieved by using addresses as startpoint.")
            else:
                # We do not have a number, therefore we want the lowest number there is an entry for
                id_and_lowest_number = sorted(number_results, key=sort_key)
                return return_coords(id_and_lowest_number[0][0], "Location retrieved by using addresses as startpoint without potential number.")

        cursor.close()
        connection.close()
        return JSONResponse(content={"error": "No location has been found!"}, status_code=404)
    except Exception as e:
        print(f"Exception in get_location: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except IndexError as e:
        print(f"Exception in get_location: {str(e)}")
        return JSONResponse(content={"error": "No location has been found!"}, status_code=404)
    except KeyError as e:
        print(f"KeyError in get_location: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# This method returns the JSON object which will be sent back to the client.
# It needs an ID
def return_coords(id, msg):
    try:
        if not id:
            raise ArgumentError(message='ID is not defined!', argument=id)
        connection = psycopg2.connect(**location_db_config)
        cursor = connection.cursor()
        cursor.execute(f"SELECT street, number, name, coord_x, coord_y FROM adresses WHERE id = {id};")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return JSONResponse(
            content={
                "address": f"{result[0]} {result[1]}",
                "name": result[2],
                "x": float(result[3]),
                "y": float(result[4]),
                "message": msg
            },
            status_code=200
        )
    except Exception as e:
        print(f"Exception in return_coords: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

#
#   SHARE
#

shared_items = {}
share_lock_holmes = threading.Lock()

class UUIDPayload(BaseModel):
    uuid: str


class LocationObject(BaseModel):
    lon: float
    lat: float
    city: str
    address: str

class ImageObject(BaseModel):
    title: str
    img: str

class WebObject(BaseModel):
    url: str
    title: str

class SpecificationObject(BaseModel):
    title: str
    spec: str
    desc: str

class Item(BaseModel):
    location: Optional[LocationObject] = None
    image: Optional[ImageObject] = None
    web: Optional[WebObject] = None
    specification: Optional[SpecificationObject] = None

class ShareData(BaseModel):
    id: str
    object: str
    confidence: float
    lat: float
    lon: float
    lastSpotted: int
    img: str
    items: Optional[Dict[str, Dict[str, Any]]] = None

@app.post("/fetch", response_model=ShareData)
async def fetch(
        payload: UUIDPayload
    ):
    try:
        with share_lock_holmes:
            info = shared_items[payload.uuid]
            return info
    except KeyError:
        return JSONResponse(content={"error": "The given ID is not being shared."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/share")
async def share(
        request: Request
    ):
    try:
        body = await request.body()
        data = ShareData.model_validate_json(body)  # now parse explicitly
        print(f"Upload for object: {data.id}")
        with share_lock_holmes:
            shared_items[data.id] = data
            print(f"Current state of shared infos: {shared_items.keys()}")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/exitshare")
async def exitshare(
        payload: UUIDPayload
    ):
    try:
        print(f"Exit share for object: {payload.uuid}")
        with share_lock_holmes:
            shared_items.pop(payload.uuid)
            print(f"Current state of shared infos: {shared_items.keys()}")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#
#   DATE EXTRACTION
#

class TextInput(BaseModel):
    text: str

@app.post("/date")
async def get_date(data: TextInput):
    try:
        print(f"Get_date {data.text}")
        response = requests.post("http://spark:5050/extractdates", json={"text": data.text})
        if response.status_code != 200:
            return JSONResponse(content={"error": str(response.json()["error"])}, status_code=response.status_code)
        else:
            print(f"Returning date {response.json()['dates']}")
            return JSONResponse(content={"dates": response.json()["dates"][0]}, status_code=response.status_code)
    except Exception as e:
        print(f"Exception in get_date {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

#
# HELP
#
@app.get("/help")
async def get_help(
        language: str = "ENG"
    ):
    try:
        return FileResponse(path="./Mr__Intenso__How_To.pdf" if language == "ENG" else "Mr__Intenso__Hilfe.pdf", status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)