from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from service.bird_session import Base, engine
from service.resources import clear_images, deletor
from service.patchnotes import puller
from routers.bird import bird_api_router
from routers.date import date_api_router
from routers.jokes import jokes_api_router
from routers.location import location_api_router
from routers.patchnotes import patchnotes_api_router
from routers.resources import resources_api_router
from routers.share import share_api_router
from routers.shopping import shopping_api_router
from routers.weather import weather_api_router
from routers.utils import utils_api_router

#Startup method which is being called when you start up application
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting...")
    startup()
    yield
    print("Terminating...")
    clear_images()

app = FastAPI(lifespan=lifespan) # init FastAPI-Application
app.include_router(bird_api_router)
app.include_router(date_api_router)
app.include_router(jokes_api_router)
app.include_router(location_api_router)
app.include_router(patchnotes_api_router)
app.include_router(resources_api_router)
app.include_router(share_api_router)
app.include_router(shopping_api_router)
app.include_router(weather_api_router)
app.include_router(utils_api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)

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
