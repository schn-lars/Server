from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ReverseRequest(BaseModel):
    apikey: str
    object: str
    id: str

class UploadRequest(BaseModel):
    img: str
    id: str

class PreprocessedInput(BaseModel):
    zip_codes: Optional[List[str]] = []
    streets: Optional[List[str]] = []
    numbers: Optional[List[str]] = []
    cities: Optional[List[str]] = []

class LocationRequest(BaseModel):
    raw_text: str
    preprocessed: Optional[PreprocessedInput] = None

class HourlyMeasurement(BaseModel):
    degrees: float
    millimeters: float
    chance: float
    hours: str

class HourlyRequest(BaseModel):
    slots: List[HourlyMeasurement]
    language: str

class MinutelyMeasurement(BaseModel):
    chance: float
    minute: str

class MinutelyRequest(BaseModel):
    slots: List[MinutelyMeasurement]
    language: str

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

class TextInput(BaseModel):
    text: str

class ShareInformationRequest(BaseModel):
    id: str
    coord_x: float
    coord_y: float
    json: str

class FetchSharedIdsRequest(BaseModel):
    coord_x: float
    coord_y: float