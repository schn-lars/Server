from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from service.location_session import get_location_db
from sqlalchemy.orm import Session
from service import location
from service.requestforms import LocationRequest

location_api_router = APIRouter(
    prefix="/api/location"
)

@location_api_router.get("/city")
def get_city(
        lat: float,
        long: float,
        db: Session = Depends(get_location_db)
    ):
    try:
        closest_city = location.get_city(lat=lat, long=long, db=db)
        if closest_city:
            return JSONResponse(content={"city": closest_city}, status_code=200)
        else:
            return JSONResponse(content={"error": "City not found!"}, status_code=404)
    except Exception as e:
        print(f"Exception in get_city {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@location_api_router.post("/location")
async def get_location(request: LocationRequest, db: Session = Depends(get_location_db)):
    print("Get_location: Raw text:", request.raw_text)
    status_code, content = location.get_location(request=request, db=db)
    return JSONResponse(content=content, status_code=status_code)