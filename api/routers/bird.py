from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from service.location import get_canton
from service.bird import classify_bird, create_bird_plot
from service.bird_session import get_db
from sqlalchemy.orm import Session

bird_api_router = APIRouter(
    prefix="/api/birds"
)

@bird_api_router.post("/classify")
async def classify_bird(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
    ):
    try:
        bird_name = await classify_bird(file, db)
        return {"birdName": bird_name}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@bird_api_router.post("/birdplot")
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