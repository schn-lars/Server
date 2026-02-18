from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from service.requestforms import HourlyRequest, MinutelyRequest
from service import weather

weather_api_router = APIRouter(
    prefix="/api/weather"
)

@weather_api_router.post("/hourlyweather")
async def get_hourly_weather(
        request: HourlyRequest
    ):
    try:
        buf = weather.get_hourly_weather(request=request)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Exception in get_hourly_weather: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@weather_api_router.post("/minutelyweather")
async def get_minutely_weather(
        request: MinutelyRequest
    ):
    try:
        buf = weather.get_minutely_weather(request=request)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"Exception in get_minutely_weather: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)