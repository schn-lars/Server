from fastapi import APIRouter
from fastapi.responses import JSONResponse
from service.requestforms import TextInput
import requests

date_api_router = APIRouter(
    prefix="/api/date"
)

@date_api_router.post("/date")
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