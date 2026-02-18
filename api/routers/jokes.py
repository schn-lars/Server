from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import pyjokes

jokes_api_router = APIRouter(
    prefix="/api/jokes"
)

@jokes_api_router.post("/joke")
async def get_joke(
        language: str = Form(...),
        category: str = Form(...)
    ):
    try:
        return JSONResponse(content={"joke": pyjokes.get_joke(language=language, category=category)}, status_code=200)
    except Exception as e:
        print(f"get_joke() error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
