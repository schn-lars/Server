from pathlib import Path
import threading
from fastapi import APIRouter
from typing import Optional
from fastapi.responses import JSONResponse
from service import patchnotes

patchnotes_api_router = APIRouter(
    prefix="/api/patchnotes"
)

@patchnotes_api_router.get("")
@patchnotes_api_router.get("/{commit_hash}")
async def get_patch_notes(commit_hash: Optional[str] = ""):
    try:
        last_hash, commits = patchnotes.get_patch_notes(commit_hash=commit_hash)
        if last_hash == commit_hash:
             return JSONResponse(content={"message": "You are up to date."}, status_code=200)

        if len(commits) == 0 and commit_hash != "":
            return JSONResponse(content={"error": "Commits not found!"}, status_code=404)
        return JSONResponse(content={"commits": commits}, status_code=200)
    except Exception as e:
        print("ERROR in /patchnotes/:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)