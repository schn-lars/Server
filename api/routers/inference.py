from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from users import CurrentUser
from service import inference
import json
import uuid
from inference import InferenceSession

inference_api_router = APIRouter(
    prefix="/api/inference"
)


@inference_api_router.websocket("/ws/inference")
async def inference_ws(current_user: CurrentUser, websocket: WebSocket):
    await websocket.accept()

    session = InferenceSession()

    try:
        while True:
            message = await websocket.receive()

            # upon receival of text, this is mostly yused for controls and settings of the current session
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "init":
                    session.load_model(data["model"])
                    await websocket.send_json({"status": "model_loaded"})

                elif msg_type == "set_prompt":
                    # Make sure, that 'prompt' is already a list of the current prompts we are using!
                    session.prompt = data.get("prompt", inference.SAM3_DEFAULT_PROMPT)
                    await websocket.send_json({"status": "prompt_updated"})

                elif msg_type == "switch_model":
                    session.load_model(data["model"])
                    await websocket.send_json({"status": "model_switched"})

                elif msg_type == "start_stream":
                    session.streaming = True
                    await websocket.send_json({"status": "streaming_started"})

            # BYTES -> used for frames which we run inference on. Fastr like that
            elif "bytes" in message:
                if not session.model_type:
                    print("WebSocket: Inference ERROR - model_type is false")
                    continue  # or error

                frame_bytes = message["bytes"]

                results = session.predict(frame_bytes)

                await websocket.send_json(results)
    except WebSocketDisconnect:
        print("Client disconnected")

    finally:
        print("Cleaning session")
        del session