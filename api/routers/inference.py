from fastapi import WebSocket, WebSocketDisconnect, Depends, APIRouter
from service.users import WebSocketUser
from service import inference
import json
from service.inference import InferenceSession
from service.users import get_current_user_ws

inference_api_router = APIRouter(
    prefix="/api/inference"
)

@inference_api_router.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket, current_user: WebSocketUser):
    await websocket.accept()

    session = InferenceSession()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print("Connection has been cut.")
                break

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