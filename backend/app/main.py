import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .inference import SessionInference
from .model import SignSequenceClassifier

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


def load_model(checkpoint_path: Path):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train a model first with train/train.py before starting the server."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SignSequenceClassifier(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["classes"]


def create_app(checkpoint_path: Path = CHECKPOINT_PATH) -> FastAPI:
    from .landmarks import create_hand_landmarker

    model, class_names = load_model(checkpoint_path)
    app = FastAPI()

    @app.websocket("/ws/sign-stream")
    async def sign_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            detector = create_hand_landmarker()
        except FileNotFoundError:
            logger.exception("Failed to create hand landmarker")
            await websocket.close(code=1011, reason="Server not ready: missing hand landmarker model")
            return

        session = SessionInference(model=model, hands_detector=detector, class_names=class_names)
        try:
            while True:
                data = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    # Malformed/truncated frame (e.g. a dropped or partial
                    # packet) — skip it rather than letting a decode failure
                    # kill the whole connection. One bad frame over an
                    # inherently lossy live-video transport shouldn't end
                    # the session.
                    continue
                try:
                    result = session.process_frame(frame)
                except Exception:
                    logger.exception("Error processing frame, skipping")
                    continue
                await websocket.send_json(result.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            detector.close()

    return app
