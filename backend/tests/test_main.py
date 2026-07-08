import numpy as np
import cv2
import torch
from fastapi.testclient import TestClient

from app.main import create_app, load_model
from app.model import SignSequenceClassifier


def _save_dummy_checkpoint(path, class_names=("a", "b")):
    model = SignSequenceClassifier(num_classes=len(class_names))
    torch.save({"model_state": model.state_dict(), "classes": list(class_names)}, path)


def test_load_model_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "nope.pt"
    try:
        load_model(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_websocket_blank_frame_returns_no_hand(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    _save_dummy_checkpoint(checkpoint_path)
    app = create_app(checkpoint_path=checkpoint_path)
    client = TestClient(app)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok

    with client.websocket_connect("/ws/sign-stream") as ws:
        ws.send_bytes(encoded.tobytes())
        response = ws.receive_json()

    assert response["status"] == "no_hand"
