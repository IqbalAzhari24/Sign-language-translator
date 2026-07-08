from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MAX_HANDS = 2
NUM_LANDMARKS = 21
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def create_hand_landmarker(model_path=DEFAULT_MODEL_PATH, num_hands=MAX_HANDS):
    """Build a MediaPipe HandLandmarker (Tasks API) in synchronous IMAGE mode.

    IMAGE mode (stateless, one call per frame) is used both here and by the
    training pipeline so train-time and inference-time landmark extraction
    never diverge.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. Download it first:\n"
            f"curl -L -o {model_path} "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=num_hands,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_bgr, hands_detector):
    """Run a HandLandmarker-compatible detector over one BGR frame.

    Returns (landmarks, hand_count) where landmarks is a
    (MAX_HANDS, NUM_LANDMARKS, 3) float32 array, zero-padded when
    fewer than MAX_HANDS are detected. `hands_detector` must expose a
    `.detect(mp_image)` method returning an object with a
    `.hand_landmarks` attribute: a list of hands, each hand a list of
    landmark-like objects with .x/.y/.z (matches MediaPipe Tasks'
    HandLandmarker, or a test fake with the same shape).
    """
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hands_detector.detect(mp_image)

    landmarks = np.zeros((MAX_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)
    hand_count = 0

    for i, hand in enumerate(results.hand_landmarks[:MAX_HANDS]):
        for j, point in enumerate(hand):
            landmarks[i, j] = (point.x, point.y, point.z)
        hand_count += 1

    return landmarks, hand_count
