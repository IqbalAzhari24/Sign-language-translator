import numpy as np
import torch
import torch.nn as nn

from app.inference import SessionInference
from app.landmarks import NUM_LANDMARKS


class ZeroHandDetector:
    def detect(self, mp_image):
        class R:
            hand_landmarks = []
        return R()


class OneHandDetector:
    def detect(self, mp_image):
        class Point:
            def __init__(self):
                self.x = self.y = self.z = 0.1

        hand = [Point() for _ in range(NUM_LANDMARKS)]

        class R:
            hand_landmarks = [hand]
        return R()


class AlwaysReadyBoundary:
    """Fake boundary detector: every frame immediately completes a 1-frame sequence."""
    def update(self, landmarks):
        return [landmarks]


class NeverReadyBoundary:
    def update(self, landmarks):
        return None


class FixedLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits)

    def forward(self, x):
        return self._logits.unsqueeze(0)


def _blank_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_no_hand_returns_no_hand_status():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),
        hands_detector=ZeroHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "no_hand"


def test_tracking_when_boundary_not_ready():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=NeverReadyBoundary(),
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "tracking"


def test_recognized_when_confidence_high():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),  # softmax -> ~1.0 for class 0
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
        confidence_threshold=0.6,
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "recognized"
    assert result.label == "a"


def test_unsure_when_confidence_low():
    session = SessionInference(
        model=FixedLogitsModel([0.1, 0.0]),  # near 50/50 -> low confidence
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
        confidence_threshold=0.9,
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "unsure"
