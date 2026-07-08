from pathlib import Path

import numpy as np
import pytest

from app.landmarks import extract_landmarks, create_hand_landmarker, MAX_HANDS, NUM_LANDMARKS, DEFAULT_MODEL_PATH


class FakePoint:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeResults:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class FakeDetector:
    def __init__(self, hand_landmarks):
        self._hand_landmarks = hand_landmarks

    def detect(self, mp_image):
        return FakeResults(self._hand_landmarks)


def _fake_hand(seed):
    # A "hand" from the Tasks API is directly a list of landmark points
    # (unlike the legacy API, which wrapped it in a `.landmark` attribute).
    return [FakePoint(seed, seed, seed) for _ in range(NUM_LANDMARKS)]


def test_extract_landmarks_no_hands_returns_zero_padded():
    detector = FakeDetector(hand_landmarks=[])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, detector)
    assert hand_count == 0
    assert landmarks.shape == (MAX_HANDS, NUM_LANDMARKS, 3)
    assert np.all(landmarks == 0)


def test_extract_landmarks_one_hand_zero_pads_second():
    detector = FakeDetector(hand_landmarks=[_fake_hand(seed=0.5)])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, detector)
    assert hand_count == 1
    assert np.allclose(landmarks[0], 0.5)
    assert np.all(landmarks[1] == 0)


def test_extract_landmarks_real_hand_landmarker_blank_frame_has_no_hands():
    if not Path(DEFAULT_MODEL_PATH).exists():
        pytest.skip("hand_landmarker.task not downloaded — see Task 2 Step 0")

    detector = create_hand_landmarker()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, detector)
    detector.close()
    assert hand_count == 0
    assert landmarks.shape == (MAX_HANDS, NUM_LANDMARKS, 3)
