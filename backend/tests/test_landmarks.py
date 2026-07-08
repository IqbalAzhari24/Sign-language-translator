import numpy as np

from app.landmarks import extract_landmarks, MAX_HANDS, NUM_LANDMARKS


class FakePoint:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeHand:
    def __init__(self, seed):
        self.landmark = [FakePoint(seed, seed, seed) for _ in range(NUM_LANDMARKS)]


class FakeResults:
    def __init__(self, hands):
        self.multi_hand_landmarks = hands


class FakeDetector:
    def __init__(self, hands):
        self._hands = hands

    def process(self, image_rgb):
        return FakeResults(self._hands)


def test_extract_landmarks_no_hands_returns_zero_padded():
    detector = FakeDetector(hands=[])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, detector)
    assert hand_count == 0
    assert landmarks.shape == (MAX_HANDS, NUM_LANDMARKS, 3)
    assert np.all(landmarks == 0)


def test_extract_landmarks_one_hand_zero_pads_second():
    detector = FakeDetector(hands=[FakeHand(seed=0.5)])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, detector)
    assert hand_count == 1
    assert np.allclose(landmarks[0], 0.5)
    assert np.all(landmarks[1] == 0)


def test_extract_landmarks_real_mediapipe_blank_frame_has_no_hands():
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=MAX_HANDS)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks, hand_count = extract_landmarks(image, hands)
    hands.close()
    assert hand_count == 0
    assert landmarks.shape == (MAX_HANDS, NUM_LANDMARKS, 3)
