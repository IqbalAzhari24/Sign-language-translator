import numpy as np

MAX_HANDS = 2
NUM_LANDMARKS = 21


def extract_landmarks(image_bgr, hands_detector):
    """Run a MediaPipe-Hands-compatible detector over one BGR frame.

    Returns (landmarks, hand_count) where landmarks is a
    (MAX_HANDS, NUM_LANDMARKS, 3) float32 array, zero-padded when
    fewer than MAX_HANDS are detected. `hands_detector` must expose
    a `.process(image_rgb)` method returning an object with a
    `.multi_hand_landmarks` attribute (real MediaPipe Hands, or a
    test fake with the same shape).
    """
    image_rgb = image_bgr[:, :, ::-1]
    results = hands_detector.process(image_rgb)
    landmarks = np.zeros((MAX_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)
    hand_count = 0

    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            for j, point in enumerate(hand.landmark):
                landmarks[i, j] = (point.x, point.y, point.z)
            hand_count += 1

    return landmarks, hand_count
