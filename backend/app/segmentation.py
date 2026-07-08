import numpy as np


class SignBoundaryDetector:
    """Buffers landmark frames and signals when a sign has finished.

    Heuristic: wait for motion (hand actively moving), then wait for
    `still_frames_required` consecutive near-still frames. That
    transition (moved -> settled) is treated as the end of one sign.
    """

    def __init__(self, motion_threshold=0.02, still_frames_required=5):
        self.motion_threshold = motion_threshold
        self.still_frames_required = still_frames_required
        self.buffer = []
        self.moved = False
        self.still_count = 0

    def update(self, landmarks: np.ndarray):
        """Feed one frame's landmarks.

        Returns the buffered frame sequence when a sign boundary is
        detected, else None.
        """
        self.buffer.append(landmarks)

        if len(self.buffer) < 2:
            return None

        motion = float(np.linalg.norm(self.buffer[-1] - self.buffer[-2]))

        if motion > self.motion_threshold:
            self.moved = True
            self.still_count = 0
        else:
            self.still_count += 1

        if self.moved and self.still_count >= self.still_frames_required:
            sequence = self.buffer.copy()
            self.buffer = []
            self.moved = False
            self.still_count = 0
            return sequence

        return None
