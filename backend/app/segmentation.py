import numpy as np


class SignBoundaryDetector:
    """Buffers landmark frames and signals when a sign has finished.

    Heuristic: wait for motion (hand actively moving), then wait for
    `still_frames_required` consecutive near-still frames. That
    transition (moved -> settled) is treated as the end of one sign.

    Two-phase motion tracking:
      - Before "moved" is armed, each frame is compared against a fixed
        baseline (the last frame seen while still at rest), and arming
        requires `motion_debounce_frames` *consecutive* frames away from
        that baseline. Comparing against a fixed anchor (rather than the
        immediately previous frame) means a lone jitter frame that
        bounces away and immediately back doesn't count as two separate
        motion events — the return-to-rest frame reads as zero motion
        against the same baseline, so debounce isn't defeated by noise
        that reverts on its own.
      - Once "moved" is armed, stillness is tracked frame-to-frame (as
        before) — a real sign's motion is expected to keep moving
        relative to its own immediately preceding frame.

    Motion is the max single-coordinate displacement across the hand
    slots present in both frames being compared (not a full-vector L2
    norm), so the threshold's meaning doesn't shift with hand count, and
    `min_segment_frames` stops a short, only-technically-valid motion
    blip from emitting a near-empty segment.
    """

    def __init__(
        self,
        motion_threshold=0.02,
        still_frames_required=5,
        min_segment_frames=10,
        motion_debounce_frames=2,
    ):
        self.motion_threshold = motion_threshold
        self.still_frames_required = still_frames_required
        self.min_segment_frames = min_segment_frames
        self.motion_debounce_frames = motion_debounce_frames
        self.buffer = []
        self.hand_counts = []
        self.moved = False
        self.still_count = 0
        self.motion_count = 0
        self.reference = None
        self.reference_hands = 0

    def update(self, landmarks: np.ndarray, hand_count: int):
        """Feed one frame's landmarks.

        `hand_count` is the number of real (non-zero-padded) hand slots
        in `landmarks` — required, since a zero-padded slot (no hand
        present) is indistinguishable from a real hand parked at the
        origin, so it can't be inferred from `landmarks` alone. Pass it
        so a hand appearing or disappearing between frames isn't scored
        as motion.

        Returns the buffered frame sequence when a sign boundary is
        detected, else None.
        """
        self.buffer.append(landmarks)
        self.hand_counts.append(hand_count)

        if self.reference is None:
            self.reference = landmarks
            self.reference_hands = hand_count

        if len(self.buffer) < 2:
            return None

        if not self.moved:
            active_hands = min(self.reference_hands, hand_count)
            motion = (
                0.0
                if active_hands == 0
                else float(np.abs(landmarks[:active_hands] - self.reference[:active_hands]).max())
            )

            if motion > self.motion_threshold:
                self.motion_count += 1
            else:
                self.motion_count = 0
                self.reference = landmarks
                self.reference_hands = hand_count

            if self.motion_count >= self.motion_debounce_frames:
                self.moved = True
                self.still_count = 0
        else:
            prev, prev_hands = self.buffer[-2], self.hand_counts[-2]
            curr, curr_hands = self.buffer[-1], self.hand_counts[-1]
            active_hands = min(prev_hands, curr_hands)
            motion = (
                0.0
                if active_hands == 0
                else float(np.abs(curr[:active_hands] - prev[:active_hands]).max())
            )

            if motion > self.motion_threshold:
                self.still_count = 0
            else:
                self.still_count += 1

        if (
            self.moved
            and self.still_count >= self.still_frames_required
            and len(self.buffer) >= self.min_segment_frames
        ):
            sequence = self.buffer.copy()
            self.buffer = []
            self.hand_counts = []
            self.moved = False
            self.still_count = 0
            self.motion_count = 0
            self.reference = None
            self.reference_hands = 0
            return sequence

        return None
