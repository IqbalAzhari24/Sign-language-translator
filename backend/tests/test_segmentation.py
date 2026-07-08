import numpy as np

from app.segmentation import SignBoundaryDetector


def _frame(value):
    return np.full((2, 21, 3), value, dtype=np.float32)


def test_no_trigger_when_hand_stays_still():
    detector = SignBoundaryDetector(motion_threshold=0.02, still_frames_required=3)
    for _ in range(10):
        result = detector.update(_frame(0.1))
        assert result is None


def test_triggers_after_motion_then_stillness():
    detector = SignBoundaryDetector(motion_threshold=0.02, still_frames_required=3)
    assert detector.update(_frame(0.0)) is None
    assert detector.update(_frame(0.5)) is None  # big jump -> motion detected
    assert detector.update(_frame(0.5)) is None  # still, count 1
    assert detector.update(_frame(0.5)) is None  # still, count 2
    result = detector.update(_frame(0.5))        # still, count 3 -> trigger
    assert result is not None
    assert len(result) == 5


def test_resets_after_trigger():
    detector = SignBoundaryDetector(motion_threshold=0.02, still_frames_required=2)
    detector.update(_frame(0.0))
    detector.update(_frame(0.5))
    detector.update(_frame(0.5))
    triggered = detector.update(_frame(0.5))
    assert triggered is not None
    # After a trigger, the buffer must be cleared and require fresh motion.
    assert detector.update(_frame(0.5)) is None
    assert detector.update(_frame(0.5)) is None
