import numpy as np

from app.segmentation import SignBoundaryDetector


def _frame(value, hands=2):
    frame = np.zeros((2, 21, 3), dtype=np.float32)
    frame[:hands] = value
    return frame


def test_no_trigger_when_hand_stays_still():
    detector = SignBoundaryDetector(motion_threshold=0.02, still_frames_required=3)
    for _ in range(10):
        result = detector.update(_frame(0.1))
        assert result is None


def test_triggers_after_motion_then_stillness():
    detector = SignBoundaryDetector(
        motion_threshold=0.02,
        still_frames_required=3,
        min_segment_frames=5,
        motion_debounce_frames=1,
    )
    assert detector.update(_frame(0.0)) is None
    assert detector.update(_frame(0.5)) is None  # big jump -> motion detected
    assert detector.update(_frame(0.5)) is None  # still, count 1
    assert detector.update(_frame(0.5)) is None  # still, count 2
    result = detector.update(_frame(0.5))        # still, count 3 -> trigger
    assert result is not None
    assert len(result) == 5


def test_resets_after_trigger():
    detector = SignBoundaryDetector(
        motion_threshold=0.02,
        still_frames_required=2,
        min_segment_frames=3,
        motion_debounce_frames=1,
    )
    detector.update(_frame(0.0))
    detector.update(_frame(0.5))
    detector.update(_frame(0.5))
    triggered = detector.update(_frame(0.5))
    assert triggered is not None
    # After a trigger, the buffer must be cleared and require fresh motion.
    assert detector.update(_frame(0.5)) is None
    assert detector.update(_frame(0.5)) is None


def test_single_jitter_frame_does_not_arm_moved():
    # Regression test: a lone above-threshold frame (jitter) right after a
    # hand settles must not re-arm "moved" and cause a spurious re-trigger
    # while the sign is just being held steady.
    detector = SignBoundaryDetector(
        motion_threshold=0.02,
        still_frames_required=3,
        min_segment_frames=3,
        motion_debounce_frames=2,
    )
    detector.update(_frame(0.5))
    detector.update(_frame(0.5 + 0.05))  # one jitter frame over threshold
    for _ in range(10):
        result = detector.update(_frame(0.5))
        assert result is None


def test_min_segment_frames_blocks_short_spurious_segment():
    # Two consecutive motion frames clear the debounce and enough stillness
    # follows early, but a trigger must not fire until the buffer holds at
    # least min_segment_frames frames.
    detector = SignBoundaryDetector(
        motion_threshold=0.02,
        still_frames_required=2,
        min_segment_frames=8,
        motion_debounce_frames=2,
    )
    detector.update(_frame(0.0))          # buffer len 1
    detector.update(_frame(0.3))          # len 2, motion 1/2 -> still not moved
    detector.update(_frame(0.6))          # len 3, motion 2/2 -> moved=True
    assert detector.update(_frame(0.6)) is None  # len 4, still count 1
    assert detector.update(_frame(0.6)) is None  # len 5, still count 2 but len < 8
    assert detector.update(_frame(0.6)) is None  # len 6
    assert detector.update(_frame(0.6)) is None  # len 7
    result = detector.update(_frame(0.6))        # len 8 -> trigger
    assert result is not None
    assert len(result) == 8


def test_hand_appearing_is_not_scored_as_motion():
    # A second hand entering frame (1 -> 2 detected hands) zero-pads its
    # slot beforehand; without masking, that jump would dwarf real jitter
    # and falsely register as motion.
    detector = SignBoundaryDetector(motion_threshold=0.02, still_frames_required=3)
    detector.update(_frame(0.1, hands=1), hand_count=1)
    for _ in range(10):
        result = detector.update(_frame(0.1, hands=2), hand_count=2)
        assert result is None
