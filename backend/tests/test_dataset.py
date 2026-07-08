import numpy as np
import cv2

from train.dataset import SignDataset


class FakeResults:
    hand_landmarks = []


class FakeDetector:
    def detect(self, mp_image):
        return FakeResults()


def _write_blank_frames(dir_path, count):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"frame_{i:04d}.jpg"), frame)


def test_dataset_lists_classes_and_samples(tmp_path):
    _write_blank_frames(tmp_path / "a" / "sample1", 5)
    _write_blank_frames(tmp_path / "b" / "sample1", 5)

    dataset = SignDataset(tmp_path, FakeDetector(), sequence_length=10)

    assert dataset.classes == ["a", "b"]
    assert len(dataset) == 2


def test_dataset_getitem_shape_and_label(tmp_path):
    _write_blank_frames(tmp_path / "a" / "sample1", 5)

    dataset = SignDataset(tmp_path, FakeDetector(), sequence_length=10)
    sequence, label = dataset[0]

    assert sequence.shape == (10, 2 * 21 * 3)
    assert label == 0


def test_dataset_over_length_sample_keeps_last_frames(tmp_path):
    class IntensityResults:
        def __init__(self, hand_landmarks):
            self.hand_landmarks = hand_landmarks

    class IntensityDetector:
        def detect(self, mp_image):
            value = float(mp_image.numpy_view().mean()) / 255.0
            hand = [type("P", (), {"x": value, "y": value, "z": value})() for _ in range(21)]
            return IntensityResults([hand])

    sample_dir = tmp_path / "a" / "sample1"
    sample_dir.mkdir(parents=True)
    # 15 frames with distinct, increasing fill values 0..14
    for i in range(15):
        frame = np.full((32, 32, 3), i * 10, dtype=np.uint8)
        cv2.imwrite(str(sample_dir / f"frame_{i:04d}.jpg"), frame)

    dataset = SignDataset(tmp_path, IntensityDetector(), sequence_length=10)
    sequence, _label = dataset[0]

    # Last 10 frames (fill values 5*10..14*10) should be kept, not the first 10.
    # sequence is flattened (sequence_length, MAX_HANDS*NUM_LANDMARKS*3); every
    # value in a given timestep's row should equal that frame's intensity/255.
    first_kept_frame_value = (5 * 10) / 255.0
    assert abs(sequence[0, 0].item() - first_kept_frame_value) < 1e-4
    last_frame_value = (14 * 10) / 255.0
    assert abs(sequence[-1, 0].item() - last_frame_value) < 1e-4
