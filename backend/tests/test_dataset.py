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
