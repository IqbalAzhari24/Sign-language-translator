import numpy as np
import cv2

from train.train import train


class FakePoint:
    def __init__(self, v):
        self.x = self.y = self.z = v


class FakeResults:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class MeanIntensityDetector:
    """Fake detector: landmark value = mean pixel intensity of the frame.

    Makes the two synthetic classes (different fill values) actually
    separable, so the training loss test is meaningful. Mimics the
    Tasks API shape: each hand is directly a list of landmark points.
    """

    def detect(self, mp_image):
        value = float(np.asarray(mp_image.numpy_view()).mean()) / 255.0
        hand = [FakePoint(value) for _ in range(21)]
        return FakeResults([hand])


def _write_frames(dir_path, count, value):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.full((32, 32, 3), value, dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"frame_{i:04d}.jpg"), frame)


def test_train_runs_and_saves_checkpoint(tmp_path):
    root = tmp_path / "dataset"
    for i in range(6):
        _write_frames(root / "a" / f"sample{i}", 5, value=10)
    for i in range(6):
        _write_frames(root / "b" / f"sample{i}", 5, value=200)

    checkpoint_path = tmp_path / "model.pt"

    losses = train(
        root,
        MeanIntensityDetector(),
        checkpoint_path,
        epochs=15,
        sequence_length=5,
        batch_size=4,
    )

    assert checkpoint_path.exists()
    assert len(losses) == 15
    assert losses[-1] < losses[0]
