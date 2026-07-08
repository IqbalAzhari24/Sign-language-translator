# BIM Sign-to-Text/Speech Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working Phase 1 BIM (Bahasa Isyarat Malaysia) sign-to-text/speech translator: webcam → MediaPipe hand landmarks → PyTorch sequence classifier → live text + speech in a web UI.

**Architecture:** React/Tailwind frontend streams webcam frames over WebSocket to a Python FastAPI backend. The backend runs OpenCV decode + MediaPipe hand-landmark extraction + a small PyTorch GRU sequence classifier, using a gesture-boundary heuristic to know when to run inference. The same landmark-extraction code is shared between the offline training pipeline and the live server so train/inference never drift apart.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, OpenCV, MediaPipe, PyTorch, pytest — React 18, TypeScript, Vite, Tailwind CSS, Vitest, React Testing Library.

---

## File Structure

```
backend/
  pytest.ini
  requirements.txt
  app/
    __init__.py
    landmarks.py        # shared MediaPipe landmark extraction (used by training + serving)
    segmentation.py      # sign-boundary (motion/stillness) heuristic
    model.py              # PyTorch sequence classifier
    inference.py          # per-connection orchestration: landmarks -> boundary -> model -> result
    main.py                # FastAPI app + WebSocket endpoint
  train/
    __init__.py
    dataset.py             # dataset loader (class/sample/frames folder convention)
    train.py                 # training loop + checkpoint export
  tests/
    test_landmarks.py
    test_segmentation.py
    test_model.py
    test_dataset.py
    test_train.py
    test_inference.py
    test_main.py

frontend/
  (Vite scaffold: package.json, vite.config.ts, tailwind.config.js, postcss.config.js, tsconfig*.json, index.html)
  src/
    setupTests.ts
    lib/
      captureFrame.ts
      captureFrame.test.ts
    hooks/
      useWebcam.ts
      useSignSocket.ts
      useSignSocket.test.ts
      useSpeech.ts
      useSpeech.test.ts
    components/
      SubtitleDisplay.tsx
      SubtitleDisplay.test.tsx
      StatusIndicator.tsx
      StatusIndicator.test.tsx
    App.tsx
    index.css

.gitignore
```

---

### Task 1: Backend project scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `backend/app/__init__.py`
- Create: `backend/train/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create requirements file**

```
fastapi
uvicorn[standard]
opencv-python
mediapipe
torch
numpy
pytest
httpx
```

- [ ] **Step 2: Create pytest config so `import app.x` / `import train.x` resolve from `backend/`**

`backend/pytest.ini`:
```ini
[pytest]
pythonpath = .
```

- [ ] **Step 3: Create empty package init files**

`backend/app/__init__.py`, `backend/train/__init__.py`, `backend/tests/__init__.py` — all empty.

- [ ] **Step 4: Create root .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
backend/checkpoints/
node_modules/
frontend/dist/
```

- [ ] **Step 5: Install backend dependencies**

Run: `cd backend && pip install -r requirements.txt`
Expected: all packages install without error.

- [ ] **Step 6: Verify pytest collects with zero tests, zero errors**

Run: `cd backend && pytest --collect-only`
Expected: `no tests ran` (or `collected 0 items`), no import errors.

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/pytest.ini backend/app/__init__.py backend/train/__init__.py backend/tests/__init__.py .gitignore
git commit -m "chore: scaffold backend project structure"
```

---

### Task 2: Shared landmark extraction module

> **Revision note:** the plan originally targeted MediaPipe's legacy `mediapipe.solutions.hands`
> API. That API has been removed from current MediaPipe releases (verified: `mediapipe==0.10.35`
> ships only the Tasks API — `mp.solutions` no longer includes `hands`). This task now targets
> the current **MediaPipe Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`) instead.
> This changes the detector interface from `.process(image_rgb) -> .multi_hand_landmarks` (each
> hand wrapped in a `.landmark` list) to `.detect(mp_image) -> .hand_landmarks` (each hand is
> directly a list of landmark points). Tasks 5, 6, 7, 8 below have been updated to match.

**Files:**
- Create: `backend/app/landmarks.py`
- Create: `backend/models/.gitkeep` (directory placeholder; the actual model file is downloaded,
  not committed — see Step 0)
- Test: `backend/tests/test_landmarks.py`
- Modify: `.gitignore` (ignore the downloaded model binary)

- [ ] **Step 0: Add model directory to .gitignore and download the hand landmarker model**

Add this line to the root `.gitignore`:
```
backend/models/*.task
```

The Tasks API requires a model bundle file (not bundled with the `mediapipe` pip package). Create
the directory and download it:

Run:
```bash
mkdir -p backend/models
curl -L -o backend/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
Expected: `backend/models/hand_landmarker.task` exists and is a few MB (a "data" file, not text —
this is a binary model bundle, that's correct).

Create `backend/models/.gitkeep` (empty file) so the directory itself is tracked even though the
`.task` file inside it is gitignored.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_landmarks.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_landmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.landmarks'`

- [ ] **Step 3: Implement the module**

`backend/app/landmarks.py`:
```python
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MAX_HANDS = 2
NUM_LANDMARKS = 21
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def create_hand_landmarker(model_path=DEFAULT_MODEL_PATH, num_hands=MAX_HANDS):
    """Build a MediaPipe HandLandmarker (Tasks API) in synchronous IMAGE mode.

    IMAGE mode (stateless, one call per frame) is used both here and by the
    training pipeline so train-time and inference-time landmark extraction
    never diverge.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. Download it first:\n"
            f"curl -L -o {model_path} "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=num_hands,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_bgr, hands_detector):
    """Run a HandLandmarker-compatible detector over one BGR frame.

    Returns (landmarks, hand_count) where landmarks is a
    (MAX_HANDS, NUM_LANDMARKS, 3) float32 array, zero-padded when
    fewer than MAX_HANDS are detected. `hands_detector` must expose a
    `.detect(mp_image)` method returning an object with a
    `.hand_landmarks` attribute: a list of hands, each hand a list of
    landmark-like objects with .x/.y/.z (matches MediaPipe Tasks'
    HandLandmarker, or a test fake with the same shape).
    """
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hands_detector.detect(mp_image)

    landmarks = np.zeros((MAX_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)
    hand_count = 0

    for i, hand in enumerate(results.hand_landmarks[:MAX_HANDS]):
        for j, point in enumerate(hand):
            landmarks[i, j] = (point.x, point.y, point.z)
        hand_count += 1

    return landmarks, hand_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_landmarks.py -v`
Expected: 3 passed (the real-detector test runs, not skipped, since Step 0 downloaded the model)

- [ ] **Step 5: Commit**

```bash
git add backend/app/landmarks.py backend/tests/test_landmarks.py backend/models/.gitkeep .gitignore
git commit -m "feat: add shared MediaPipe Tasks-API landmark extraction module"
```

---

### Task 3: Sign-boundary segmentation heuristic

**Files:**
- Create: `backend/app/segmentation.py`
- Test: `backend/tests/test_segmentation.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_segmentation.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.segmentation'`

- [ ] **Step 3: Implement the module**

`backend/app/segmentation.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_segmentation.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/segmentation.py backend/tests/test_segmentation.py
git commit -m "feat: add sign-boundary segmentation heuristic"
```

---

### Task 4: PyTorch sequence classifier

**Files:**
- Create: `backend/app/model.py`
- Test: `backend/tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_model.py`:
```python
import torch

from app.model import SignSequenceClassifier, INPUT_SIZE


def test_forward_output_shape():
    model = SignSequenceClassifier(num_classes=5)
    x = torch.randn(4, 30, INPUT_SIZE)  # batch=4, seq_len=30
    logits = model(x)
    assert logits.shape == (4, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.model'`

- [ ] **Step 3: Implement the model**

`backend/app/model.py`:
```python
import torch
import torch.nn as nn

from .landmarks import MAX_HANDS, NUM_LANDMARKS

INPUT_SIZE = MAX_HANDS * NUM_LANDMARKS * 3  # 126: flattened per-frame landmark vector


class SignSequenceClassifier(nn.Module):
    def __init__(self, num_classes: int, input_size: int = INPUT_SIZE, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_model.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/model.py backend/tests/test_model.py
git commit -m "feat: add PyTorch GRU sequence classifier"
```

---

### Task 5: Dataset loader

**Files:**
- Create: `backend/train/dataset.py`
- Test: `backend/tests/test_dataset.py`

Convention: `dataset_root/<class_name>/<sample_id>/*.jpg` — one subfolder of sequential frame
images per recorded sign instance. This keeps the loader dataset-agnostic: whatever public BIM
dataset is used, reorganize it into this folder layout (a one-off conversion script, not part of
this pipeline) and the loader works unchanged.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.dataset'`

- [ ] **Step 3: Implement the loader**

`backend/train/dataset.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.landmarks import extract_landmarks, MAX_HANDS, NUM_LANDMARKS


class SignDataset(Dataset):
    """Loads dataset_root/<class_name>/<sample_id>/*.jpg into landmark sequences."""

    def __init__(self, root_dir, hands_detector, sequence_length=30):
        self.root_dir = Path(root_dir)
        self.hands_detector = hands_detector
        self.sequence_length = sequence_length

        self.classes = sorted(p.name for p in self.root_dir.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for sample_dir in sorted(class_dir.iterdir()):
                if sample_dir.is_dir():
                    self.samples.append((sample_dir, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]
        # Take the LAST sequence_length frames (not the first) so an
        # over-long recorded sample uses the same "most recent frames"
        # convention as live inference's _prepare_sequence, which also
        # keeps the tail when a boundary-detected buffer runs long.
        frame_paths = sorted(sample_dir.glob("*.jpg"))[-self.sequence_length :]

        # Left-pad (real frames placed at the end) so a sample with fewer
        # frames than sequence_length still ends on a real frame — the
        # model reads only the final GRU timestep (app/model.py), and must
        # see the same left-padding convention at train time as at
        # inference time (app/inference.py's _prepare_sequence).
        sequence = np.zeros(
            (self.sequence_length, MAX_HANDS, NUM_LANDMARKS, 3), dtype=np.float32
        )
        offset = self.sequence_length - len(frame_paths)
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            landmarks, _ = extract_landmarks(frame, self.hands_detector)
            sequence[offset + i] = landmarks

        flat = sequence.reshape(self.sequence_length, -1)
        return torch.from_numpy(flat), label
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_dataset.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/train/dataset.py backend/tests/test_dataset.py
git commit -m "feat: add dataset-agnostic sign dataset loader"
```

---

### Task 6: Training script

**Files:**
- Create: `backend/train/train.py`
- Test: `backend/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Uses a fake detector whose output value is derived from frame pixel intensity, so the two
synthetic classes are actually distinguishable and loss can meaningfully decrease (a fake that
always returns zero landmarks regardless of image content would make both classes identical and
loss wouldn't reliably drop).

`backend/tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.train'`

- [ ] **Step 3: Implement the training script**

`backend/train/train.py`:
```python
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from app.model import SignSequenceClassifier
from train.dataset import SignDataset


def train(
    dataset_root,
    hands_detector,
    checkpoint_path,
    epochs=20,
    sequence_length=30,
    batch_size=8,
    lr=1e-3,
):
    dataset = SignDataset(dataset_root, hands_detector, sequence_length=sequence_length)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, _val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = SignSequenceClassifier(num_classes=len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

    torch.save(
        {"model_state": model.state_dict(), "classes": dataset.classes},
        checkpoint_path,
    )
    return losses


if __name__ == "__main__":
    from app.landmarks import create_hand_landmarker

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Path laid out as <root>/<class>/<sample>/*.jpg")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    detector = create_hand_landmarker()
    train(
        args.dataset_root,
        detector,
        args.checkpoint_path,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    detector.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_train.py -v`
Expected: 1 passed (may take a few seconds — real training loop)

- [ ] **Step 5: Commit**

```bash
git add backend/train/train.py backend/tests/test_train.py
git commit -m "feat: add training script with checkpoint export"
```

---

### Task 7: Inference orchestration

**Files:**
- Create: `backend/app/inference.py`
- Test: `backend/tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_inference.py`:
```python
import numpy as np
import torch
import torch.nn as nn

from app.inference import SessionInference
from app.landmarks import NUM_LANDMARKS


class ZeroHandDetector:
    def detect(self, mp_image):
        class R:
            hand_landmarks = []
        return R()


class OneHandDetector:
    def detect(self, mp_image):
        class Point:
            def __init__(self):
                self.x = self.y = self.z = 0.1

        hand = [Point() for _ in range(NUM_LANDMARKS)]

        class R:
            hand_landmarks = [hand]
        return R()


class AlwaysReadyBoundary:
    """Fake boundary detector: every frame immediately completes a 1-frame sequence."""
    def update(self, landmarks):
        return [landmarks]


class NeverReadyBoundary:
    def update(self, landmarks):
        return None


class FixedLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits)

    def forward(self, x):
        return self._logits.unsqueeze(0)


def _blank_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_no_hand_returns_no_hand_status():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),
        hands_detector=ZeroHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "no_hand"


def test_tracking_when_boundary_not_ready():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=NeverReadyBoundary(),
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "tracking"


def test_recognized_when_confidence_high():
    session = SessionInference(
        model=FixedLogitsModel([10.0, 0.0]),  # softmax -> ~1.0 for class 0
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
        confidence_threshold=0.6,
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "recognized"
    assert result.label == "a"


def test_unsure_when_confidence_low():
    session = SessionInference(
        model=FixedLogitsModel([0.1, 0.0]),  # near 50/50 -> low confidence
        hands_detector=OneHandDetector(),
        class_names=["a", "b"],
        boundary_detector=AlwaysReadyBoundary(),
        confidence_threshold=0.9,
    )
    result = session.process_frame(_blank_frame())
    assert result.status == "unsure"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.inference'`

- [ ] **Step 3: Implement the orchestration module**

`backend/app/inference.py`:
```python
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .landmarks import extract_landmarks
from .segmentation import SignBoundaryDetector


@dataclass
class InferenceResult:
    status: str  # "no_hand" | "tracking" | "unsure" | "recognized"
    label: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self):
        data = {"status": self.status}
        if self.label is not None:
            data["label"] = self.label
            data["confidence"] = self.confidence
        return data


class SessionInference:
    def __init__(
        self,
        model,
        hands_detector,
        class_names,
        sequence_length=30,
        confidence_threshold=0.6,
        boundary_detector=None,
    ):
        self.model = model
        self.hands_detector = hands_detector
        self.class_names = class_names
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.boundary_detector = boundary_detector or SignBoundaryDetector()

    def process_frame(self, frame_bgr) -> InferenceResult:
        landmarks, hand_count = extract_landmarks(frame_bgr, self.hands_detector)
        if hand_count == 0:
            return InferenceResult(status="no_hand")

        sequence = self.boundary_detector.update(landmarks)
        if sequence is None:
            return InferenceResult(status="tracking")

        tensor_seq = self._prepare_sequence(sequence)
        with torch.no_grad():
            logits = self.model(tensor_seq.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

        if confidence.item() < self.confidence_threshold:
            return InferenceResult(status="unsure")

        return InferenceResult(
            status="recognized",
            label=self.class_names[idx.item()],
            confidence=confidence.item(),
        )

    def _prepare_sequence(self, sequence: List[np.ndarray]) -> torch.Tensor:
        flat = [frame.reshape(-1) for frame in sequence]
        if len(flat) < self.sequence_length:
            # Left-pad (pad at the start) so the sequence always *ends* on a
            # real frame. The model reads only the final GRU timestep
            # (see app/model.py) — right-padding would leave that final
            # timestep dominated by trailing zero-padding instead of the
            # actual sign.
            pad = [np.zeros_like(flat[0])] * (self.sequence_length - len(flat))
            flat = pad + flat
        else:
            flat = flat[-self.sequence_length :]
        return torch.tensor(np.stack(flat), dtype=torch.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_inference.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/inference.py backend/tests/test_inference.py
git commit -m "feat: add per-connection inference orchestration"
```

---

### Task 8: FastAPI WebSocket server

**Files:**
- Create: `backend/app/main.py`
- Test: `backend/tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_main.py`:
```python
import numpy as np
import cv2
import torch
from fastapi.testclient import TestClient

from app.main import create_app, load_model
from app.model import SignSequenceClassifier


def _save_dummy_checkpoint(path, class_names=("a", "b")):
    model = SignSequenceClassifier(num_classes=len(class_names))
    torch.save({"model_state": model.state_dict(), "classes": list(class_names)}, path)


def test_load_model_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "nope.pt"
    try:
        load_model(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_websocket_blank_frame_returns_no_hand(tmp_path):
    from pathlib import Path

    import pytest
    from app.landmarks import DEFAULT_MODEL_PATH

    if not Path(DEFAULT_MODEL_PATH).exists():
        pytest.skip("hand_landmarker.task not downloaded — see Task 2 Step 0")

    checkpoint_path = tmp_path / "model.pt"
    _save_dummy_checkpoint(checkpoint_path)
    app = create_app(checkpoint_path=checkpoint_path)
    client = TestClient(app)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok

    with client.websocket_connect("/ws/sign-stream") as ws:
        ws.send_bytes(encoded.tobytes())
        response = ws.receive_json()

    assert response["status"] == "no_hand"


def test_websocket_malformed_frame_is_skipped_not_fatal(tmp_path):
    from pathlib import Path

    import pytest
    from app.landmarks import DEFAULT_MODEL_PATH

    if not Path(DEFAULT_MODEL_PATH).exists():
        pytest.skip("hand_landmarker.task not downloaded — see Task 2 Step 0")

    checkpoint_path = tmp_path / "model.pt"
    _save_dummy_checkpoint(checkpoint_path)
    app = create_app(checkpoint_path=checkpoint_path)
    client = TestClient(app)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok

    with client.websocket_connect("/ws/sign-stream") as ws:
        ws.send_bytes(b"not a real jpeg")  # cv2.imdecode returns None for this
        ws.send_bytes(encoded.tobytes())  # connection must still be alive after
        response = ws.receive_json()

    assert response["status"] == "no_hand"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement the server**

`backend/app/main.py`:
```python
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .inference import SessionInference
from .model import SignSequenceClassifier

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


def load_model(checkpoint_path: Path):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train a model first with train/train.py before starting the server."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SignSequenceClassifier(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["classes"]


def create_app(checkpoint_path: Path = CHECKPOINT_PATH) -> FastAPI:
    from .landmarks import create_hand_landmarker

    model, class_names = load_model(checkpoint_path)
    app = FastAPI()

    @app.websocket("/ws/sign-stream")
    async def sign_stream(websocket: WebSocket):
        await websocket.accept()
        detector = create_hand_landmarker()
        session = SessionInference(model=model, hands_detector=detector, class_names=class_names)
        try:
            while True:
                data = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    # Malformed/truncated frame (e.g. a dropped or partial
                    # packet) — skip it rather than letting a decode failure
                    # kill the whole connection. One bad frame over an
                    # inherently lossy live-video transport shouldn't end
                    # the session.
                    continue
                try:
                    result = session.process_frame(frame)
                except Exception:
                    logger.exception("Error processing frame, skipping")
                    continue
                await websocket.send_json(result.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            detector.close()

    return app
```

> **Revision note (robustness):** code review on the initial implementation found that a
> malformed frame (e.g. `cv2.imdecode` returning `None` for corrupt/truncated bytes) would raise
> an uncaught `TypeError` inside `extract_landmarks`, terminating the entire WebSocket connection
> for what should be a recoverable, momentary glitch — realistic on a live video stream over an
> unreliable transport. The fix above skips a frame that fails to decode (`frame is None`) and
> also catches any exception from `session.process_frame` itself, continuing the loop either way
> instead of crashing the connection. `detector.close()` in `finally` still runs on genuine
> disconnects exactly as before.
>
> **Follow-up revision:** a later code-quality review pointed out that catching `Exception`
> silently (no log line, no client-facing signal) makes a genuine bug in the inference pipeline
> just as invisible as a malformed frame — a real regression would degrade to "responses quietly
> stop" instead of a loud, debuggable failure. Added `logging.getLogger(__name__)` and a
> `logger.exception(...)` call in the except block so a real error is still recorded, while the
> connection itself stays alive either way.

> **Revision note:** the plan originally had a module-level `app = create_app()` line, intended
> to give `uvicorn app.main:app` a ready-made ASGI app to serve, matching the "fail fast if
> checkpoint missing" requirement. In practice this breaks `test_main.py`: `from app.main import
> create_app, load_model` executes the whole module, including that module-level line, which
> calls `load_model(CHECKPOINT_PATH)` and raises `FileNotFoundError` **at import time** — before
> either test function runs — since no checkpoint exists in a fresh checkout. pytest then reports
> a collection error, not "2 passed". The fix: drop the module-level `app = create_app()` line
> entirely and run the server via uvicorn's **factory pattern** instead (`uvicorn
> app.main:create_app --factory`), which calls `create_app()` lazily at server startup, not at
> import time. Fail-fast behavior is preserved (the server still refuses to start without a
> checkpoint) — it just fails at server-start time instead of module-import time, which is
> actually more correct: importing the module for testing/tooling purposes should never have a
> side effect that depends on unrelated runtime state.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_main.py -v`
Expected: 3 passed

Note: with the module-level `app = create_app()` line removed, `import app.main` (and thus
`from app.main import create_app, load_model` in tests) has no side effects and always succeeds.
Running the real server later (Task 14) uses `uvicorn app.main:create_app --factory`, which calls
`create_app()` — and therefore `load_model(CHECKPOINT_PATH)` — only when the server actually
starts, still failing fast if `backend/checkpoints/model.pt` is missing at that point. Tests use
`create_app(checkpoint_path=...)` directly so they don't depend on the default path either way.

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/tests/test_main.py
git commit -m "feat: add FastAPI WebSocket sign-stream server"
```

---

### Task 9: Frontend scaffold (Vite + React + TS + Tailwind + Vitest)

**Files:**
- Create: `frontend/` (Vite scaffold)
- Modify: `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/vite.config.ts`, `frontend/src/index.css`, `frontend/package.json`
- Create: `frontend/src/setupTests.ts`

- [ ] **Step 1: Scaffold the Vite React-TS app**

Run: `npm create vite@latest frontend -- --template react-ts --yes`
Expected: `frontend/` created with a working React+TS app.

- [ ] **Step 2: Install Tailwind and test dependencies**

Run: `cd frontend && npm install && npm install -D tailwindcss postcss autoprefixer vitest @testing-library/react @testing-library/jest-dom jsdom`

- [ ] **Step 3: Configure Tailwind**

`frontend/tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

`frontend/postcss.config.js`:
```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

`frontend/src/index.css` (replace entire contents):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 4: Configure Vitest**

`frontend/vite.config.ts`:
```typescript
/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/setupTests.ts",
    globals: true,
  },
});
```

`frontend/src/setupTests.ts`:
```typescript
import "@testing-library/jest-dom/vitest";
```

Add to `frontend/package.json` `"scripts"`:
```json
"test": "vitest run"
```

- [ ] **Step 5: Verify the test runner works with zero tests**

Run: `cd frontend && npm run test`
Expected: Vitest runs, reports `No test files found` (or 0 tests), exits without error.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "chore: scaffold frontend with Vite, Tailwind, Vitest"
```

---

### Task 10: Webcam hook + frame capture helper

**Files:**
- Create: `frontend/src/hooks/useWebcam.ts`
- Create: `frontend/src/lib/captureFrame.ts`
- Test: `frontend/src/lib/captureFrame.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/lib/captureFrame.test.ts`:
```typescript
import { describe, expect, it, vi } from "vitest";
import { captureFrameBlob } from "./captureFrame";

describe("captureFrameBlob", () => {
  it("draws the video frame onto a canvas and requests a jpeg blob", async () => {
    const drawImage = vi.fn();
    const toBlob = vi.fn((cb: BlobCallback) => cb(new Blob(["x"])));
    const getContext = vi.fn(() => ({ drawImage }));

    vi.spyOn(document, "createElement").mockReturnValue({
      width: 0,
      height: 0,
      getContext,
      toBlob,
    } as unknown as HTMLCanvasElement);

    const video = { videoWidth: 320, videoHeight: 240 } as HTMLVideoElement;
    const blob = await captureFrameBlob(video);

    expect(getContext).toHaveBeenCalledWith("2d");
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 320, 240);
    expect(blob).toBeInstanceOf(Blob);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- captureFrame`
Expected: FAIL — `Failed to resolve import "./captureFrame"`

- [ ] **Step 3: Implement capture helper and webcam hook**

`frontend/src/lib/captureFrame.ts`:
```typescript
export function captureFrameBlob(
  video: HTMLVideoElement,
  quality = 0.7
): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}
```

`frontend/src/hooks/useWebcam.ts`:
```typescript
import { useEffect, useRef, useState } from "react";

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          setReady(true);
        }
      })
      .catch((err: Error) => setError(err.message));

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, ready, error };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm run test -- captureFrame`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useWebcam.ts frontend/src/lib/captureFrame.ts frontend/src/lib/captureFrame.test.ts
git commit -m "feat: add webcam hook and frame capture helper"
```

---

### Task 11: Sign-stream WebSocket hook

**Files:**
- Create: `frontend/src/hooks/useSignSocket.ts`
- Test: `frontend/src/hooks/useSignSocket.test.ts`

- [ ] **Step 1: Write the failing tests**

`frontend/src/hooks/useSignSocket.test.ts`:
```typescript
import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { useSignSocket } from "./useSignSocket";

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: unknown[] = [];

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }

  send(data: unknown) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  // @ts-expect-error test override of global WebSocket
  global.WebSocket = FakeWebSocket;
});

describe("useSignSocket", () => {
  it("connects and updates result on incoming message", async () => {
    const { result } = renderHook(() => useSignSocket("ws://test"));

    await waitFor(() => expect(result.current.connected).toBe(true));

    const socket = FakeWebSocket.instances[0];
    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({ status: "recognized", label: "A", confidence: 0.9 }),
      });
    });

    expect(result.current.result).toEqual({ status: "recognized", label: "A", confidence: 0.9 });
  });

  it("sends frames only while connected", async () => {
    const { result } = renderHook(() => useSignSocket("ws://test"));
    await waitFor(() => expect(result.current.connected).toBe(true));

    const blob = new Blob(["frame"]);
    act(() => {
      result.current.sendFrame(blob);
    });

    expect(FakeWebSocket.instances[0].sent).toContain(blob);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- useSignSocket`
Expected: FAIL — `Failed to resolve import "./useSignSocket"`

- [ ] **Step 3: Implement the hook**

`frontend/src/hooks/useSignSocket.ts`:
```typescript
import { useEffect, useRef, useState, useCallback } from "react";

export type SignResult =
  | { status: "no_hand" | "tracking" | "unsure" }
  | { status: "recognized"; label: string; confidence: number };

export function useSignSocket(url: string) {
  const [connected, setConnected] = useState(false);
  const [result, setResult] = useState<SignResult>({ status: "no_hand" });
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryDelay = 500;

    function connect() {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        retryDelay = 500;
        setConnected(true);
      };
      socket.onmessage = (event) => {
        setResult(JSON.parse(event.data));
      };
      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          setTimeout(connect, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 8000);
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      socketRef.current?.close();
    };
  }, [url]);

  const sendFrame = useCallback((blob: Blob) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(blob);
    }
  }, []);

  return { connected, result, sendFrame };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- useSignSocket`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useSignSocket.ts frontend/src/hooks/useSignSocket.test.ts
git commit -m "feat: add reconnecting sign-stream WebSocket hook"
```

---

### Task 12: Speech-output hook

**Files:**
- Create: `frontend/src/hooks/useSpeech.ts`
- Test: `frontend/src/hooks/useSpeech.test.ts`

- [ ] **Step 1: Write the failing test**

`frontend/src/hooks/useSpeech.test.ts`:
```typescript
import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useSpeech } from "./useSpeech";

beforeEach(() => {
  // @ts-expect-error test override
  window.speechSynthesis = { speak: vi.fn() };
  // @ts-expect-error test override
  global.SpeechSynthesisUtterance = vi.fn().mockImplementation((text: string) => ({ text }));
});

describe("useSpeech", () => {
  it("speaks once per newly recognized label", () => {
    const { rerender } = renderHook(({ result }) => useSpeech(result), {
      initialProps: { result: { status: "recognized", label: "A", confidence: 0.9 } as const },
    });

    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1);

    rerender({ result: { status: "recognized", label: "A", confidence: 0.95 } as const });
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1);

    rerender({ result: { status: "recognized", label: "B", confidence: 0.9 } as const });
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- useSpeech`
Expected: FAIL — `Failed to resolve import "./useSpeech"`

- [ ] **Step 3: Implement the hook**

`frontend/src/hooks/useSpeech.ts`:
```typescript
import { useEffect, useRef } from "react";
import type { SignResult } from "./useSignSocket";

export function useSpeech(result: SignResult) {
  const lastSpokenRef = useRef<string | null>(null);

  useEffect(() => {
    if (result.status !== "recognized") return;
    if (result.label === lastSpokenRef.current) return;
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    lastSpokenRef.current = result.label;
    const utterance = new SpeechSynthesisUtterance(result.label);
    window.speechSynthesis.speak(utterance);
  }, [result]);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm run test -- useSpeech`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useSpeech.ts frontend/src/hooks/useSpeech.test.ts
git commit -m "feat: add speech-output hook for newly recognized signs"
```

---

### Task 13: Subtitle and status display components

**Files:**
- Create: `frontend/src/components/SubtitleDisplay.tsx`
- Create: `frontend/src/components/StatusIndicator.tsx`
- Test: `frontend/src/components/SubtitleDisplay.test.tsx`
- Test: `frontend/src/components/StatusIndicator.test.tsx`

- [ ] **Step 1: Write the failing tests**

`frontend/src/components/SubtitleDisplay.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SubtitleDisplay } from "./SubtitleDisplay";

describe("SubtitleDisplay", () => {
  it("shows the recognized label", () => {
    render(<SubtitleDisplay result={{ status: "recognized", label: "SATU", confidence: 0.9 }} />);
    expect(screen.getByText("SATU")).toBeInTheDocument();
  });

  it("shows nothing when not recognized", () => {
    render(<SubtitleDisplay result={{ status: "tracking" }} />);
    expect(screen.queryByText("SATU")).not.toBeInTheDocument();
  });
});
```

`frontend/src/components/StatusIndicator.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusIndicator } from "./StatusIndicator";

describe("StatusIndicator", () => {
  it("shows the label for the given status", () => {
    render(<StatusIndicator status="no_hand" />);
    expect(screen.getByText("No hand detected")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- SubtitleDisplay StatusIndicator`
Expected: FAIL — modules don't exist yet

- [ ] **Step 3: Implement the components**

`frontend/src/components/SubtitleDisplay.tsx`:
```tsx
import type { SignResult } from "../hooks/useSignSocket";

export function SubtitleDisplay({ result }: { result: SignResult }) {
  const text = result.status === "recognized" ? result.label : "";

  return (
    <div className="absolute bottom-6 left-1/2 min-h-[3rem] min-w-[8rem] -translate-x-1/2 rounded-lg bg-black/70 px-6 py-3 text-center text-2xl font-semibold text-white">
      {text}
    </div>
  );
}
```

`frontend/src/components/StatusIndicator.tsx`:
```tsx
import type { SignResult } from "../hooks/useSignSocket";

const LABELS: Record<SignResult["status"], string> = {
  no_hand: "No hand detected",
  tracking: "Recognizing…",
  unsure: "Unsure",
  recognized: "Recognized",
};

const COLORS: Record<SignResult["status"], string> = {
  no_hand: "bg-gray-400",
  tracking: "bg-yellow-400",
  unsure: "bg-orange-400",
  recognized: "bg-green-500",
};

export function StatusIndicator({ status }: { status: SignResult["status"] }) {
  return (
    <div className="flex items-center gap-2 rounded-full bg-white/90 px-3 py-1 text-sm shadow">
      <span className={`h-2 w-2 rounded-full ${COLORS[status]}`} />
      {LABELS[status]}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- SubtitleDisplay StatusIndicator`
Expected: 3 passed

- [ ] **Step 5 (optional polish): Refine visual styling with 21st.dev**

If available, call `mcp__magic__21st_magic_component_refiner` against these two components to
polish spacing/typography/animation, then re-run Step 4 to confirm tests still pass — the
refiner should only touch markup/classes, not the `result`/`status` prop contracts the tests
assert against.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: add subtitle and status display components"
```

---

### Task 14: Wire up App and manual end-to-end check

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Wire hooks and components together**

`frontend/src/App.tsx`:
```tsx
import { useEffect } from "react";
import { useWebcam } from "./hooks/useWebcam";
import { useSignSocket } from "./hooks/useSignSocket";
import { useSpeech } from "./hooks/useSpeech";
import { captureFrameBlob } from "./lib/captureFrame";
import { SubtitleDisplay } from "./components/SubtitleDisplay";
import { StatusIndicator } from "./components/StatusIndicator";

const SEND_INTERVAL_MS = 80; // ~12.5 fps
const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/sign-stream";

export default function App() {
  const { videoRef, ready } = useWebcam();
  const { connected, result, sendFrame } = useSignSocket(WS_URL);
  useSpeech(result);

  useEffect(() => {
    if (!ready) return;
    const interval = setInterval(async () => {
      const video = videoRef.current;
      if (!video) return;
      const blob = await captureFrameBlob(video);
      if (blob) sendFrame(blob);
    }, SEND_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [ready, videoRef, sendFrame]);

  return (
    <div className="relative h-screen w-screen bg-black">
      <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />
      <div className="absolute right-4 top-4">
        <StatusIndicator status={connected ? result.status : "no_hand"} />
      </div>
      <SubtitleDisplay result={result} />
    </div>
  );
}
```

- [ ] **Step 2: Train a real checkpoint**

Reorganize a chosen BIM alphabet/digit dataset into `<root>/<class>/<sample>/*.jpg`, then run:

Run: `cd backend && python -m train.train /path/to/dataset backend/checkpoints/model.pt --epochs 30`
Expected: script completes, `backend/checkpoints/model.pt` exists.

- [ ] **Step 3: Start backend server**

Run: `cd backend && uvicorn app.main:create_app --factory --reload --port 8000`
Expected: server starts without error (fails fast if checkpoint from Step 2 is missing).

- [ ] **Step 4: Start frontend dev server**

Run: `cd frontend && npm run dev`
Expected: Vite dev server starts, prints local URL.

- [ ] **Step 5: Manual end-to-end check**

Open the printed URL in a browser, grant webcam permission, and verify:
- Status indicator shows "No hand detected" with no hand in frame
- Performing a trained sign shows "Recognizing…" then the recognized label as subtitle text
- The recognized label is spoken aloud (may require one click first, per browser autoplay policy)
- Removing the hand returns status to "No hand detected" without a stale subtitle stuck on screen

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire webcam capture, sign-stream, and speech output into App"
```

---

## Self-Review Notes

- **Spec coverage:** live webcam input (Task 10), server-side MediaPipe over WebSocket (Task 8),
  sequence model for motion signs (Task 4), segmentation heuristic (Task 3), text + browser TTS
  output (Tasks 11–12), dataset-agnostic loader (Task 5), shared landmark code between train/serve
  (Task 2 imported by both Task 5 and Task 8), confidence-threshold "unsure" state (Task 7),
  fail-fast missing checkpoint (Task 8), local-only dev run (Task 14) — all covered.
- **Placeholder scan:** no TBD/TODO in any step; the one open item (exact dataset source) is
  explicitly deferred to a manual reorganize-into-folder-convention step in Task 14, not a gap in
  the pipeline itself.
- **Type consistency:** `MAX_HANDS`/`NUM_LANDMARKS` defined once in `landmarks.py` and imported
  everywhere else (`model.py`, `dataset.py`) rather than re-declared. `SignResult` type defined
  once in `useSignSocket.ts` and imported by `useSpeech.ts`, `SubtitleDisplay.tsx`,
  `StatusIndicator.tsx`. `InferenceResult.to_dict()` keys (`status`, `label`, `confidence`) match
  what the frontend `SignResult` type expects.
