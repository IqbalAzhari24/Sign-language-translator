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
