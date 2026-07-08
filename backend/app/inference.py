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
