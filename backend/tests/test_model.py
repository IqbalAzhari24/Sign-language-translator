import torch

from app.model import SignSequenceClassifier, INPUT_SIZE


def test_forward_output_shape():
    model = SignSequenceClassifier(num_classes=5)
    x = torch.randn(4, 30, INPUT_SIZE)  # batch=4, seq_len=30
    logits = model(x)
    assert logits.shape == (4, 5)
