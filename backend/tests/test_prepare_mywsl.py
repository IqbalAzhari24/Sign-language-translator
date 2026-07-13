import numpy as np
import cv2

from train.prepare_mywsl import prepare


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def test_prepare_reorganizes_split_class_images_into_sample_folders(tmp_path):
    source = tmp_path / "source"
    _write_image(source / "train" / "air" / "air (1).jpg")
    _write_image(source / "train" / "air" / "air (2).jpg")
    _write_image(source / "test" / "air" / "air (301).jpg")
    _write_image(source / "train" / "demam" / "demam (1).jpg")

    output = tmp_path / "output"
    prepare(source, output)

    assert sorted(p.name for p in output.iterdir()) == ["air", "demam"]

    air_samples = sorted(p.name for p in (output / "air").iterdir())
    assert air_samples == ["test_301", "train_1", "train_2"]
    assert list((output / "air" / "train_1").glob("*.jpg"))[0].name == "air (1).jpg"

    demam_samples = sorted(p.name for p in (output / "demam").iterdir())
    assert demam_samples == ["train_1"]
