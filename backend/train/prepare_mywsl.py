"""Reorganizes the MyWSL2023 dataset into the <root>/<class>/<sample>/*.jpg
layout expected by train.dataset.SignDataset.

Source: MyWSL2023 - Malaysian Words Sign Language Dataset (Johari et al.,
2023), Mendeley Data, CC BY 4.0. https://data.mendeley.com/datasets/zvk55p7ktd

Download and extract the "MyWSL2023 CROP DATA.zip" file from that page first;
it contains train/ and test/ folders, each with one subfolder per class
(air, demam, dengar, makan, minum, salah, saya, senyap, tidur, waktu) holding
that class's static hand-gesture photos. Every MyWSL sample is a single
photo rather than a motion sequence, so each output sample folder holds
exactly one frame; SignDataset already handles single-frame samples by
left-padding the rest of the sequence.
"""

import argparse
import re
import shutil
from pathlib import Path

SAMPLE_INDEX_RE = re.compile(r"\((\d+)\)")


def prepare(source_root, output_root):
    source_root = Path(source_root)
    output_root = Path(output_root)

    for split_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.glob("*.jpg")):
                match = SAMPLE_INDEX_RE.search(image_path.stem)
                sample_id = match.group(1) if match else image_path.stem
                sample_dir = output_root / class_dir.name / f"{split_dir.name}_{sample_id}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, sample_dir / image_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", help="Extracted 'MyWSL2023 CROP DATA' folder (contains train/ and test/)")
    parser.add_argument("output_root", help="Destination laid out as <root>/<class>/<sample>/*.jpg")
    args = parser.parse_args()

    prepare(args.source_root, args.output_root)
