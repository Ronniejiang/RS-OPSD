#!/usr/bin/env python3
"""Read-only mixture audit; no image copies or rewritten manifests."""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
from PIL import Image

from verl.utils.dataset.pa_opd_dataset import _REMOVE_HINT, load_pa_opd_direct_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geoevidence-root", type=Path, required=True)
    parser.add_argument("--visionopd-root", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    for name, root, mode in (("geoevidence", args.geoevidence_root, "bbox_images"),
                             ("visionopd", args.visionopd_root, "native_bbox")):
        sources.append(load_pa_opd_direct_jsonl(root / "train.jsonl", student_image_mode=mode,
            teacher_full_image_mode=mode, separate_teacher_views=True, image_max_side=2048, source_name=name))
    mixed = datasets.concatenate_datasets(sources)
    counts = Counter()
    hints = Counter()
    paths = set()
    identities = set()
    for row in mixed:
        name = row["data_source"]
        counts[name] += 1
        hints[name] += _REMOVE_HINT in row["prompt"][0]["content"]
        images = row["images"] + row["teacher_images"]
        assert len(row["teacher_images"]) == 2
        assert row["images"][0] == row["teacher_images"][0]
        assert all(image["pa_opd_max_side"] == 2048 for image in images)
        paths.update(image["path"] for image in images)
        identity = (row["extra_info"]["source_jsonl"], row["extra_info"]["source_line"])
        assert identity not in identities
        identities.add(identity)

    def inspect(path):
        with Image.open(path) as image:
            width, height = image.size
            # Detect truncated/corrupt encoded payloads without writing pixels.
            image.verify()
        scale = min(1, 2048 / max(width, height))
        return (max(width, height), max(1, round(width * scale)), max(1, round(height * scale)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(inspect, sorted(paths)))
    assert all(max(width, height) <= 2048 for _, width, height in sizes)
    print(json.dumps({"source_rows": dict(counts), "total_rows": len(mixed),
        "box_hint_rows": dict(hints), "unique_verified_image_files": len(paths),
        "files_downscaled_at_read": sum(side > 2048 for side, _, _ in sizes),
        "mixing": "concatenate then shuffle, no oversampling", "batch_size": 32,
        "full_batches_per_epoch": len(mixed) // 32, "drop_last_rows_per_epoch": len(mixed) % 32,
        "max_read_side": max(max(width, height) for _, width, height in sizes)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
