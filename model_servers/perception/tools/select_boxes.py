"""Step 1 of context authoring: draw a box on each reference image.

Runs on the HOST, not in the perception-api container — it just needs a
display and plain opencv-python (with GUI support; the container's
opencv-python-headless has none). No torch/transformers/sam3 required here.

    pip install opencv-python
    python select_boxes.py --label tent --images tent1.jpg tent2.jpg tent3.jpg

For each image a window opens — drag a box around the object, press ENTER/SPACE
to confirm (or `c` to skip that image), then it moves to the next one. Writes
concepts/<label>.boxes.json: [{"image": "<abs path>", "box": [x1,y1,x2,y2]}, ...]

Step 2 (embed_concept.py) turns this into the actual concept pack — run that
one inside the perception-api container, where the model stack lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="concept name, e.g. tent")
    ap.add_argument("--images", nargs="+", required=True, help="reference image paths")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "concepts"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes_path = out_dir / f"{args.label}.boxes.json"

    entries = []
    for image_path in args.images:
        image_path = str(Path(image_path).resolve())
        img = cv2.imread(image_path)
        if img is None:
            print(f"!! could not read {image_path}, skipping")
            continue

        win = f"select box: {args.label} — drag, ENTER to confirm, c to skip"
        x, y, w, h = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)

        if w == 0 or h == 0:
            print(f"skipped {image_path}")
            continue

        box = [int(x), int(y), int(x + w), int(y + h)]
        entries.append({"image": image_path, "box": box})
        print(f"{image_path}: box={box}")

    boxes_path.write_text(json.dumps(entries, indent=2))
    print(f"\nwrote {len(entries)} boxes -> {boxes_path}")
    print(f"next: run embed_concept.py --boxes {boxes_path} --label {args.label} (in the perception-api container)")


if __name__ == "__main__":
    main()
