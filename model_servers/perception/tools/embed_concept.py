"""Step 2 of context authoring: turn boxes.json into a real concept pack.

Runs where the model stack lives — inside the perception-api container:

    docker compose --profile debug up -d perception-api
    docker compose exec perception-api python tools/embed_concept.py \
        --boxes concepts/tent.boxes.json --label tent --priority watch \
        --text-prompt tent

Loads each reference image, crops the box you drew, runs it through DINOv3
(reusing Perception.embed — a reference crop is treated as a single fake
detection so no extra model code is needed), and writes concepts/<label>.json:
the concept pack test_inference.py and the future backend both consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interface import Concept, Detection, Perception  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="boxes.json from select_boxes.py")
    ap.add_argument("--label", required=True)
    ap.add_argument("--priority", default="watch", choices=["alert", "watch", "ignore"])
    ap.add_argument("--text-prompt", default=None, help="defaults to --label")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "concepts"))
    args = ap.parse_args()

    entries = json.loads(Path(args.boxes).read_text())
    if not entries:
        raise SystemExit(f"no boxes in {args.boxes} — nothing to embed")

    percept = Perception.from_env()

    embeddings: list[list[float]] = []
    for entry in entries:
        frame = np.array(Image.open(entry["image"]).convert("RGB"))
        box = tuple(entry["box"])
        fake_det = Detection(track_id=0, concept=args.label, box=box, mask_rle="", score=1.0)
        [embedded] = percept.embed(frame, [fake_det])
        embeddings.append(embedded.embedding)
        print(f"embedded {entry['image']} box={box} -> {len(embedded.embedding)}-d vector")

    pack = {
        "label": args.label,
        "priority": args.priority,
        "text_prompt": args.text_prompt or args.label,
        "reference_embeddings": embeddings,
        "source_images": [e["image"] for e in entries],
    }

    out_path = Path(args.out_dir) / f"{args.label}.json"
    out_path.write_text(json.dumps(pack, indent=2))
    print(f"\nwrote concept pack ({len(embeddings)} exemplars) -> {out_path}")


if __name__ == "__main__":
    main()
