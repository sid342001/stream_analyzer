"""Step 3: run concept packs against a real frame and see what matches.

Runs in the perception-api container, same as embed_concept.py:

    docker compose exec perception-api python tools/test_inference.py \
        --frame sample.jpg --concepts concepts/tent.json concepts/person.json \
        --out annotated.jpg

For each concept: SAM3 does broad recall on the concept's text prompt, DINOv3
embeds every candidate, and each candidate is kept only if it's similar enough
to that concept's reference embeddings (cosine similarity >= --threshold).
This is the "does it actually look like my examples" gate described in
../README.md and is why a generic text prompt doesn't just flood the output
with false positives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interface import Concept, Perception  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="test image / video frame")
    ap.add_argument("--concepts", nargs="+", required=True, help="concept pack JSON files")
    ap.add_argument("--threshold", type=float, default=0.55, help="min cosine similarity to count as a match")
    ap.add_argument("--out", default="annotated.jpg")
    args = ap.parse_args()

    frame = np.array(Image.open(args.frame).convert("RGB"))
    percept = Perception.from_env()

    packs = [json.loads(Path(p).read_text()) for p in args.concepts]
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    all_matches = []

    for pack in packs:
        concept = Concept(label=pack["label"], priority=pack["priority"], text_prompt=pack["text_prompt"])
        candidates = percept.detect_track(frame, [concept])   # SAM3: broad recall, text prompt only
        candidates = percept.embed(frame, candidates)          # DINOv3: embed every candidate

        refs = np.array(pack["reference_embeddings"])
        for det in candidates:
            sim = max(cosine(np.array(det.embedding), ref) for ref in refs)
            if sim < args.threshold:
                continue
            all_matches.append({"concept": pack["label"], "box": det.box, "similarity": round(sim, 3)})
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(canvas, f"{pack['label']} {sim:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    cv2.imwrite(args.out, canvas)
    print(json.dumps(all_matches, indent=2))
    print(f"\n{len(all_matches)} match(es) above threshold {args.threshold} -> {args.out}")


if __name__ == "__main__":
    main()
