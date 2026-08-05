"""In-memory perception: SAM 3 (+ DINOv3, + optionally YOLOE-26) loaded once,
called in-process.

This is the real-time hot path. It is imported by the backend, NOT served over
HTTP on the live path (see ../README.md §1.3). Two backends hide behind one
interface so SAM 3 and a SAM 2.1 + GroundingDINO fallback are interchangeable.

SAM 3's checkpoint is gated on Hugging Face (facebookresearch/sam3) — accept
access and run `huggingface-cli login` on the HOST during the download step
(scripts/download_models.sh), before weights are mounted read-only here.

The `sam3` package (0.1.0) has no stable public API docs at the time this was
written; the calls below were verified against the installed package's source
directly, not guessed from examples. If `sam3` is upgraded, re-verify with:

    docker compose run --rm --entrypoint python3 perception-api -c "
    import inspect
    from sam3.model.sam3_image_processor import Sam3Processor
    print(inspect.signature(Sam3Processor.set_image))"

`Sam3Processor.add_geometric_prompt(box, label, state)` supports exemplar-box
prompting alongside text prompts in one grounding pass, but isn't used here —
this project's exemplar matching goes through DINOv3 similarity instead (see
../README.md's "Context authoring" section for why).

## YOLOE-26 (optional, exemplar/visual-prompt recall)

SAM 3's text-prompt recall (above) is a hard bottleneck for visually unique
targets that no text phrasing describes well: it's the *only* thing that
proposes candidate boxes, and DINOv3 only ever filters what SAM 3 already
found. YOLOE-26 (Ultralytics) adds a second recall path, driven by an
exemplar *image* instead of text — see `detect_by_exemplar` below and
`app/config.py`'s `exemplar_recall_backend`.

Verified against ultralytics==8.4.115 (`docker compose run --rm --entrypoint
python3 backend -c "import inspect; from ultralytics import YOLOE;
print(inspect.getsource(YOLOE.predict))"`) — `YOLOE.predict(source, ...,
visual_prompts=None, refer_image=None, predictor=...)`: when `refer_image`
is given, it calls `self.predictor.get_vpe(refer_image)` (using the boxes
already registered via `set_prompts`, in `refer_image`'s own absolute pixel
coordinates — not normalized) to get a visual-prompt embedding, bakes it
into the model via `self.model.set_classes(...)`, resets `self.predictor`,
then still runs inference on `source` in that same call — so one call both
bakes the prompt and returns results for the current frame, and later plain
`.predict(frame)` calls (no `refer_image`) reuse the baked classes.

`refer_image` is a single image, not a list — the "multiple images" branch
in `predict()`'s source only applies when *source* (the target, not the
reference) is a list, a separate case. There's no code path for fusing
boxes from multiple *separate* reference images into one baked prompt, so
`detect_by_exemplar` only bakes the most recently authored exemplar per
concept — confirmed limitation, not an assumption.
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils


@dataclass
class Detection:
    track_id: int
    concept: str
    box: tuple[int, int, int, int]      # x1, y1, x2, y2
    mask_rle: str                       # COCO RLE
    score: float
    embedding: list[float] | None = None


@dataclass
class Concept:
    label: str
    priority: str = "watch"             # alert | watch | ignore
    text_prompt: str | None = None
    exemplar_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def _encode_mask_rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def _decode_exemplar_image(data_url: str) -> Image.Image:
    _, _, b64data = data_url.partition(",")
    raw = base64.b64decode(b64data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class Perception:
    """Loads SAM 3 + DINOv3 (+ optionally YOLOE-26) onto the GPU once;
    exposes in-process inference."""

    def __init__(self, backend: str, sam_weights: str, dinov3_weights: str,
                 device: str = "cuda", exemplar_recall_backend: str = "sam3",
                 yoloe_weights: str | None = None) -> None:
        self.backend = backend
        self.device = device
        self.sam_weights = sam_weights
        self.dinov3_weights = dinov3_weights
        self.exemplar_recall_backend = exemplar_recall_backend
        self.yoloe_weights = yoloe_weights
        self._next_track_id = 0
        self._yoloe_model = None
        # concept_label -> id() of the Exemplar object last baked into
        # YOLOE for it, so a re-bake only happens when the exemplar set for
        # that concept actually changed (see detect_by_exemplar).
        self._yoloe_baked: dict[str, int] = {}
        self._load()

    @classmethod
    def from_env(cls) -> "Perception":
        return cls(
            backend=os.environ.get("PERCEPTION_BACKEND", "sam3"),
            sam_weights=os.environ.get("SAM_WEIGHTS", "/models/sam3"),
            dinov3_weights=os.environ.get("DINOV3_WEIGHTS", "/models/dinov3-vitl16"),
            exemplar_recall_backend=os.environ.get("EXEMPLAR_RECALL_BACKEND", "both"),
            yoloe_weights=os.environ.get("YOLOE_WEIGHTS", "/models/yoloe-26/yoloe-26s-seg.pt"),
        )

    def _load(self) -> None:
        if self.backend == "sam3":
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # load_from_HF=True (the default) downloads sam3.pt over the network
            # when checkpoint_path is None — point it at the weights already on
            # disk instead, so this never touches the network at runtime.
            checkpoint_path = os.path.join(self.sam_weights, "sam3.pt")
            self._sam_model = build_sam3_image_model(
                checkpoint_path=checkpoint_path, load_from_HF=False
            )
            self._sam_processor = Sam3Processor(self._sam_model)
        elif self.backend == "sam2_gdino":
            raise NotImplementedError(
                "sam2_gdino fallback not wired up. Load SAM 2.1 "
                "(facebookresearch/sam2) + GroundingDINO/YOLO-World here if "
                "SAM 3 checkpoint access isn't available — see README.md."
            )
        else:
            raise ValueError(f"unknown PERCEPTION_BACKEND: {self.backend}")

        from transformers import AutoImageProcessor, AutoModel

        self._dino_processor = AutoImageProcessor.from_pretrained(self.dinov3_weights)
        self._dino_model = AutoModel.from_pretrained(
            self.dinov3_weights, device_map=self.device, attn_implementation="sdpa"
        ).eval()

        # Only load YOLOE-26 when the setting could actually call it — no
        # reason to spend GPU memory on a model that's guaranteed unused.
        if self.exemplar_recall_backend in ("yoloe26", "both"):
            from ultralytics import YOLOE

            self._yoloe_model = YOLOE(self.yoloe_weights)
            if self.device == "cuda":
                self._yoloe_model.to(self.device)

    def detect_track(self, frame: np.ndarray, concepts: list[Concept],
                     want_masks: bool = True) -> list[Detection]:
        """Tier-1: detect + segment every concept instance in `frame`.

        `frame` is a decoded HxWx3 RGB array handed straight from the pipeline —
        no encode/decode round-trip.

        `want_masks=False` skips RLE-encoding the segmentation masks. The live
        pipeline only consumes boxes and scores, and encoding a full-resolution
        mask per detection per frame is expensive — so the backend passes False.
        The default stays True so existing callers (service.py's debug API) keep
        their exact output.

        NOTE: this calls SAM 3's single-image API fresh per frame, so
        `track_id` here is just an incrementing counter, NOT a real persistent
        identity. For real cross-frame tracking, swap to
        `sam3.model_builder.build_sam3_video_predictor` + `handle_request()`
        with one `inference_state` per stream, kept alive across calls — that's
        what actually maintains track IDs. `backend/app/tracker.py` currently
        papers over this with greedy IOU matching.
        """
        # SAM 3's weights are bfloat16; its own predictor code always runs inference
        # inside torch.autocast(dtype=bfloat16) (see sam3_base_predictor.py) — without
        # it, set_image()'s float32 input mismatches the model's dtype.
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with autocast:
            return self._detect_track_sam3(frame, concepts, want_masks)

    def _detect_track_sam3(self, frame: np.ndarray, concepts: list[Concept],
                           want_masks: bool) -> list[Detection]:
        detections: list[Detection] = []
        # set_image()'s ndarray/tensor branch reads `height, width = image.shape[-2:]`,
        # which assumes channel-first (C,H,W); our frames are channel-last (H,W,C), so
        # that branch silently reads (width, channels) as (height, width) and produces
        # garbage boxes. The PIL.Image branch reads `.size` correctly regardless.
        pil_frame = Image.fromarray(frame)

        # ONE vision-backbone pass for the whole frame, reused by every concept.
        # set_image() runs the (expensive) image encoder and stores the result in
        # state["backbone_out"]; set_text_prompt() then merges the text features
        # into that same dict and runs only the grounding decoder. Its own source
        # comment — "will erase the previous text prompt if any" — confirms
        # calling it repeatedly against one state is the intended pattern.
        #
        # This used to sit inside the concept loop, so an N-concept watch-list
        # paid for N identical full-frame encodes per detection pass. The
        # backbone is prompt-independent, so that was pure waste.
        state = self._sam_processor.set_image(pil_frame)

        for concept in concepts:
            if not concept.text_prompt:
                continue
            if concept.exemplar_boxes:
                # Not wired up — see module docstring for why (DINOv3 routing
                # instead). Would look like:
                #   for box in concept.exemplar_boxes:
                #       state = self._sam_processor.add_geometric_prompt(
                #           box=list(box), label=True, state=state)
                pass
            state = self._sam_processor.set_text_prompt(prompt=concept.text_prompt, state=state)

            # One device sync per concept rather than three-or-four per
            # detection: pull the whole tensors across at once.
            boxes = state["boxes"].cpu().tolist()
            scores = state["scores"].cpu().tolist()
            masks = state["masks"] if want_masks else None

            for i, (box_xyxy, score) in enumerate(zip(boxes, scores)):
                self._next_track_id += 1
                mask_rle = ""
                if masks is not None:
                    mask_rle = _encode_mask_rle(masks[i, 0].cpu().numpy())
                detections.append(Detection(
                    track_id=self._next_track_id,
                    concept=concept.label,
                    box=tuple(int(v) for v in box_xyxy),
                    mask_rle=mask_rle,
                    score=float(score),
                ))
        return detections

    def detect_by_exemplar(self, frame: np.ndarray, concept_label: str,
                           exemplars: list) -> list[Detection]:
        """Tier-1 recall via YOLOE-26 visual prompting — complementary to
        detect_track's SAM 3 text-prompt recall, see app/config.py's
        exemplar_recall_backend. `exemplars` is ConceptRecord.exemplars (see
        app/store.py): objects with `.image` (base64 JPEG data URL) and
        `.box`.

        Only the most recently authored exemplar is baked in per concept —
        see this module's docstring for why multi-exemplar fusion isn't
        attempted (the documented API demonstrates one reference image at a
        time, not fusing several).
        """
        if self._yoloe_model is None or not exemplars:
            return []

        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        pil_frame = Image.fromarray(frame)
        exemplar = exemplars[-1]
        # id(), not equality — Exemplar is frozen/immutable, so a changed
        # exemplar set always produces a new object even if a value happens
        # to repeat; cheap and sufficient as a "did this change" check.
        cache_key = id(exemplar)

        if self._yoloe_baked.get(concept_label) != cache_key:
            ref_image = _decode_exemplar_image(exemplar.image)
            visual_prompts = {
                "bboxes": np.array([exemplar.box], dtype=np.float32),
                "cls": np.array([0]),
            }
            results = self._yoloe_model.predict(
                pil_frame, refer_image=ref_image, visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor, verbose=False,
            )
            self._yoloe_baked[concept_label] = cache_key
        else:
            results = self._yoloe_model.predict(pil_frame, verbose=False)

        return self._yoloe_results_to_detections(results, concept_label)

    def _yoloe_results_to_detections(self, results, concept_label: str) -> list[Detection]:
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []
        detections: list[Detection] = []
        for box_xyxy, score in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
            self._next_track_id += 1
            detections.append(Detection(
                track_id=self._next_track_id,
                concept=concept_label,
                box=tuple(int(v) for v in box_xyxy),
                mask_rle="",
                score=float(score),
            ))
        return detections

    def embed(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Attach a DINOv3 embedding to each detection's crop (in-place)."""
        if not detections:
            return detections

        crops = [self._crop(frame, d.box) for d in detections]
        inputs = self._dino_processor(images=crops, return_tensors="pt").to(self._dino_model.device)
        with torch.inference_mode():
            out = self._dino_model(**inputs)
        vectors = out.pooler_output.float().cpu().numpy()

        for det, vec in zip(detections, vectors):
            det.embedding = vec.tolist()
        return detections

    @staticmethod
    def _crop(frame: np.ndarray, box: tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        return Image.fromarray(frame[y1:y2, x1:x2])
