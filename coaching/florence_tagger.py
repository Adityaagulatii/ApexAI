"""
Florence-2 scene tagger — replaces moondream for Stage 2 frame triage.

Uses microsoft/Florence-2-base (232M) on CPU.
Downloads model on first run (~900MB), cached to HuggingFace hub afterward.

Tasks used:
  <CAPTION>           — one-sentence scene description
  <OD>                — object detection (bounding boxes + labels)
  <DETAILED_CAPTION>  — richer description for reasoning
"""

import os
import re
import threading

_model_lock = threading.Lock()
_model_cache: tuple | None = None

from PIL import Image

MODEL_ID = "microsoft/Florence-2-base"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".hf_cache")

_VEHICLE_CLASSES = {
    "kart", "karting", "car", "racing car", "go-kart",
    "motorcycle", "bike", "motorbike", "cyclist", "bicycle", "person",
}

_OBSTACLE_CLASSES = {
    "car", "truck", "bus", "barrier", "cone", "tire", "wall",
    "person", "marshal", "flag", "cone", "hay bale",
}


def _load_model() -> tuple:
    """Load Florence-2 once per process using double-checked locking."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is None:
            print(f"[florence] Loading {MODEL_ID}...")
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            processor = AutoProcessor.from_pretrained(
                MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            model.eval()
            _model_cache = (model, processor)
            print("[florence] Model loaded.")
    return _model_cache


def _run_task(image: Image.Image, task: str) -> dict:
    """Run a Florence-2 task and return the parsed result dict."""
    import torch
    model, processor = _load_model()

    inputs = processor(text=task, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        raw, task=task, image_size=(image.width, image.height)
    )


def tag_frame(frame_path: str, sport: str = "karting") -> dict:
    """
    Run Florence-2 on a frame and return structured tags.

    Returns:
        {
          "caption":          str,
          "vehicle_visible":  bool,
          "in_corner":        bool,
          "obstacle_ahead":   bool,
          "detected_objects": list[str],
          "raw_caption":      str,
        }
    """
    try:
        image = Image.open(frame_path).convert("RGB")
    except Exception as e:
        print(f"[florence] Cannot open {frame_path}: {e}")
        return _empty_tags()

    # Task 1 — caption for corner/turn detection
    try:
        cap_result = _run_task(image, "<CAPTION>")
        caption = cap_result.get("<CAPTION>", "")
    except Exception as e:
        print(f"[florence] CAPTION failed: {e}")
        caption = ""

    # Task 2 — object detection for vehicles + obstacles
    try:
        od_result = _run_task(image, "<OD>")
        labels = [l.lower() for l in od_result.get("<OD>", {}).get("labels", [])]
    except Exception as e:
        print(f"[florence] OD failed: {e}")
        labels = []

    caption_lower = caption.lower()

    vehicle_visible = (
        any(v in caption_lower for v in ("kart", "bike", "motorcycle", "cyclist", "racer")) or
        any(l in _VEHICLE_CLASSES for l in labels)
    )

    in_corner = any(w in caption_lower for w in (
        "corner", "turn", "bend", "curve", "apex", "chicane", "hairpin",
        "lean", "tilted", "banking",
    ))

    obstacle_labels = [l for l in labels if l in _OBSTACLE_CLASSES]
    obstacle_ahead = len(obstacle_labels) > 0 or any(
        w in caption_lower for w in ("ahead", "in front", "traffic", "flag", "yellow")
    )

    return {
        "caption":          caption,
        "vehicle_visible":  vehicle_visible,
        "in_corner":        in_corner,
        "obstacle_ahead":   obstacle_ahead,
        "detected_objects": list(set(labels)),
        "raw_caption":      caption,
    }


def _empty_tags() -> dict:
    return {
        "caption":          "",
        "vehicle_visible":  False,
        "in_corner":        False,
        "obstacle_ahead":   False,
        "detected_objects": [],
        "raw_caption":      "",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python florence_tagger.py <frame_path> [sport]")
        sys.exit(1)
    result = tag_frame(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "karting")
    import json
    print(json.dumps(result, indent=2))
