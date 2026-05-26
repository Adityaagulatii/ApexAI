import os
import json
import glob

from ultralytics import YOLO

CV_DIR = os.path.dirname(__file__)
FRAMES_DIR = os.path.join(CV_DIR, "frames")
DETECTIONS_OUT = os.path.join(CV_DIR, "detections.json")

KARTING_CLASSES = {"car", "sports car", "truck"}
BIKING_CLASSES = {"motorcycle", "person"}
ALL_TARGET_CLASSES = KARTING_CLASSES | BIKING_CLASSES


def detect(sport: str = "karting") -> list:
    target_classes = BIKING_CLASSES if sport == "biking" else KARTING_CLASSES

    model = YOLO("yolov8n.pt")

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {FRAMES_DIR}. Run ingest.py first.")

    print(f"[detect] Running YOLOv8n on {len(frame_paths)} frames (sport={sport})...")

    all_detections = []

    for idx, frame_path in enumerate(frame_paths):
        timestamp = round(idx / 10.0, 1)
        results = model.track(frame_path, tracker="bytetrack.yaml", persist=True, verbose=False)

        boxes_data = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id].lower()
                if cls_name not in ALL_TARGET_CLASSES:
                    continue
                if cls_name not in target_classes:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                boxes_data.append({
                    "id": track_id,
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 3),
                })

        all_detections.append({
            "frame_idx": idx,
            "timestamp": timestamp,
            "frame_path": frame_path,
            "boxes": boxes_data,
        })

        if idx % 50 == 0:
            print(f"[detect] Processed {idx}/{len(frame_paths)} frames...")

    with open(DETECTIONS_OUT, "w") as f:
        json.dump(all_detections, f, indent=2)

    total_boxes = sum(len(d["boxes"]) for d in all_detections)
    print(f"[detect] Done. {total_boxes} detections saved to {DETECTIONS_OUT}")
    return all_detections


if __name__ == "__main__":
    import sys
    sport = sys.argv[1] if len(sys.argv) > 1 else "karting"
    detect(sport)
