import os
import json
from collections import defaultdict

import cv2
import numpy as np

CV_DIR = os.path.dirname(__file__)
DETECTIONS_PATH = os.path.join(CV_DIR, "detections.json")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
VIDEO_IN = os.path.join(CV_DIR, "video.mp4")
VIDEO_OUT = os.path.join(CV_DIR, "output_overlay.mp4")

TRAIL_LEN = 15
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def render() -> None:
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)

    with open(EVENTS_PATH) as f:
        events_data = json.load(f)

    sport = events_data.get("sport", "karting")
    events = events_data.get("events", [])
    event_frames = {e["frame"]: e for e in events}

    # Index detections by frame
    det_by_frame = {d["frame_idx"]: d for d in detections}

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

    # Track centroid history per track id
    trail: dict[int, list] = defaultdict(list)

    # Build speed map from detections for display
    prev_centroids: dict[int, tuple] = {}
    diag = (w**2 + h**2) ** 0.5

    frame_idx = 0
    print(f"[overlay] Rendering {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = det_by_frame.get(frame_idx, {"boxes": []})
        event = event_frames.get(frame_idx)

        # Update trails and compute speed per track
        current_speeds: dict[int, float] = {}
        for box in det["boxes"]:
            tid = box["id"]
            cx, cy = _centroid(box["bbox"])
            trail[tid].append((cx, cy))
            if len(trail[tid]) > TRAIL_LEN:
                trail[tid].pop(0)

            if tid in prev_centroids:
                dx = cx - prev_centroids[tid][0]
                dy = cy - prev_centroids[tid][1]
                spd = (dx**2 + dy**2) ** 0.5 / diag
                current_speeds[tid] = spd
            prev_centroids[tid] = (cx, cy)

        # Draw trails (racing line)
        for tid, pts in trail.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], BLUE, 2, cv2.LINE_AA)

        # Draw bounding boxes
        for box in det["boxes"]:
            tid = box["id"]
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            label = f"ID{tid} {box['class']}"
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

        # Draw event marker
        if event:
            cx, cy = int(event["centroid"][0]), int(event["centroid"][1])
            ev_type = event["type"]
            if ev_type == "late_braking":
                cv2.circle(frame, (cx, cy), 18, RED, -1)
                cv2.putText(frame, "! LATE BRAKE", (cx + 22, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
            elif ev_type == "early_braking":
                cv2.circle(frame, (cx, cy), 18, YELLOW, -1)
                cv2.putText(frame, "EARLY BRAKE", (cx + 22, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            elif ev_type == "wide_exit":
                cv2.circle(frame, (cx, cy), 18, (0, 165, 255), -1)
                cv2.putText(frame, "WIDE EXIT", (cx + 22, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            elif ev_type == "good_apex":
                cv2.circle(frame, (cx, cy), 18, GREEN, -1)
                cv2.putText(frame, "GOOD APEX", (cx + 22, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)

            # Lean angle for biking
            if sport == "biking" and event.get("lean_angle") is not None:
                cv2.putText(frame, f"Lean: {event['lean_angle']}°",
                            (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

        # Speed HUD top-left
        if current_speeds:
            avg_spd = sum(current_speeds.values()) / len(current_speeds)
            kmh_est = int(avg_spd * 3000)
            cv2.putText(frame, f"~{kmh_est} km/h", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

        # Watermark bottom-right
        timestamp = round(frame_idx / fps, 1)
        watermark = f"{sport.upper()}  {timestamp}s"
        (tw, th), _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, watermark, (w - tw - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"[overlay] {frame_idx}/{total_frames} frames rendered...")

    cap.release()
    writer.release()
    print(f"[overlay] Done. Output: {VIDEO_OUT}")


if __name__ == "__main__":
    render()
