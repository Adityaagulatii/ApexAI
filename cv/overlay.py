import os
import json
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np

CV_DIR = os.path.dirname(__file__)
DETECTIONS_PATH = os.path.join(CV_DIR, "detections.json")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
VIDEO_IN = os.path.join(CV_DIR, "video.mp4")
VIDEO_OUT = os.path.join(CV_DIR, "output_overlay.mp4")

TRAIL_LEN = 40  # recent frames for speed-colored gradient
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _speed_color(speed: float, diag: float) -> tuple:
    """Green = fast, red = slow — shows deceleration zones on racing line."""
    ref = diag * 0.008
    t = min(speed / ref, 1.0) if ref > 0 else 0.0
    r = int((1.0 - t) * 220)
    g = int(t * 220 + (1.0 - t) * 60)
    return (0, g, r)  # BGR


def render(sport: str | None = None, video_in: str | None = None,
           video_out: str | None = None) -> None:
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)

    with open(EVENTS_PATH) as f:
        events_data = json.load(f)

    if sport is None:
        sport = events_data.get("sport", "karting")
    events = events_data.get("events", [])
    event_frames = {e["frame"]: e for e in events}

    if video_in is None:
        video_in = VIDEO_IN
    if video_out is None:
        video_out = os.path.join(CV_DIR, f"{sport}_overlay.mp4")

    video_tmp = video_out.replace(".mp4", "_tmp.mp4")

    # Index detections by frame
    det_by_frame = {d["frame_idx"]: d for d in detections}

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diag = (w**2 + h**2) ** 0.5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_tmp, fourcc, fps, (w, h))

    # full_trail: cumulative path for the ghost racing line
    full_trail: dict[int, list] = defaultdict(list)
    # recent_trail: last TRAIL_LEN points for speed-colored gradient
    recent_trail: dict[int, list] = defaultdict(list)
    prev_centroids: dict[int, tuple] = {}
    speed_history: dict[int, list] = defaultdict(list)

    frame_idx = 0
    print(f"[overlay] Rendering {total_frames} frames for {sport}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = det_by_frame.get(frame_idx, {"boxes": []})
        event = event_frames.get(frame_idx)

        current_speeds: dict[int, float] = {}
        for box in det["boxes"]:
            tid = box["id"]
            cx, cy = _centroid(box["bbox"])

            if tid in prev_centroids:
                dx = cx - prev_centroids[tid][0]
                dy = cy - prev_centroids[tid][1]
                spd = (dx**2 + dy**2) ** 0.5
                current_speeds[tid] = spd
                speed_history[tid].append(spd)
            else:
                speed_history[tid].append(0.0)

            full_trail[tid].append((cx, cy))
            recent_trail[tid].append((cx, cy, speed_history[tid][-1]))
            if len(recent_trail[tid]) > TRAIL_LEN:
                recent_trail[tid].pop(0)
            prev_centroids[tid] = (cx, cy)

        # Draw ghost full racing line (semi-transparent dark overlay)
        overlay = frame.copy()
        for tid, pts in full_trail.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i - 1], pts[i], (60, 60, 120), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw speed-colored recent trail (green=fast, red=slow)
        for tid, pts in recent_trail.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                color = _speed_color(pts[i][2], diag)
                thickness = 3 if i > len(pts) - 5 else 2
                cv2.line(frame, pts[i - 1][:2], pts[i][:2], color, thickness, cv2.LINE_AA)
            # Draw dot at current position
            if pts:
                cv2.circle(frame, pts[-1][:2], 5, WHITE, -1, cv2.LINE_AA)

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

    print("[overlay] Re-encoding to H.264 for browser compatibility...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_tmp,
        "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p", video_out,
    ], capture_output=True, check=True)
    os.remove(video_tmp)
    print(f"[overlay] Done. Output: {video_out}")


if __name__ == "__main__":
    sport_arg = sys.argv[1] if len(sys.argv) > 1 else None
    video_in_arg = sys.argv[2] if len(sys.argv) > 2 else None
    render(sport=sport_arg, video_in=video_in_arg)
