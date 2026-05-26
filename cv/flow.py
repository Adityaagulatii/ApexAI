import os
import json
import math

import cv2
import numpy as np

CV_DIR = os.path.dirname(__file__)
DETECTIONS_PATH = os.path.join(CV_DIR, "detections.json")
EVENTS_OUT = os.path.join(CV_DIR, "events.json")
VIDEO_PATH = os.path.join(CV_DIR, "video.mp4")


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _lean_angle_mediapipe(frame_bgr):
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                return None
            lm = res.pose_landmarks.landmark
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
            shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            return round(angle, 1)
    except Exception:
        return None


def analyze(sport: str = "karting", fps: int = 10) -> dict:
    with open(DETECTIONS_PATH) as f:
        detections = json.load(f)

    if not detections:
        raise ValueError("detections.json is empty.")

    frame_h, frame_w = None, None
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    if frame_w is None or frame_w == 0:
        frame_w, frame_h = 1280, 720
    diagonal = math.sqrt(frame_w ** 2 + frame_h ** 2)

    # Build per-track history: {track_id: [(frame_idx, centroid), ...]}
    track_history: dict[int, list] = {}
    for det in detections:
        for box in det["boxes"]:
            tid = box["id"]
            cx, cy = _centroid(box["bbox"])
            track_history.setdefault(tid, []).append((det["frame_idx"], (cx, cy)))

    # Compute per-frame speeds for each track
    # speed[frame_idx][track_id] = normalized speed
    speed_map: dict[int, dict[int, float]] = {}
    for tid, history in track_history.items():
        for i in range(1, len(history)):
            fidx_prev, c_prev = history[i - 1]
            fidx_curr, c_curr = history[i]
            raw_speed = _dist(c_prev, c_curr) / diagonal
            speed_map.setdefault(fidx_curr, {})[tid] = raw_speed

    events = []

    for tid, history in track_history.items():
        if len(history) < 6:
            continue
        for i in range(5, len(history)):
            fidx = history[i][0]
            # collect speeds over last 5 frames for this track
            speeds = []
            for j in range(i - 4, i + 1):
                f = history[j][0]
                s = speed_map.get(f, {}).get(tid, 0.0)
                speeds.append(s)

            if speeds[0] < 1e-6:
                continue

            speed_drop = (speeds[0] - speeds[-1]) / speeds[0]
            centroid = history[i][1]
            frame_path = detections[min(fidx, len(detections) - 1)]["frame_path"]
            timestamp = round(fidx / fps, 1)

            event_type = None
            if speed_drop > 0.30:
                # late braking: sharp deceleration
                event_type = "late_braking"
            elif speed_drop < -0.20 and speeds[0] < 0.02:
                # was slow then accelerated — early_braking heuristic
                event_type = "early_braking"

            if event_type is None:
                continue

            lean_angle = None
            if sport == "biking" and event_type in ("late_braking", "early_braking"):
                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is not None:
                    lean_angle = _lean_angle_mediapipe(frame_bgr)

            events.append({
                "frame": fidx,
                "timestamp": timestamp,
                "type": event_type,
                "speed_delta": round((speeds[-1] - speeds[0]) * 1000, 2),
                "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
                "image_path": frame_path,
                "lean_angle": lean_angle,
            })

    # Also detect wide_exit and good_apex from trajectory curvature
    for tid, history in track_history.items():
        if len(history) < 10:
            continue
        for i in range(5, len(history) - 5):
            fidx = history[i][0]
            # simplified: if track goes to edge of frame, flag wide_exit
            cx, cy = history[i][1]
            if frame_w and (cx < frame_w * 0.05 or cx > frame_w * 0.95):
                timestamp = round(fidx / fps, 1)
                frame_path = detections[min(fidx, len(detections) - 1)]["frame_path"]
                events.append({
                    "frame": fidx,
                    "timestamp": timestamp,
                    "type": "wide_exit",
                    "speed_delta": 0.0,
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "image_path": frame_path,
                    "lean_angle": None,
                })

    events.sort(key=lambda e: e["frame"])

    # Deduplicate nearby events of same type (within 10 frames)
    deduped = []
    last_by_type: dict[str, int] = {}
    for ev in events:
        last = last_by_type.get(ev["type"], -100)
        if ev["frame"] - last > 10:
            deduped.append(ev)
            last_by_type[ev["type"]] = ev["frame"]

    output = {
        "video_path": VIDEO_PATH,
        "sport": sport,
        "fps": fps,
        "events": deduped,
    }

    with open(EVENTS_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[flow] Done. {len(deduped)} events written to {EVENTS_OUT}")
    return output


if __name__ == "__main__":
    import sys
    sport = sys.argv[1] if len(sys.argv) > 1 else "karting"
    analyze(sport)
