"""
cv/racing_line.py — Racing line analysis from centroid trajectory.

Standalone module: reads detections.json → writes racing_line.json.

Pipeline:
  Stage A  Trajectory extraction  — build ordered centroid sequence
  Stage B  Smoothing              — moving-average to kill noise
  Stage C  Corner detection       — heading-change threshold
  Stage D  Per-corner analysis    — apex, entry, exit, errors
  Stage E  Line consistency       — lateral variance across session

Outputs racing_line.json with:
  - per-corner events (errors + good moments) with timestamps
  - session-level metrics (consistency score, apex tightness, etc.)
"""

import os
import json
import math
from collections import defaultdict

CV_DIR         = os.path.dirname(__file__)
DETECTIONS_IN  = os.path.join(CV_DIR, "detections.json")
EVENTS_IN      = os.path.join(CV_DIR, "events.json")
RACING_LINE_OUT = os.path.join(CV_DIR, "racing_line.json")

# ── tuning constants ───────────────────────────────────────────────────────────
SMOOTH_WINDOW       = 7     # moving-average window for trajectory smoothing
CORNER_ANGLE_DEG    = 22    # min heading change (°) to enter a corner (raised: onboard camera exaggerates small turns)
MIN_CORNER_FRAMES   = 8     # ignore micro-wiggles shorter than this
APEX_EARLY_THRESH   = 0.35  # apex in first 35% of corner = early (bad)
APEX_LATE_THRESH    = 0.55  # apex after 55% = late (good in racing)
INSIDE_TIGHT_PCT    = 0.32  # within 32% of inside edge = tight apex (good); raised for camera foreshortening
INSIDE_MISS_PCT     = 0.60  # never within 60% of inside = missed apex (bad); high threshold = only clear misses
OUTSIDE_ENTRY_PCT   = 0.12  # within 12% of outside edge = good wide entry
OUTSIDE_EXIT_PCT    = 0.85  # beyond 85% to outside after apex = wide exit (bad)
CONSISTENCY_BUCKETS = 8     # lateral quartile buckets for consistency score
MAX_EVENTS_PER_TYPE = 6     # cap per event type to prevent score collapse from repeated detections


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


# ── Stage A — Trajectory extraction ───────────────────────────────────────────

def _build_trajectory(detections: list[dict]) -> list[dict]:
    """
    Build a sorted list of {frame_idx, timestamp, cx, cy} from detections.
    When multiple tracks exist in a frame, prefer id=1 (main tracked object).
    """
    traj = []
    for det in detections:
        boxes = det.get("boxes", [])
        if not boxes:
            continue
        # prefer the stably-tracked id if present
        preferred = [b for b in boxes if b["id"] >= 0]
        box = preferred[0] if preferred else boxes[0]
        x1, y1, x2, y2 = box["bbox"]
        traj.append({
            "frame_idx": det["frame_idx"],
            "timestamp": det["timestamp"],
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })
    return sorted(traj, key=lambda p: p["frame_idx"])


# ── Stage B — Smoothing ────────────────────────────────────────────────────────

def _smooth(traj: list[dict], w: int) -> list[dict]:
    """Moving-average smooth on cx and cy."""
    out = []
    n = len(traj)
    for i, pt in enumerate(traj):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        window = traj[lo:hi]
        out.append({
            **pt,
            "cx": sum(p["cx"] for p in window) / len(window),
            "cy": sum(p["cy"] for p in window) / len(window),
        })
    return out


# ── Stage C — Corner detection ─────────────────────────────────────────────────

def _heading(p1: dict, p2: dict) -> float:
    """Heading angle in degrees from p1 to p2."""
    dx = p2["cx"] - p1["cx"]
    dy = p2["cy"] - p1["cy"]
    return math.degrees(math.atan2(dy, dx))


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest difference between two angles."""
    d = (b - a + 180) % 360 - 180
    return d


def _detect_corners(traj: list[dict]) -> list[dict]:
    """
    Return list of corner segments: {start, end, direction, frames}.
    direction: 'left' | 'right'
    """
    if len(traj) < 4:
        return []

    # Compute heading change at each point
    angle_changes = [0.0]
    for i in range(1, len(traj) - 1):
        if i < 2:
            angle_changes.append(0.0)
            continue
        h_prev = _heading(traj[i - 2], traj[i - 1])
        h_curr = _heading(traj[i - 1], traj[i])
        angle_changes.append(_angle_diff(h_prev, h_curr))
    angle_changes.append(0.0)

    # Segment into corners (consecutive frames with abs angle change > threshold)
    in_corner = False
    corner_start = 0
    corners = []
    for i, ac in enumerate(angle_changes):
        if not in_corner and abs(ac) > CORNER_ANGLE_DEG:
            in_corner = True
            corner_start = i
        elif in_corner and abs(ac) <= CORNER_ANGLE_DEG:
            in_corner = False
            if i - corner_start >= MIN_CORNER_FRAMES:
                seg = traj[corner_start:i]
                # direction: left = heading turns left (negative angle change)
                avg_ac = sum(angle_changes[corner_start:i]) / max(1, i - corner_start)
                corners.append({
                    "start":     corner_start,
                    "end":       i,
                    "direction": "left" if avg_ac < 0 else "right",
                    "frames":    seg,
                })
    return corners


# ── Stage D — Per-corner analysis ─────────────────────────────────────────────

def _analyze_corner(corner: dict, track_min_x: float, track_max_x: float,
                    fps: float) -> list[dict]:
    """
    Analyze one corner segment and return a list of racing line events.
    inside_x = track edge toward which the vehicle turns (apex side)
    outside_x = opposite edge (entry/exit side)
    """
    frames   = corner["frames"]
    n        = len(frames)
    if n < 2:
        return []

    direction  = corner["direction"]
    track_w    = max(track_max_x - track_min_x, 1)

    # For a LEFT turn: inside = left edge (min_x), outside = right edge (max_x)
    # For a RIGHT turn: inside = right edge (max_x), outside = left edge (min_x)
    if direction == "left":
        inside_x  = track_min_x
        outside_x = track_max_x
        def dist_to_inside(cx):  return (cx - inside_x)  / track_w
        def dist_to_outside(cx): return (outside_x - cx) / track_w
    else:
        inside_x  = track_max_x
        outside_x = track_min_x
        def dist_to_inside(cx):  return (outside_x - cx + track_w) / track_w  # actually (track_max_x - cx)/track_w
        def dist_to_outside(cx): return (cx - outside_x) / track_w

    # Redefine simply:
    if direction == "left":
        lateral_pct = [(f["cx"] - track_min_x) / track_w for f in frames]  # 0=left/inside, 1=right/outside
        inside_pct  = lambda p: p          # small = close to left inside
        outside_pct = lambda p: 1.0 - p   # small = close to right outside
    else:
        lateral_pct = [(f["cx"] - track_min_x) / track_w for f in frames]  # 0=left, 1=right/inside
        inside_pct  = lambda p: 1.0 - p   # small = close to right inside
        outside_pct = lambda p: p         # small = close to left outside

    # Apex: frame where vehicle is closest to inside
    apex_inside_dists = [inside_pct(p) for p in lateral_pct]
    apex_idx    = apex_inside_dists.index(min(apex_inside_dists))
    apex_frame  = frames[apex_idx]
    apex_pct_in_corner = apex_idx / max(n - 1, 1)
    apex_dist   = min(apex_inside_dists)

    # Entry: average of first 20% of corner frames
    entry_end  = max(1, int(n * 0.2))
    entry_pcts = lateral_pct[:entry_end]
    entry_outside_dist = sum(outside_pct(p) for p in entry_pcts) / len(entry_pcts)

    # Exit: average of last 20% of corner frames
    exit_start = min(n - 1, int(n * 0.8))
    exit_pcts  = lateral_pct[exit_start:]
    exit_outside_dist = sum(outside_pct(p) for p in exit_pcts) / len(exit_pcts)

    events = []

    def make_event(etype, frame, sev, desc):
        ts = frame["timestamp"]
        return {
            "frame":       frame["frame_idx"],
            "timestamp":   _fmt_ts(ts),
            "seconds":     int(ts),
            "type":        etype,
            "severity":    sev,
            "description": desc,
            "centroid":    [round(frame["cx"], 1), round(frame["cy"], 1)],
            "is_error":    etype not in ("good_apex", "good_entry", "consistent_line"),
        }

    # ── Apex classification ──────────────────────────────────────────────────
    if apex_dist < INSIDE_TIGHT_PCT:
        if apex_pct_in_corner > APEX_LATE_THRESH:
            events.append(make_event(
                "good_apex", apex_frame, "good",
                f"Late apex on {direction} turn — tight line at {apex_frame['timestamp']:.1f}s, "
                f"good exit position."
            ))
        else:
            events.append(make_event(
                "early_apex", apex_frame, "significant",
                f"Early apex on {direction} turn at {apex_frame['timestamp']:.1f}s — "
                f"hit apex in first {int(apex_pct_in_corner*100)}% of corner, likely causes wide exit."
            ))
    elif apex_dist > INSIDE_MISS_PCT:
        events.append(make_event(
            "missed_apex", apex_frame, "significant",
            f"Missed apex on {direction} turn at {apex_frame['timestamp']:.1f}s — "
            f"stayed {int(apex_dist*100)}% away from inside, losing time through corner."
        ))
    else:
        # Reasonable apex but check timing
        if apex_pct_in_corner < APEX_EARLY_THRESH:
            events.append(make_event(
                "early_apex", apex_frame, "minor",
                f"Slightly early apex on {direction} turn at {apex_frame['timestamp']:.1f}s — "
                f"aim to hold outside longer before turning in."
            ))

    # ── Entry classification ─────────────────────────────────────────────────
    if entry_outside_dist < OUTSIDE_ENTRY_PCT:
        events.append(make_event(
            "good_entry", frames[0], "good",
            f"Wide entry on {direction} turn at {frames[0]['timestamp']:.1f}s — "
            f"using outside edge correctly to maximise corner radius."
        ))
    elif entry_outside_dist > 0.4:
        events.append(make_event(
            "poor_entry", frames[0], "minor",
            f"Narrow entry on {direction} turn at {frames[0]['timestamp']:.1f}s — "
            f"not starting from outside edge, reducing geometric advantage."
        ))

    # ── Exit classification ──────────────────────────────────────────────────
    if exit_outside_dist < 0.15:
        events.append(make_event(
            "wide_exit", frames[-1], "significant",
            f"Wide exit on {direction} turn at {frames[-1]['timestamp']:.1f}s — "
            f"drifted to {int((1-exit_outside_dist)*100)}% of track width, losing straightline speed."
        ))

    return events


# ── Stage E — Line consistency ─────────────────────────────────────────────────

def _consistency_score(traj: list[dict], track_min_x: float, track_max_x: float) -> dict:
    """
    Score lateral consistency 0-100.
    Split session into CONSISTENCY_BUCKETS time slices.
    Measure std-dev of lateral position within each slice.
    Low std-dev = consistent = high score.
    """
    if not traj:
        return {"score": 65, "variance": 0.0}

    track_w = max(track_max_x - track_min_x, 1)
    n = len(traj)
    bucket_size = max(1, n // CONSISTENCY_BUCKETS)

    variances = []
    for b in range(CONSISTENCY_BUCKETS):
        sl = traj[b * bucket_size:(b + 1) * bucket_size]
        if len(sl) < 2:
            continue
        pcts = [(p["cx"] - track_min_x) / track_w for p in sl]
        mean = sum(pcts) / len(pcts)
        var  = sum((p - mean) ** 2 for p in pcts) / len(pcts)
        variances.append(var)

    if not variances:
        return {"score": 65, "variance": 0.0}

    avg_var = sum(variances) / len(variances)
    # variance 0 = perfect = 100, variance 0.1 = score ~0
    score = max(0, min(100, int(100 - avg_var * 1000)))
    return {"score": score, "variance": round(avg_var, 4)}


# ── Main ───────────────────────────────────────────────────────────────────────

def analyze(sport: str | None = None) -> dict:
    with open(DETECTIONS_IN, encoding="utf-8") as f:
        detections = json.load(f)
    with open(EVENTS_IN, encoding="utf-8") as f:
        events_data = json.load(f)

    if sport is None:
        sport = events_data.get("sport", "karting")
    fps = events_data.get("fps", 10)

    print(f"[racing_line] Analyzing trajectory for sport={sport}...")

    # Stage A — extract trajectory
    traj_raw = _build_trajectory(detections)
    if not traj_raw:
        print("[racing_line] No trajectory data — empty output.")
        out = {"sport": sport, "events": [], "metrics": {"line_score": 0, "consistency_score": 0}}
        with open(RACING_LINE_OUT, "w") as f:
            json.dump(out, f, indent=2)
        return out

    print(f"[racing_line] {len(traj_raw)} trajectory points extracted.")

    # Stage B — smooth
    traj = _smooth(traj_raw, SMOOTH_WINDOW)

    # Track bounds
    xs = [p["cx"] for p in traj]
    track_min_x = min(xs)
    track_max_x = max(xs)

    # Stage C — corner detection
    corners = _detect_corners(traj)
    print(f"[racing_line] {len(corners)} corners detected.")

    # Stage D — per-corner analysis
    all_events = []
    for corner in corners:
        all_events.extend(_analyze_corner(corner, track_min_x, track_max_x, fps))

    # Sort and deduplicate nearby events of same type (within 30 frames)
    all_events.sort(key=lambda e: e["frame"])
    deduped = []
    last_by_type: dict[str, int] = {}
    count_by_type: dict[str, int] = {}
    for ev in all_events:
        last = last_by_type.get(ev["type"], -100)
        count = count_by_type.get(ev["type"], 0)
        if ev["frame"] - last > 30 and count < MAX_EVENTS_PER_TYPE:
            deduped.append(ev)
            last_by_type[ev["type"]] = ev["frame"]
            count_by_type[ev["type"]] = count + 1

    # Stage E — consistency
    consistency = _consistency_score(traj, track_min_x, track_max_x)

    # Aggregate metrics
    err_types    = [e["type"] for e in deduped if e["is_error"]]
    good_types   = [e["type"] for e in deduped if not e["is_error"]]
    early_apexes = err_types.count("early_apex")
    missed_apexes = err_types.count("missed_apex")
    wide_exits   = err_types.count("wide_exit")
    good_apexes  = good_types.count("good_apex")
    good_entries = good_types.count("good_entry")

    tight_dists = []
    for corner in corners:
        frames = corner["frames"]
        if not frames:
            continue
        track_w = max(track_max_x - track_min_x, 1)
        direction = corner["direction"]
        if direction == "left":
            dists = [(f["cx"] - track_min_x) / track_w for f in frames]
        else:
            dists = [(track_max_x - f["cx"]) / track_w for f in frames]
        if dists:
            tight_dists.append(min(dists))

    avg_apex_tightness = round(sum(tight_dists) / len(tight_dists), 3) if tight_dists else 0.5

    # Ratio-based scoring: penalty scales with error rate per corner, not raw count.
    # This prevents score collapse when many corners are detected.
    n_corners = max(1, len(corners))
    error_rate = (early_apexes * 1.0 + missed_apexes * 1.5 + wide_exits * 0.5) / n_corners
    # error_rate of 0 = 100, of 1.0 = ~65 (amateur), of 2.0 = ~30 (very rough)
    line_component = max(35, int(100 - error_rate * 35))
    line_score = max(0, min(100, int(
        consistency["score"] * 0.35 + line_component * 0.65
    )))

    metrics = {
        "corners_detected":     len(corners),
        "line_score":           line_score,
        "consistency_score":    consistency["score"],
        "lateral_variance":     consistency["variance"],
        "avg_apex_tightness_pct": int(avg_apex_tightness * 100),
        "early_apex_count":     early_apexes,
        "missed_apex_count":    missed_apexes,
        "wide_exit_count":      wide_exits,
        "good_apex_count":      good_apexes,
        "good_entry_count":     good_entries,
        "track_width_px":       int(track_max_x - track_min_x),
    }

    output = {
        "sport":   sport,
        "events":  deduped,
        "metrics": metrics,
    }

    with open(RACING_LINE_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    errors = [e for e in deduped if e["is_error"]]
    goods  = [e for e in deduped if not e["is_error"]]
    print(f"[racing_line] Done — {len(errors)} errors, {len(goods)} good moments. "
          f"Line score: {line_score}/100  Consistency: {consistency['score']}/100")
    return output


if __name__ == "__main__":
    import sys
    analyze(sys.argv[1] if len(sys.argv) > 1 else None)
