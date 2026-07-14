"""
Local coaching pipeline — multi-stage, parallel, model-routed.

Stage 1 — Event Loader      : read CV events.json, classify severity
Stage 2 — Frame Triage      : moondream binary Q&A per bad-corner frame
Stage 3 — Conditional VLM   : llava only when triage is ambiguous
Stage 4 — Per-event Writer  : llama3.2, parallel via asyncio + ThreadPool
Stage 5 — Session Aggregator: llama3.2, tiny prompt over all notes
"""

import os
import json
import base64
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests

# ── paths ──────────────────────────────────────────────────────────────────────
CV_DIR          = os.path.join(os.path.dirname(__file__), "..", "cv")
FRAMES_DIR      = os.path.join(CV_DIR, "frames")
EVENTS_PATH     = os.path.join(CV_DIR, "events.json")
RACING_LINE_PATH = os.path.join(CV_DIR, "racing_line.json")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "output")

# ── model routing ──────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
TRIAGE_MODEL  = "moondream:latest"   # Stage 2 fallback if Florence-2 unavailable
REASON_MODEL  = "llava:latest"       # Stage 3 — conditional deep reasoning
WRITER_MODEL  = "llama3.2:latest"    # Stage 4 — per-event coaching note
AGGREG_MODEL  = "llama3.2:latest"    # Stage 5 — session aggregation

MAX_PARALLEL  = 4                    # concurrent Ollama calls
_executor     = ThreadPoolExecutor(max_workers=MAX_PARALLEL)

SPORT_SCORE_KEYS = {
    "karting": ["racing_line", "braking", "throttle", "consistency"],
    "biking":  ["lean_commitment", "braking", "body_position", "consistency"],
    "cycling": ["pacing", "positioning", "cornering", "consistency"],
}

SPORT_CONTEXT = {
    "karting": "kart racing",
    "biking":  "motorcycle racing",
    "cycling": "road cycling",
}

CV_LABELS = {
    "late_braking":  "braked too late — overshot braking zone",
    "early_braking": "braked too early — lost corner entry speed",
    "wide_exit":     "ran wide on exit — missed or overshot apex",
    "good_apex":     "tight apex hit — good commitment",
}


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _ollama(model: str, prompt: str, images: list[str] | None = None,
            timeout: int = 90, think: bool = False,
            num_predict: int = 1024) -> str:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "think":   think,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    if images:
        payload["images"] = images
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json()["response"].strip()
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    return text


def _ollama_available() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


async def _async_ollama(model: str, prompt: str,
                        images: list[str] | None = None,
                        timeout: int = 90) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: _ollama(model, prompt, images, timeout),
    )


# ── Stage 1 — Event loader & severity classifier ───────────────────────────────

def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _severity(event: dict) -> str:
    """Route event to: 'good' | 'minor' | 'significant'."""
    etype = event.get("type", "")
    if etype == "good_apex":
        return "good"
    delta = abs(event.get("speed_delta", 0))
    return "significant" if delta >= 5 else "minor"


def _frame_path(frame_idx: int) -> str | None:
    p = os.path.join(FRAMES_DIR, f"frame_{frame_idx + 1:04d}.jpg")
    return p if os.path.exists(p) else None


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_racing_line() -> dict:
    """Load racing_line.json if available, else run analysis first."""
    if not os.path.exists(RACING_LINE_PATH):
        try:
            from cv.racing_line import analyze
            analyze()
        except Exception as e:
            print(f"[S1] Racing line analysis failed: {e}")
            return {"events": [], "metrics": {}}
    try:
        with open(RACING_LINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"events": [], "metrics": {}}


def load_events(sport: str) -> tuple[list[dict], dict]:
    """Returns (events_for_pipeline, racing_line_metrics)."""
    with open(EVENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    fps = data.get("fps", 10)

    # CV braking/line events from flow.py
    enriched = []
    for ev in data.get("events", []):
        fidx = ev.get("frame", 0)
        ts   = ev.get("timestamp", round(fidx / fps, 1))
        enriched.append({
            **ev,
            "timestamp":  ts,
            "fmt_ts":     _fmt_ts(ts),
            "frame_path": _frame_path(fidx),
            "severity":   _severity(ev),
            "label":      CV_LABELS.get(ev.get("type", ""), ev.get("type", "")),
        })

    # Racing line events from racing_line.py (apex, entry, exit analysis)
    rl_data = load_racing_line()
    for rl_ev in rl_data.get("events", []):
        fidx = rl_ev.get("frame", 0)
        enriched.append({
            **rl_ev,
            "fmt_ts":     rl_ev.get("timestamp", _fmt_ts(rl_ev.get("seconds", 0))),
            "frame_path": _frame_path(fidx),
            "severity":   "good" if not rl_ev.get("is_error", True) else
                          ("significant" if rl_ev.get("type") in
                           ("missed_apex", "early_apex", "wide_exit") else "minor"),
            "label":      rl_ev.get("description", rl_ev.get("type", "")),
            "speed_delta": 0,
        })

    # Sort by timestamp, keep max 12 most significant for pipeline speed
    enriched.sort(key=lambda e: e.get("seconds", e.get("timestamp", 0)))
    significant = [e for e in enriched if e["severity"] == "significant"][:6]
    good        = [e for e in enriched if e["severity"] == "good"][:3]
    minor       = [e for e in enriched if e["severity"] == "minor"][:3]
    enriched    = sorted(significant + good + minor,
                         key=lambda e: e.get("seconds", e.get("timestamp", 0)))

    # Fallback: context frames if nothing was found at all
    if not enriched:
        all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
        step = max(1, len(all_frames) // 4)
        for i in range(0, len(all_frames), step):
            if len(enriched) >= 4:
                break
            enriched.append({
                "frame":      i,
                "type":       "context",
                "timestamp":  round(i / fps, 1),
                "fmt_ts":     _fmt_ts(round(i / fps, 1)),
                "frame_path": all_frames[i],
                "severity":   "minor",
                "label":      "context frame",
                "speed_delta": 0,
                "centroid":   [0, 0],
                "seconds":    int(round(i / fps, 0)),
            })

    print(f"[S1] {len(enriched)} events loaded "
          f"({sum(1 for e in enriched if e['severity']=='significant')} significant, "
          f"{sum(1 for e in enriched if e['severity']=='minor')} minor, "
          f"{sum(1 for e in enriched if e['severity']=='good')} good)")
    return enriched, rl_data.get("metrics", {})


# ── Stage 2 — Frame triage (Florence-2 primary, moondream fallback) ───────────

def _try_florence(fpath: str, sport: str) -> dict | None:
    """Run Florence-2 triage. Returns None if unavailable."""
    try:
        from coaching.florence_tagger import tag_frame
        return tag_frame(fpath, sport)
    except Exception as e:
        print(f"[S2] Florence-2 unavailable: {e}")
        return None


async def triage_frame(event: dict, sport: str) -> dict:
    """
    Stage 2 triage: Florence-2 → structured tags (fast, no LLM needed).
    Falls back to moondream binary Q&A if Florence-2 fails.
    """
    fpath = event.get("frame_path")
    if not fpath:
        return {"vehicle_visible": False, "in_corner": False, "obstacle_ahead": False}

    # Try Florence-2 first (CPU, ~1-3s per frame, no Ollama needed)
    loop = asyncio.get_event_loop()
    tags = await loop.run_in_executor(_executor,
                                      lambda: _try_florence(fpath, sport))
    if tags is not None:
        return tags

    # Fallback: moondream binary Q&A via Ollama
    sport_ctx = SPORT_CONTEXT.get(sport, "motorsport")
    prompt = (
        f"You are reviewing a {sport_ctx} onboard camera frame. "
        f"Answer ONLY with 'yes' or 'no' for each:\n"
        f"1. Is a vehicle (kart/bike/cyclist) clearly visible?\n"
        f"2. Is the vehicle actively cornering or turning?\n"
        f"3. Is there any obstacle, traffic, or hazard visible ahead?"
    )
    try:
        img = _encode(fpath)
        raw = await _async_ollama(TRIAGE_MODEL, prompt, images=[img], timeout=45)
        lines = raw.lower().split("\n")
        def yn(line: str) -> bool:
            return "yes" in line
        return {
            "vehicle_visible":  yn(lines[0]) if len(lines) > 0 else False,
            "in_corner":        yn(lines[1]) if len(lines) > 1 else False,
            "obstacle_ahead":   yn(lines[2]) if len(lines) > 2 else False,
            "detected_objects": [],
            "caption":          "",
        }
    except Exception as e:
        print(f"[S2] moondream fallback also failed at {event['fmt_ts']}: {e}")
        return {"vehicle_visible": False, "in_corner": False, "obstacle_ahead": False}


# ── Stage 3 — Conditional VLM reasoning (llava, only when needed) ──────────────

def _needs_vlm(event: dict, tags: dict) -> bool:
    """Fire llava only for significant events OR when obstacle detected."""
    return event["severity"] == "significant" or tags.get("obstacle_ahead", False)


async def reason_frame(event: dict, sport: str, tags: dict) -> str:
    """Narrow targeted question — much faster than open-ended description."""
    fpath = event.get("frame_path")
    if not fpath:
        return ""

    sport_ctx = SPORT_CONTEXT.get(sport, "motorsport")
    label     = event["label"]
    obstacle  = tags.get("obstacle_ahead", False)

    if obstacle:
        q = (f"CV telemetry flagged: {label}. There appears to be an obstacle ahead. "
             f"In 1 sentence: was this error caused by traffic/obstacle, or pure driver mistake?")
    else:
        q = (f"CV telemetry flagged: {label}. "
             f"In 1 sentence: what specific {sport_ctx} technique error is visible in this frame?")

    try:
        img = _encode(fpath)
        return await _async_ollama(REASON_MODEL, q, images=[img], timeout=60)
    except Exception as e:
        print(f"[S3] llava reasoning failed at {event['fmt_ts']}: {e}")
        return ""


# ── Stage 4 — Per-event writer (llama3.2, parallel) ──────────────────────────

async def write_event_note(event: dict, sport: str,
                           tags: dict, reasoning: str) -> dict:
    """~200-token prompt per event — runs in parallel across all events."""
    sport_ctx = SPORT_CONTEXT.get(sport, "motorsport")
    etype     = event.get("type", "context")
    is_good   = (etype == "good_apex")

    # Model routing: good events use a shorter prompt, no deep analysis
    if is_good:
        prompt = (
            f"You are a {sport_ctx} coach. At {event['fmt_ts']}, CV detected a tight apex — "
            f"good commitment. Write one sentence of positive reinforcement."
        )
    else:
        reasoning_line = f"\nVisual analysis: {reasoning}" if reasoning else ""
        prompt = (
            f"You are a {sport_ctx} coach. At {event['fmt_ts']}, "
            f"CV detected: {event['label']} (speed delta: {event.get('speed_delta', 0):.1f})."
            f"{reasoning_line}\n"
            f"Write 2 sentences of specific coaching feedback. Be technical and direct."
        )

    try:
        note = await _async_ollama(WRITER_MODEL, prompt, timeout=60)
    except Exception as e:
        print(f"[S4] writer failed at {event['fmt_ts']}: {e}")
        note = f"CV flagged {event['label']} at {event['fmt_ts']}."

    return {
        "timestamp":   event["fmt_ts"],
        "seconds":     event.get("seconds", 0),
        "type":        etype,
        "severity":    event["severity"],
        "description": note,
        "is_error":    etype not in ("good_apex", "context"),
    }


# ── Stage 5 — Session aggregator ──────────────────────────────────────────────

def aggregate_session(sport: str, notes: list[dict],
                      rl_metrics: dict | None = None) -> dict:
    score_keys = SPORT_SCORE_KEYS.get(sport, ["technique", "consistency", "line", "pace"])
    sport_ctx  = SPORT_CONTEXT.get(sport, "motorsport")

    errors  = [n for n in notes if n["is_error"]]
    moments = [n for n in notes if not n["is_error"] and n["type"] != "context"]

    notes_text = "\n".join(
        "  {} [{}]: {}".format(
            n["timestamp"], n["severity"],
            n["description"].replace("\n", " ").replace("\r", "").replace('"', "'")
        )
        for n in notes
    )
    score_schema = ", ".join(f'"{k}": 78' for k in score_keys)

    rl_text = ""
    if rl_metrics:
        rl_text = (
            f"\nRacing line CV metrics: line_score={rl_metrics.get('line_score', '?')}/100, "
            f"consistency={rl_metrics.get('consistency_score', '?')}/100, "
            f"missed_apexes={rl_metrics.get('missed_apex_count', 0)}, "
            f"early_apexes={rl_metrics.get('early_apex_count', 0)}, "
            f"good_apexes={rl_metrics.get('good_apex_count', 0)}, "
            f"corners={rl_metrics.get('corners_detected', 0)}."
        )
        # Override racing_line score from CV data if available
        rl_score = rl_metrics.get("line_score")
        if rl_score is not None and "racing_line" in score_keys:
            score_schema = score_schema.replace(
                '"racing_line": 75', f'"racing_line": {rl_score}'
            )

    n_errors  = len(errors)
    n_moments = len(moments)
    prompt = (
        f"You are a {sport_ctx} performance analyst scoring a session objectively.\n"
        f"Per-event notes ({n_errors} errors, {n_moments} strengths):\n{notes_text}{rl_text}\n\n"
        f"SCORING CALIBRATION (real-world amateur benchmark):\n"
        f"  - 85-95: elite/near-perfect execution\n"
        f"  - 70-84: solid amateur with minor recurring issues\n"
        f"  - 55-69: clear technical weaknesses appearing multiple times\n"
        f"  - <55: fundamental technique issues every lap\n"
        f"Balance errors AND strengths equally. A driver with good throttle and 2-3 early braking "
        f"events should score 70-80 for braking, not 50. If strengths outnumber errors, score >=75.\n\n"
        f"RULES: coaching_analysis is ONE short string (not a list or array). "
        f"Scores are integers 0-100. Use racing line CV metrics for racing_line score.\n"
        f"Return ONLY valid JSON:\n"
        f'{{"session_summary": "One sentence summary.", '
        f'"coaching_analysis": "One sentence insight referencing a timestamp.", '
        f'"scores": {{{score_schema}}}, '
        f'"driver_archetype": "Label"}}'
    )

    try:
        raw = _ollama(AGGREG_MODEL, prompt, timeout=180, num_predict=1500)
        import re
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        raw = re.sub(r"```\s*$", "", raw).strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        agg = json.loads(raw[start:end]) if start >= 0 and end > start else {}
    except Exception as e:
        print(f"[S5] aggregator failed: {e} — using note-based fallback")
        agg = {}

    return {
        "sport":            sport,
        "session_summary":  agg.get("session_summary", f"{sport_ctx} session analyzed locally."),
        "errors":           [{"timestamp": e["timestamp"], "seconds": e["seconds"],
                              "description": e["description"]} for e in errors],
        "best_moments":     [{"timestamp": m["timestamp"], "seconds": m["seconds"],
                              "description": m["description"]} for m in moments],
        "coaching_analysis": agg.get("coaching_analysis", "See per-event notes above."),
        "scores":           agg.get("scores", {k: 65 for k in score_keys}),
        "driver_archetype": agg.get("driver_archetype", "Data-analyzed driver"),
    }


# ── Async pipeline runner ──────────────────────────────────────────────────────

async def _run_pipeline(events: list[dict], sport: str) -> list[dict]:
    """
    Florence-2 triage runs SERIALLY (CPU, GIL-bound — parallelism hurts).
    Ollama writers run in PARALLEL (I/O-bound — parallelism helps).
    """
    # Stage 2: Florence-2 triage — serial
    all_tags: list[dict] = []
    for ev in events:
        if ev["severity"] == "good":
            all_tags.append({})
        else:
            all_tags.append(await triage_frame(ev, sport))

    # Stage 3: Conditional VLM — serial (only fires for significant events)
    all_reasoning: list[str] = []
    for ev, tags in zip(events, all_tags):
        if ev["severity"] != "good" and _needs_vlm(ev, tags):
            all_reasoning.append(await reason_frame(ev, sport, tags))
        else:
            all_reasoning.append("")

    # Stage 4: Per-event writers — PARALLEL (Ollama is I/O-bound)
    write_tasks = [
        write_event_note(ev, sport, tags, reasoning)
        for ev, tags, reasoning in zip(events, all_tags, all_reasoning)
    ]
    return await asyncio.gather(*write_tasks)


# ── Main entry point ───────────────────────────────────────────────────────────

def caption_local(sport: str | None = None) -> dict:
    """Local replacement for caption() — zero paid APIs."""
    with open(EVENTS_PATH, encoding="utf-8") as f:
        events_data = json.load(f)
    if sport is None:
        sport = events_data.get("sport", "karting")

    print(f"[local_caption] Pipeline start — sport={sport}")

    if not _ollama_available():
        print("[local_caption] Ollama unreachable — CV-only fallback")
        score_keys = SPORT_SCORE_KEYS.get(sport, ["technique", "consistency", "line", "pace"])
        result = _cv_fallback(sport, events_data, score_keys)
    else:
        events, rl_metrics = load_events(sport)

        print(f"[S2-S4] Running parallel triage + write for {len(events)} events...")
        notes = asyncio.run(_run_pipeline(events, sport))

        print(f"[S5] Aggregating session report...")
        result = aggregate_session(sport, notes, rl_metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"{sport}_structured.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[local_caption] Done: {out}")
    return result


def _cv_fallback(sport: str, events_data: dict, score_keys: list) -> dict:
    events = events_data.get("events", [])
    errors = [{"timestamp": _fmt_ts(e.get("timestamp", 0)),
               "seconds":   int(e.get("timestamp", 0)),
               "description": CV_LABELS.get(e["type"], e["type"])}
              for e in events if e.get("type") not in ("good_apex",)]
    moments = [{"timestamp": _fmt_ts(e.get("timestamp", 0)),
                "seconds":   int(e.get("timestamp", 0)),
                "description": "Good apex commitment"}
               for e in events if e.get("type") == "good_apex"]
    return {
        "sport": sport,
        "session_summary": f"{sport.title()} session — CV pipeline only (Ollama unavailable).",
        "errors": errors or [{"timestamp": "0:00", "seconds": 0,
                               "description": "No major events detected"}],
        "best_moments": moments or [{"timestamp": "0:00", "seconds": 0,
                                     "description": "Session completed cleanly"}],
        "coaching_analysis": f"{len(errors)} CV events flagged. Start Ollama for full analysis.",
        "scores": {k: 65 for k in score_keys},
        "driver_archetype": "Data-analyzed driver",
    }


if __name__ == "__main__":
    import sys
    caption_local(sys.argv[1] if len(sys.argv) > 1 else None)
