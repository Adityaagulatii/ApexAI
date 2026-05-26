import os
import json
import time
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

CV_DIR = os.path.join(os.path.dirname(__file__), "..", "cv")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

KARTING_PROMPT = """You are an expert kart racing coach watching onboard race footage.

Return ONLY valid JSON — no markdown fences, no explanations — in this exact schema:

{
  "session_summary": "2-3 sentence overall performance summary",
  "errors": [
    {"timestamp": "M:SS", "seconds": 14, "description": "specific technical error description"}
  ],
  "best_moments": [
    {"timestamp": "M:SS", "seconds": 45, "description": "specific description of what was done well"}
  ],
  "coaching_analysis": "Full multi-paragraph technical analysis covering braking points, racing line, throttle application, and corner exits. Be specific and reference timestamps.",
  "scores": {
    "racing_line": 72,
    "braking": 65,
    "throttle": 78,
    "consistency": 70
  },
  "driver_archetype": "One-line driver style description (e.g. Aggressive entry, passive exit)"
}

Rules:
- Find at minimum 4 errors and 4 best moments with exact timestamps from the footage
- scores are integers 0-100
- seconds must be the numeric value matching the timestamp
- Be direct, technical, and specific — coaching tone throughout"""

BIKING_PROMPT = """You are an expert motorcycle racing coach watching onboard race footage.

Return ONLY valid JSON — no markdown fences, no explanations — in this exact schema:

{
  "session_summary": "2-3 sentence overall performance summary",
  "errors": [
    {"timestamp": "M:SS", "seconds": 14, "description": "specific technical error description"}
  ],
  "best_moments": [
    {"timestamp": "M:SS", "seconds": 45, "description": "specific description of what was done well"}
  ],
  "coaching_analysis": "Full multi-paragraph technical analysis covering lean angle, body position, braking points, corner entry and exit. Be specific and reference timestamps.",
  "scores": {
    "lean_commitment": 72,
    "braking": 65,
    "body_position": 78,
    "consistency": 70
  },
  "driver_archetype": "One-line rider style description"
}

Rules:
- Find at minimum 4 errors and 4 best moments with exact timestamps from the footage
- scores are integers 0-100
- seconds must be the numeric value matching the timestamp
- Be direct, technical, and specific — coaching tone throughout"""

CYCLING_PROMPT = """You are an expert road cycling coach watching onboard race footage.

Return ONLY valid JSON — no markdown fences, no explanations — in this exact schema:

{
  "session_summary": "2-3 sentence overall performance summary",
  "errors": [
    {"timestamp": "M:SS", "seconds": 14, "description": "specific technical error description"}
  ],
  "best_moments": [
    {"timestamp": "M:SS", "seconds": 45, "description": "specific description of what was done well"}
  ],
  "coaching_analysis": "Full multi-paragraph technical analysis covering pacing strategy, drafting and positioning, cornering technique, and climbing or sprinting efficiency. Be specific and reference timestamps.",
  "scores": {
    "pacing": 72,
    "positioning": 65,
    "cornering": 78,
    "consistency": 70
  },
  "driver_archetype": "One-line rider style description (e.g. Steady Climber, Aggressive Sprinter)"
}

Rules:
- Find at minimum 4 errors and 4 best moments with exact timestamps from the footage
- scores are integers 0-100
- seconds must be the numeric value matching the timestamp
- Be direct, technical, and specific — coaching tone throughout"""


def _fix_multiline_strings(text: str) -> str:
    """Escape literal newlines inside JSON string values."""
    result = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif ch == "\n" and in_string:
            result.append("\\n")
        elif ch == "\r" and in_string:
            pass
        else:
            result.append(ch)
    return "".join(result)


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = _fix_multiline_strings(text.strip())
    return json.loads(text)


def caption() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(EVENTS_PATH) as f:
        events_meta = json.load(f)

    sport = events_meta.get("sport", "karting")
    video_path = events_meta.get("video_path", os.path.join(CV_DIR, "video.mp4"))

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)

    print(f"[caption] Uploading {video_path} to Gemini...")
    video_file = client.files.upload(
        file=video_path,
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )

    print("[caption] Waiting for Gemini to process video...")
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file processing failed: {video_file.state.name}")

    if sport == "biking":
        prompt = BIKING_PROMPT
    elif sport == "cycling":
        prompt = CYCLING_PROMPT
    else:
        prompt = KARTING_PROMPT

    raw_text = None
    for model_id in ("gemini-2.5-flash", "gemini-2.0-flash"):
        try:
            print(f"[caption] Sending prompt to {model_id}...")
            response = client.models.generate_content(
                model=model_id,
                contents=[video_file, prompt],
            )
            raw_text = response.text
            break
        except Exception as e:
            print(f"[caption] {model_id} failed ({e}), trying next model...")

    if raw_text is None:
        raise RuntimeError("All Gemini models failed.")

    raw_out = os.path.join(OUTPUT_DIR, f"{sport}_gemini_analysis.txt")
    structured_out = os.path.join(OUTPUT_DIR, f"{sport}_structured.json")

    with open(raw_out, "w", encoding="utf-8") as f:
        f.write(raw_text)

    try:
        structured = _parse_json(raw_text)
    except json.JSONDecodeError:
        structured = {
            "session_summary": "",
            "errors": [],
            "best_moments": [],
            "coaching_analysis": raw_text,
            "scores": {},
            "driver_archetype": "",
        }

    structured["sport"] = sport

    with open(structured_out, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    print(f"[caption] Done. Structured data saved to {structured_out}")
    return structured


if __name__ == "__main__":
    result = caption()
    print(json.dumps(result, indent=2))
