import os
import json
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

CV_DIR = os.path.join(os.path.dirname(__file__), "..", "cv")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GEMINI_OUT = os.path.join(OUTPUT_DIR, "gemini_analysis.txt")

KARTING_PROMPT = """You are an expert kart racing coach and race engineer. Watch this race footage carefully.

Analyze the following and be specific:
1. Braking points: are they early, correct, or late into each corner?
2. Racing line: is the driver hitting apexes or running wide?
3. Throttle application: is it smooth and progressive or abrupt?
4. Corner exit: is the driver maximizing exit speed?
5. Any specific corners where significant time is being lost?

Be direct, technical, and specific. Coaching tone. Reference timestamps where possible."""

BIKING_PROMPT = """You are an expert motorcycle racing coach. Watch this race footage carefully.

Analyze the following and be specific:
1. Lean angle: is the rider committing fully or being conservative?
2. Body position: tucked, upright, or hanging off correctly?
3. Braking points: late, early, or correct?
4. Corner entry vs exit posture: any inconsistencies?
5. Safety flags: any moments where knee clearance looks dangerously low?

Be direct, technical, and specific. Coaching tone. Reference timestamps where possible."""


def caption() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(EVENTS_PATH) as f:
        events_data = json.load(f)

    sport = events_data.get("sport", "karting")
    video_path = events_data.get("video_path", os.path.join(CV_DIR, "video.mp4"))

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)

    print(f"[caption] Uploading {video_path} to Gemini...")
    video_file = client.files.upload(
        path=video_path,
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )

    print("[caption] Waiting for Gemini to process video...")
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file processing failed: {video_file.state.name}")

    prompt = BIKING_PROMPT if sport == "biking" else KARTING_PROMPT

    print("[caption] Sending prompt to Gemini 2.5 Flash...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[video_file, prompt],
    )
    analysis = response.text

    with open(GEMINI_OUT, "w", encoding="utf-8") as f:
        f.write(analysis)

    print(f"[caption] Done. Analysis saved to {GEMINI_OUT}")
    return analysis


if __name__ == "__main__":
    result = caption()
    print("\n--- Gemini Analysis ---")
    print(result)
