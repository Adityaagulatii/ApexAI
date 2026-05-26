import os
import json

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CV_DIR = os.path.join(os.path.dirname(__file__), "..", "cv")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GEMINI_ANALYSIS_PATH = os.path.join(OUTPUT_DIR, "gemini_analysis.txt")
REPORT_OUT = os.path.join(OUTPUT_DIR, "report.md")

REPORT_PROMPT = """You are a professional race engineer writing a post-session coaching report.

Here is the computer vision analysis of key events detected in the footage:
{cv_events_summary}

Here is the semantic video analysis from an AI coach:
{gemini_analysis}

Write a structured coaching report with these exact sections:
## Session Summary
2-3 sentences on overall performance.

## Top 3 Issues to Fix
Numbered list. Each issue: what it is, why it costs time/safety, exact correction.

## What You Did Well
2-3 specific positives.

## Priority for Next Session
One single focus point.

Be direct, specific, and actionable. No filler."""


def _summarize_events(events: list, sport: str) -> str:
    if not events:
        return "No significant CV events detected."

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev["type"]] = counts.get(ev["type"], 0) + 1

    lines = [f"Sport: {sport}", f"Total flagged events: {len(events)}"]
    for etype, count in counts.items():
        lines.append(f"  - {etype}: {count} occurrence(s)")

    lines.append("")
    lines.append("Event details:")
    for ev in events[:15]:
        lean = f", lean={ev['lean_angle']}°" if ev.get("lean_angle") else ""
        lines.append(
            f"  t={ev['timestamp']}s  [{ev['type']}]  speed_delta={ev['speed_delta']}{lean}"
        )
    if len(events) > 15:
        lines.append(f"  ... and {len(events) - 15} more events")

    return "\n".join(lines)


def generate_report() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(EVENTS_PATH) as f:
        events_data = json.load(f)

    with open(GEMINI_ANALYSIS_PATH, encoding="utf-8") as f:
        gemini_analysis = f.read()

    sport = events_data.get("sport", "karting")
    events = events_data.get("events", [])
    cv_summary = _summarize_events(events, sport)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment.")

    client = Groq(api_key=api_key)

    prompt = REPORT_PROMPT.format(
        cv_events_summary=cv_summary,
        gemini_analysis=gemini_analysis,
    )

    print("[report] Calling Groq LLaMA 70B...")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1500,
    )

    report = completion.choices[0].message.content

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[report] Done. Report saved to {REPORT_OUT}")
    return report


if __name__ == "__main__":
    result = generate_report()
    print("\n--- Coaching Report ---")
    print(result)
