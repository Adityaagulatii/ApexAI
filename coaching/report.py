import os
import json

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GEMINI_STRUCTURED_PATH = os.path.join(OUTPUT_DIR, "gemini_structured.json")
REPORT_OUT = os.path.join(OUTPUT_DIR, "report.md")

DRIVER_STYLE_PROMPT = """You are a motorsport performance analyst. Based on these performance scores and observations, generate a driver style profile.

Scores (0-100):
{scores}

Key observations:
{observations}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "archetype": "One punchy driver archetype label (e.g. Momentum Driver, Aggressive Braker, Smooth Operator)",
  "tags": [
    {{"label": "Short label 2-3 words", "emoji": "single emoji", "sentiment": "positive"}},
    {{"label": "Short label 2-3 words", "emoji": "single emoji", "sentiment": "negative"}},
    {{"label": "Short label 2-3 words", "emoji": "single emoji", "sentiment": "positive"}},
    {{"label": "Short label 2-3 words", "emoji": "single emoji", "sentiment": "neutral"}}
  ]
}}

Rules:
- Exactly 4 tags
- sentiment must be "positive", "negative", or "neutral"
- Tags should reflect the actual driving data
- Be direct and specific"""


def _groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment.")
    return Groq(api_key=api_key)


def generate_driver_style(data: dict) -> dict:
    scores = data.get("scores", {})
    errors = data.get("errors", [])
    moments = data.get("best_moments", [])
    analysis = data.get("coaching_analysis", "")

    observations = []
    for e in errors[:4]:
        observations.append(f"Error: {e['description']}")
    for m in moments[:4]:
        observations.append(f"Strength: {m['description']}")
    if analysis:
        observations.append(analysis[:400])

    scores_str = "\n".join(f"  {k.replace('_', ' ').title()}: {v}/100" for k, v in scores.items())
    obs_str = "\n".join(f"- {o}" for o in observations)

    prompt = DRIVER_STYLE_PROMPT.format(scores=scores_str, observations=obs_str)

    client = _groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )

    raw = completion.choices[0].message.content.strip()
    # strip markdown fences if present
    import re
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    return json.loads(raw.strip())


def ask_engineer(question: str, structured: dict) -> str:
    errors = structured.get("errors", [])
    moments = structured.get("best_moments", [])
    analysis = structured.get("coaching_analysis", "")
    scores = structured.get("scores", {})
    sport = structured.get("sport", "karting")

    context_lines = [
        f"Sport: {sport}",
        f"Performance scores: {json.dumps(scores)}",
        "",
        "Errors detected:",
    ]
    for e in errors:
        context_lines.append(f"  {e['timestamp']} — {e['description']}")
    context_lines.append("")
    context_lines.append("Best moments:")
    for m in moments:
        context_lines.append(f"  {m['timestamp']} — {m['description']}")
    context_lines += ["", "Full coaching analysis:", analysis]

    prompt = f"""You are an AI race engineer. Answer the driver's question based on the video analysis below.

Video Analysis:
{chr(10).join(context_lines)}

Driver's question: {question}

Answer in 3-5 sentences. Be direct, specific, and technical. Reference timestamps from the analysis where relevant."""

    client = _groq_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return completion.choices[0].message.content.strip()


def generate_report() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(GEMINI_STRUCTURED_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Generate driver style if not already present
    if not data.get("driver_style"):
        print("[report] Generating driver style profile via Groq...")
        try:
            driver_style = generate_driver_style(data)
            data["driver_style"] = driver_style
            with open(GEMINI_STRUCTURED_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[report] Driver style generation failed: {e}")

    lines = []

    summary = data.get("session_summary", "")
    if summary:
        lines += ["## Session Summary", summary, ""]

    errors = data.get("errors", [])
    if errors:
        lines += ["## Errors"]
        for e in errors:
            lines.append(f"**{e['timestamp']}** — {e['description']}")
            lines.append("")

    moments = data.get("best_moments", [])
    if moments:
        lines += ["## Best Moments"]
        for m in moments:
            lines.append(f"**{m['timestamp']}** — {m['description']}")
            lines.append("")

    analysis = data.get("coaching_analysis", "")
    if analysis:
        lines += ["## Coaching Analysis", analysis, ""]

    ds = data.get("driver_style", {})
    if ds:
        lines += ["## Driver Style", f"_{ds.get('archetype', '')}_", ""]

    scores = data.get("scores", {})
    if scores:
        lines += ["## Performance Scores"]
        for metric, val in scores.items():
            label = metric.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}/100")
        lines.append("")

    report = "\n".join(lines)

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[report] Done. Report saved to {REPORT_OUT}")
    return report


if __name__ == "__main__":
    print(generate_report())
