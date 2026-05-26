import os
import json
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CV_DIR = os.path.join(os.path.dirname(__file__), "cv")
VIDEO_PATH = os.path.join(CV_DIR, "video.mp4")
OVERLAY_PATH = os.path.join(CV_DIR, "output_overlay.mp4")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "coaching", "output", "report.md")

st.set_page_config(page_title="PitLane AI", layout="wide")


def run_cv_pipeline(sport: str) -> dict:
    from cv.ingest import ingest
    from cv.detect import detect
    from cv.flow import analyze
    from cv.overlay import render

    ingest(VIDEO_PATH)
    detect(sport)
    events = analyze(sport)
    render()
    return events


def run_coaching_pipeline() -> str:
    from coaching.caption import caption
    from coaching.report import generate_report

    caption()
    report = generate_report()
    return report


def main():
    col_video, col_report = st.columns([2, 1])

    with col_video:
        st.title("PitLane AI 🏎")
        st.caption("Vision-only AI race engineer. No sensors. No hardware. Just video.")

        uploaded = st.file_uploader("Upload race video", type=["mp4", "mov", "avi"])
        sport_choice = st.radio("Select sport", ["Karting", "Biking"], horizontal=True)
        sport = sport_choice.lower()

        analyze_btn = st.button("Analyze", disabled=(uploaded is None), type="primary")

        if analyze_btn and uploaded is not None:
            os.makedirs(CV_DIR, exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(__file__), "coaching", "output"), exist_ok=True)

            with open(VIDEO_PATH, "wb") as f:
                f.write(uploaded.read())

            try:
                with st.spinner("Running CV pipeline (detection + tracking + events)..."):
                    events_data = run_cv_pipeline(sport)

                with st.spinner("Analyzing footage with Gemini + generating coaching report..."):
                    report_text = run_coaching_pipeline()

                st.session_state["events_data"] = events_data
                st.session_state["report_text"] = report_text
                st.success("Analysis complete!")

            except FileNotFoundError as e:
                st.error(f"File error: {e}")
            except RuntimeError as e:
                if "No vehicles detected" in str(e) or "detections.json is empty" in str(e).lower():
                    st.error("No vehicles detected. Try a clearer video.")
                else:
                    st.error(f"Pipeline error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

        # Show overlay video if available
        if os.path.exists(OVERLAY_PATH):
            st.subheader("Analyzed Footage")
            with open(OVERLAY_PATH, "rb") as vf:
                st.video(vf.read())

            # Event summary
            if os.path.exists(EVENTS_PATH):
                with open(EVENTS_PATH) as f:
                    ev = json.load(f)
                events = ev.get("events", [])
                braking = sum(1 for e in events if "braking" in e["type"])
                wide = sum(1 for e in events if e["type"] == "wide_exit")
                apex = sum(1 for e in events if e["type"] == "good_apex")
                parts = []
                if braking:
                    parts.append(f"{braking} braking event{'s' if braking != 1 else ''}")
                if wide:
                    parts.append(f"{wide} wide exit{'s' if wide != 1 else ''}")
                if apex:
                    parts.append(f"{apex} good apex{'es' if apex != 1 else ''}")
                summary = " · ".join(parts) if parts else "No significant events detected"
                st.caption(summary)

    with col_report:
        st.subheader("Coaching Report")

        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, encoding="utf-8") as f:
                report_md = f.read()
            st.markdown(report_md)
            st.download_button(
                label="Download report.md",
                data=report_md.encode("utf-8"),
                file_name="pitlane_coaching_report.md",
                mime="text/markdown",
            )
        elif "report_text" in st.session_state:
            st.markdown(st.session_state["report_text"])
        else:
            st.info("Upload a video and click Analyze to generate your coaching report.")


if __name__ == "__main__":
    main()
