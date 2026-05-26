import os
import json

import cv2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CV_DIR = os.path.join(BASE_DIR, "cv")
VIDEO_PATH = os.path.join(CV_DIR, "video.mp4")
OVERLAY_PATH = os.path.join(CV_DIR, "output_overlay.mp4")
EVENTS_PATH = os.path.join(CV_DIR, "events.json")
STRUCTURED_PATH = os.path.join(BASE_DIR, "coaching", "output", "gemini_structured.json")
REPORT_PATH = os.path.join(BASE_DIR, "coaching", "output", "report.md")

st.set_page_config(page_title="PitLane AI", layout="wide", page_icon="🏎")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;background:#08080C;color:#fff;}
.stApp{background:#08080C;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1rem 1.5rem;max-width:100%;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#111116;border-bottom:1px solid #22222E;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#9090A0;font-size:13px;font-weight:600;padding:12px 16px;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#fff;border-bottom:2px solid #00FF87;background:transparent;}
.stTabs [data-baseweb="tab-panel"]{background:#111116;padding:0;}

/* buttons */
.stButton>button{background:transparent;border:1px solid #22222E;color:#9090A0;border-radius:6px;font-size:12px;font-weight:600;padding:4px 10px;transition:all .15s;}
.stButton>button:hover{border-color:#00FF87;color:#00FF87;background:rgba(0,255,135,0.05);}

/* analyze button */
.analyze-btn>button{background:#00FF87!important;color:#000!important;border:none!important;font-weight:700!important;font-size:14px!important;border-radius:8px!important;padding:10px 24px!important;}
.analyze-btn>button:hover{background:#00CC6A!important;box-shadow:0 0 20px rgba(0,255,135,0.3)!important;}
.analyze-btn>button:disabled{background:#1a3d2b!important;color:#3a7a55!important;}

/* send button */
.send-btn>button{background:#FF3B3B!important;color:#fff!important;border:none!important;font-weight:600!important;border-radius:6px!important;padding:6px 16px!important;font-size:13px!important;}
.send-btn>button:hover{background:#cc2f2f!important;}

/* file uploader */
[data-testid="stFileUploader"]{background:#16161D;border:1px dashed #2E2E3E;border-radius:10px;}

/* text input */
.stTextInput>div>div>input{background:#16161D!important;border:1px solid #22222E!important;color:#fff!important;border-radius:6px!important;font-size:13px!important;}
.stTextInput>div>div>input:focus{border-color:#00FF87!important;box-shadow:none!important;}
.stTextInput>div>div>input::placeholder{color:#55555F!important;}

/* metrics */
[data-testid="stMetric"]{background:#16161D;border:1px solid #22222E;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:26px;font-weight:800;}
[data-testid="stMetricLabel"]{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#9090A0;}

/* video */
[data-testid="stVideo"] video{border-radius:12px;background:#000;width:100%;}

/* divider */
hr{border-color:#22222E;}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_ts(s: float) -> str:
    return f"{int(s)//60}:{int(s)%60:02d}"

def score_color(v: int) -> str:
    return "#00FF87" if v >= 80 else "#FFD426" if v >= 60 else "#FF3B3B"

def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    return d

def load_structured() -> dict | None:
    return json.load(open(STRUCTURED_PATH, encoding="utf-8")) if os.path.exists(STRUCTURED_PATH) else None

def run_cv_pipeline(sport: str) -> None:
    from cv.ingest import ingest
    from cv.detect import detect
    from cv.flow import analyze
    from cv.overlay import render
    ingest(VIDEO_PATH); detect(sport); analyze(sport); render()

def run_coaching_pipeline() -> None:
    from coaching.caption import caption
    from coaching.report import generate_report
    caption(); generate_report()


# ── marker bar ────────────────────────────────────────────────────────────────

def marker_bar(errors: list, moments: list, dur: float) -> str:
    if dur <= 0:
        return ""
    def dot(s, col, glow):
        p = min(max(s / dur * 100, 0), 100)
        return (f'<div style="position:absolute;left:{p:.1f}%;top:50%;transform:translate(-50%,-50%);'
                f'width:8px;height:8px;border-radius:50%;background:{col};'
                f'box-shadow:0 0 6px {glow};cursor:pointer;" title="{fmt_ts(s)}"></div>')
    dots = "".join(dot(e.get("seconds",0),"#FF3B3B","rgba(255,59,59,.6)") for e in errors)
    dots += "".join(dot(m.get("seconds",0),"#00FF87","rgba(0,255,135,.6)") for m in moments)
    return f"""<div style="margin:8px 0 4px;">
  <div style="position:relative;height:24px;background:#16161D;border-radius:6px;">
    <div style="position:absolute;top:50%;left:0;right:0;height:2px;background:#22222E;transform:translateY(-50%);border-radius:2px;"></div>
    {dots}
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:3px;">
    <span style="font-size:10px;color:#55555F;font-family:monospace;">0:00</span>
    <span style="font-size:10px;color:#55555F;font-family:monospace;">{fmt_ts(dur/2)}</span>
    <span style="font-size:10px;color:#55555F;font-family:monospace;">{fmt_ts(dur)}</span>
  </div>
  <div style="display:flex;gap:14px;margin-top:5px;">
    <span style="font-size:11px;color:#9090A0;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#FF3B3B;margin-right:4px;vertical-align:middle;"></span>{len(errors)} error{'s' if len(errors)!=1 else ''}</span>
    <span style="font-size:11px;color:#9090A0;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#00FF87;margin-right:4px;vertical-align:middle;"></span>{len(moments)} best moment{'s' if len(moments)!=1 else ''}</span>
  </div>
</div>"""


# ── event cards ───────────────────────────────────────────────────────────────

def event_row(ts: str, desc: str, key: str, secs: float, color: str) -> None:
    c1, c2 = st.columns([0.17, 0.83])
    with c1:
        if st.button(f"▶ {ts}", key=key):
            st.session_state["seek_to"] = int(secs)
            st.rerun()
    with c2:
        st.markdown(f'<p style="font-size:13px;color:#fff;margin:3px 0 0;">{desc}</p>',
                    unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:#22222E;margin:7px 0;">', unsafe_allow_html=True)


# ── score cards ───────────────────────────────────────────────────────────────

def score_cards(scores: dict) -> None:
    cols = st.columns(len(scores))
    for col, (k, v) in zip(cols, scores.items()):
        col.markdown(
            f'<div style="background:#16161D;border:1px solid #22222E;border-radius:10px;'
            f'padding:12px;text-align:center;">'
            f'<div style="font-size:26px;font-weight:800;color:{score_color(v)};letter-spacing:-.03em;">{v}</div>'
            f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#9090A0;margin-top:3px;">'
            f'{k.replace("_"," ").title()}</div></div>',
            unsafe_allow_html=True)


# ── driver style ──────────────────────────────────────────────────────────────

SENTIMENT_STYLE = {
    "positive": ("rgba(0,255,135,0.08)", "rgba(0,255,135,0.18)", "#00FF87"),
    "negative": ("rgba(255,59,59,0.08)",  "rgba(255,59,59,0.18)",  "#FF3B3B"),
    "neutral":  ("rgba(255,212,38,0.08)", "rgba(255,212,38,0.18)", "#FFD426"),
}

def driver_style_card(ds: dict) -> None:
    archetype = ds.get("archetype", "")
    tags = ds.get("tags", [])
    chips = ""
    for t in tags:
        bg, border, color = SENTIMENT_STYLE.get(t.get("sentiment","neutral"),
                                                 SENTIMENT_STYLE["neutral"])
        chips += (f'<span style="display:inline-flex;align-items:center;gap:4px;'
                  f'background:{bg};border:1px solid {border};color:{color};'
                  f'border-radius:6px;padding:3px 10px;font-size:12px;font-weight:500;'
                  f'margin:3px 4px 3px 0;">{t.get("emoji","")} {t.get("label","")}</span>')
    st.markdown(
        f'<div style="background:rgba(0,255,135,0.03);border:1px solid rgba(0,255,135,0.1);'
        f'border-radius:10px;padding:14px 16px;margin-bottom:14px;">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        f'color:#55555F;margin-bottom:6px;">Driver Style</div>'
        f'<div style="font-size:15px;font-weight:700;color:#fff;margin-bottom:10px;">{archetype}</div>'
        f'<div>{chips}</div></div>',
        unsafe_allow_html=True)


# ── ask the engineer ──────────────────────────────────────────────────────────

QUICK_QUESTIONS = [
    "Where am I losing the most time?",
    "How's my braking technique?",
    "What should I focus on next session?",
    "Where are my strongest moments?",
]

def ask_section(structured: dict) -> None:
    st.markdown('<hr style="border-color:#22222E;margin:12px 0 10px;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        'color:#55555F;margin-bottom:8px;">Ask the Engineer</div>',
        unsafe_allow_html=True)

    # Quick question chips
    q_cols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        if q_cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state["eng_question"] = q
            st.session_state.pop("eng_answer", None)

    # Text input
    question = st.text_input("", placeholder="Ask anything about this session…",
                              value=st.session_state.get("eng_question", ""),
                              key="eng_input", label_visibility="collapsed")

    send_col, _ = st.columns([0.25, 0.75])
    with send_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Ask →", disabled=not question.strip(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if send and question.strip():
        with st.spinner("Analysing your footage…"):
            try:
                from coaching.report import ask_engineer
                answer = ask_engineer(question.strip(), structured)
                st.session_state["eng_answer"] = answer
                st.session_state["eng_question"] = question.strip()
            except Exception as e:
                st.session_state["eng_answer"] = f"Error: {e}"

    answer = st.session_state.get("eng_answer")
    if answer:
        st.markdown(
            f'<div style="background:#16161D;border:1px solid #22222E;border-radius:8px;'
            f'padding:12px 14px;margin-top:8px;font-size:13px;color:#fff;line-height:1.6;'
            f'max-height:180px;overflow-y:auto;">{answer}</div>',
            unsafe_allow_html=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # header
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;padding:4px 0 14px;">'
        '<span style="font-size:22px;font-weight:800;letter-spacing:-.03em;">'
        'PitLane<span style="color:#00FF87;">AI</span></span>'
        '<span style="font-size:12px;color:#9090A0;">Vision-only AI race engineer · No sensors · No hardware · Just video</span>'
        '</div>',
        unsafe_allow_html=True)

    # controls
    c1, c2, c3 = st.columns([2.5, 0.8, 0.55])
    with c1:
        uploaded = st.file_uploader("", type=["mp4","mov","avi"], label_visibility="collapsed")
    with c2:
        sport = st.radio("", ["Karting","Biking"], horizontal=True,
                         label_visibility="collapsed").lower()
    with c3:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        go = st.button("▶  Analyze", disabled=uploaded is None, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if go and uploaded:
        os.makedirs(CV_DIR, exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "coaching", "output"), exist_ok=True)
        open(VIDEO_PATH, "wb").write(uploaded.read())
        try:
            with st.spinner("Step 1/2 — CV pipeline: detection · tracking · overlay…"):
                run_cv_pipeline(sport)
            with st.spinner("Step 2/2 — Gemini watching your footage · Groq building report…"):
                run_coaching_pipeline()
            for k in ("seek_to","eng_answer","eng_question"):
                st.session_state.pop(k, None)
            st.success("Analysis complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
        return

    # ── results ───────────────────────────────────────────────────────────────
    structured = load_structured()
    overlay_exists = os.path.exists(OVERLAY_PATH)

    if not overlay_exists and structured is None:
        st.markdown(
            '<div style="margin-top:80px;text-align:center;">'
            '<div style="font-size:48px;margin-bottom:14px;">🏎</div>'
            '<div style="font-size:15px;font-weight:600;color:#9090A0;">Upload a race video and click Analyze.</div>'
            '</div>', unsafe_allow_html=True)
        return

    st.markdown('<hr style="border-color:#22222E;margin:10px 0;">', unsafe_allow_html=True)
    left, right = st.columns([11, 9])

    # ── LEFT ──────────────────────────────────────────────────────────────────
    with left:
        errors   = structured.get("errors", [])      if structured else []
        moments  = structured.get("best_moments", []) if structured else []
        seek_to  = st.session_state.get("seek_to", 0)

        if overlay_exists:
            with open(OVERLAY_PATH, "rb") as vf:
                st.video(vf.read(), format="video/mp4", start_time=seek_to)
            dur = video_duration(OVERLAY_PATH)
            if dur > 0 and structured:
                st.markdown(marker_bar(errors, moments, dur), unsafe_allow_html=True)
        else:
            st.warning("Overlay video not yet generated.")

        if structured and structured.get("scores"):
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            score_cards(structured["scores"])

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with right:
        if not structured:
            st.info("Run analysis to see coaching data.")
            return

        errs = structured.get("errors", [])
        moms = structured.get("best_moments", [])

        tab_e, tab_m, tab_c = st.tabs([
            f"⚠️  Errors ({len(errs)})",
            f"⚡  Best Moments ({len(moms)})",
            "📋  Coaching",
        ])

        hint = ('<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
                'letter-spacing:.07em;color:#55555F;margin-bottom:10px;">'
                'Click a timestamp to jump to that moment</p>')

        with tab_e:
            st.markdown(hint, unsafe_allow_html=True)
            if errs:
                for i, e in enumerate(errs):
                    event_row(e["timestamp"], e["description"],
                              f"err_{i}", e.get("seconds", 0), "#FF3B3B")
            else:
                st.markdown('<p style="color:#9090A0;font-size:13px;">No errors detected.</p>',
                            unsafe_allow_html=True)

        with tab_m:
            st.markdown(hint, unsafe_allow_html=True)
            if moms:
                for i, m in enumerate(moms):
                    event_row(m["timestamp"], m["description"],
                              f"mom_{i}", m.get("seconds", 0), "#00FF87")
            else:
                st.markdown('<p style="color:#9090A0;font-size:13px;">No highlights detected.</p>',
                            unsafe_allow_html=True)

        with tab_c:
            # session summary
            summary = structured.get("session_summary", "")
            if summary:
                st.markdown(
                    f'<div style="background:rgba(0,255,135,0.04);border:1px solid rgba(0,255,135,0.1);'
                    f'border-radius:10px;padding:12px 16px;margin-bottom:12px;font-size:13px;color:#fff;line-height:1.6;">'
                    f'{summary}</div>', unsafe_allow_html=True)

            # driver style
            ds = structured.get("driver_style")
            if ds:
                driver_style_card(ds)

            # full analysis
            analysis = structured.get("coaching_analysis", "")
            if analysis:
                st.markdown(
                    '<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
                    'letter-spacing:.07em;color:#55555F;margin-bottom:6px;">Full Analysis</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:13px;color:#fff;line-height:1.7;">{analysis}</div>',
                    unsafe_allow_html=True)

            if os.path.exists(REPORT_PATH):
                st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
                with open(REPORT_PATH, encoding="utf-8") as f:
                    report_md = f.read()
                st.download_button("⬇  Download report.md", report_md.encode(),
                                   "pitlane_report.md", "text/markdown")

        # ask the engineer — below tabs, inside right column
        if structured:
            ask_section(structured)


if __name__ == "__main__":
    main()
