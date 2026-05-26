import os
import json

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CV_DIR = os.path.join(BASE_DIR, "cv")
COACHING_OUT = os.path.join(BASE_DIR, "coaching", "output")

def _video_path(sport: str) -> str:
    return os.path.join(CV_DIR, f"{sport}_video.mp4")

def _structured_path(sport: str) -> str:
    return os.path.join(COACHING_OUT, f"{sport}_structured.json")

def _report_path(sport: str) -> str:
    return os.path.join(COACHING_OUT, f"{sport}_report.md")

st.set_page_config(page_title="PitLane AI", layout="wide", page_icon="🏎")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif;background:#08080C;color:#fff;}
.stApp{background:#08080C;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1rem 1.5rem;max-width:100%;}

.stTabs [data-baseweb="tab-list"]{background:#111116;border-bottom:1px solid #22222E;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#9090A0;font-size:13px;font-weight:600;padding:12px 16px;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#fff;border-bottom:2px solid #00FF87;background:transparent;}
.stTabs [data-baseweb="tab-panel"]{background:#111116;padding:0;}

.stButton>button{background:transparent;border:1px solid #22222E;color:#9090A0;border-radius:6px;font-size:12px;font-weight:600;padding:4px 10px;transition:all .15s;}
.stButton>button:hover{border-color:#00FF87;color:#00FF87;background:rgba(0,255,135,0.05);}
.sport-btn>button{background:#16161D!important;border:1px solid #22222E!important;color:#9090A0!important;border-radius:8px!important;font-size:12px!important;font-weight:600!important;padding:6px 14px!important;width:100%!important;}
.sport-btn>button:hover{border-color:#00FF87!important;color:#00FF87!important;background:rgba(0,255,135,0.05)!important;}
.sport-btn-active>button{background:rgba(0,255,135,0.1)!important;border:1px solid #00FF87!important;color:#00FF87!important;border-radius:8px!important;font-size:12px!important;font-weight:700!important;padding:6px 14px!important;width:100%!important;}

.analyze-btn>button{background:#00FF87!important;color:#000!important;border:none!important;font-weight:700!important;font-size:14px!important;border-radius:8px!important;padding:10px 24px!important;}
.analyze-btn>button:hover{background:#00CC6A!important;box-shadow:0 0 20px rgba(0,255,135,0.3)!important;}
.analyze-btn>button:disabled{background:#1a3d2b!important;color:#3a7a55!important;}

.send-btn>button{background:#FF3B3B!important;color:#fff!important;border:none!important;font-weight:600!important;border-radius:6px!important;padding:6px 16px!important;font-size:13px!important;}
.send-btn>button:hover{background:#cc2f2f!important;}
.ts-btn>button{background:rgba(0,255,135,0.08)!important;border:1px solid rgba(0,255,135,0.2)!important;color:#00FF87!important;border-radius:4px!important;font-size:11px!important;font-weight:700!important;font-family:monospace!important;padding:1px 7px!important;line-height:1.4!important;}
.ts-btn>button:hover{background:rgba(0,255,135,0.2)!important;border-color:#00FF87!important;}

[data-testid="stFileUploader"]{background:#16161D;border:1px dashed #2E2E3E;border-radius:10px;}
.stTextInput>div>div>input{background:#16161D!important;border:1px solid #22222E!important;color:#fff!important;border-radius:6px!important;font-size:13px!important;}
.stTextInput>div>div>input:focus{border-color:#00FF87!important;box-shadow:none!important;}
.stTextInput>div>div>input::placeholder{color:#55555F!important;}

[data-testid="stMetric"]{background:#16161D;border:1px solid #22222E;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-size:26px;font-weight:800;}
[data-testid="stMetricLabel"]{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#9090A0;}
hr{border-color:#22222E;}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_structured(sport: str) -> dict | None:
    p = _structured_path(sport)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def run_coaching_pipeline(sport: str, video_path: str) -> None:
    from coaching.caption import caption
    from coaching.report import generate_report
    os.makedirs(CV_DIR, exist_ok=True)
    # Save video to sport-specific path
    import shutil
    shutil.copy2(video_path, _video_path(sport))
    with open(os.path.join(CV_DIR, "events.json"), "w") as f:
        json.dump({"sport": sport, "video_path": _video_path(sport), "fps": 10, "events": []}, f)
    caption()
    generate_report(sport)

def score_color(v: int) -> str:
    return "#00FF87" if v >= 80 else "#FFD426" if v >= 60 else "#FF3B3B"


# ── score cards ───────────────────────────────────────────────────────────────

def score_cards(scores: dict) -> None:
    cols = st.columns(len(scores))
    for col, (k, v) in zip(cols, scores.items()):
        col.markdown(
            f'<div style="background:#16161D;border:1px solid #22222E;border-radius:10px;'
            f'padding:12px;text-align:center;">'
            f'<div style="font-size:26px;font-weight:800;color:{score_color(v)};letter-spacing:-.03em;">{v}</div>'
            f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
            f'color:#9090A0;margin-top:3px;">{k.replace("_"," ").title()}</div></div>',
            unsafe_allow_html=True)


# ── driver style ──────────────────────────────────────────────────────────────

SENTIMENT_STYLE = {
    "positive": ("rgba(0,255,135,0.08)",  "rgba(0,255,135,0.18)", "#00FF87"),
    "negative": ("rgba(255,59,59,0.08)",   "rgba(255,59,59,0.18)",  "#FF3B3B"),
    "neutral":  ("rgba(255,212,38,0.08)",  "rgba(255,212,38,0.18)", "#FFD426"),
}

def driver_style_card(ds: dict) -> None:
    chips = ""
    for t in ds.get("tags", []):
        bg, border, color = SENTIMENT_STYLE.get(t.get("sentiment", "neutral"), SENTIMENT_STYLE["neutral"])
        chips += (f'<span style="display:inline-flex;align-items:center;gap:4px;background:{bg};'
                  f'border:1px solid {border};color:{color};border-radius:6px;padding:3px 10px;'
                  f'font-size:12px;font-weight:500;margin:3px 4px 3px 0;">'
                  f'{t.get("emoji","")} {t.get("label","")}</span>')
    st.markdown(
        f'<div style="background:rgba(0,255,135,0.03);border:1px solid rgba(0,255,135,0.1);'
        f'border-radius:10px;padding:14px 16px;margin-bottom:14px;">'
        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        f'color:#55555F;margin-bottom:6px;">Driver Style</div>'
        f'<div style="font-size:15px;font-weight:700;color:#fff;margin-bottom:10px;">'
        f'{ds.get("archetype","")}</div><div>{chips}</div></div>',
        unsafe_allow_html=True)


# ── event rows ────────────────────────────────────────────────────────────────

def event_row(ts: str, desc: str, key: str, seconds: int = 0) -> None:
    c1, c2 = st.columns([0.14, 0.86])
    with c1:
        st.markdown('<div class="ts-btn">', unsafe_allow_html=True)
        if st.button(ts, key=f"seek_{key}"):
            st.session_state["video_seek"] = seconds
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<p style="font-size:13px;color:#fff;margin:2px 0 0;">{desc}</p>',
                    unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#22222E;margin:7px 0;">', unsafe_allow_html=True)


# ── coaching tab ─────────────────────────────────────────────────────────────

def _grade(v: int) -> tuple[str, str]:
    if v >= 90: return "A+", "#00FF87"
    if v >= 80: return "A",  "#00FF87"
    if v >= 70: return "B",  "#00FF87"
    if v >= 60: return "C",  "#FFD426"
    if v >= 50: return "D",  "#FF3B3B"
    return "F", "#FF3B3B"

def _section_label(text: str) -> None:
    st.markdown(
        f'<p style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;'
        f'color:#55555F;margin:14px 0 6px;">{text}</p>',
        unsafe_allow_html=True)

def _ts_pill(ts: str) -> str:
    return (f'<span style="display:inline-block;background:rgba(0,255,135,0.08);'
            f'border:1px solid rgba(0,255,135,0.2);border-radius:4px;padding:1px 7px;'
            f'font-size:11px;font-weight:700;color:#00FF87;font-family:monospace;'
            f'margin-right:6px;">{ts}</span>')

def coaching_tab(s: dict) -> None:
    # Session summary
    summary = s.get("session_summary", "")
    if summary:
        st.markdown(
            f'<div style="background:rgba(0,255,135,0.04);border:1px solid rgba(0,255,135,0.1);'
            f'border-radius:10px;padding:12px 16px;margin:10px 0 4px;font-size:13px;'
            f'color:#fff;line-height:1.6;">{summary}</div>',
            unsafe_allow_html=True)

    # Driver style
    ds = s.get("driver_style")
    if ds:
        driver_style_card(ds)

    # Performance grades
    scores = s.get("scores", {})
    if scores:
        _section_label("Performance Grades")
        grade_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:4px;">'
        for k, v in scores.items():
            letter, color = _grade(v)
            label = k.replace("_", " ").title()
            grade_html += (
                f'<div style="background:#16161D;border:1px solid #22222E;border-radius:8px;'
                f'padding:10px 14px;min-width:90px;text-align:center;">'
                f'<div style="font-size:22px;font-weight:800;color:{color};">{letter}</div>'
                f'<div style="font-size:10px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.06em;color:#9090A0;margin-top:2px;">{label}</div>'
                f'<div style="font-size:11px;color:#55555F;margin-top:1px;">{v}/100</div>'
                f'</div>')
        grade_html += '</div>'
        st.markdown(grade_html, unsafe_allow_html=True)

    # Key errors with timestamps
    errs = s.get("errors", [])
    if errs:
        _section_label(f"Key Errors — {len(errs)} flagged")
        for i, e in enumerate(errs):
            event_row(e["timestamp"], e["description"], f"cerr_{i}", e.get("seconds", 0))

    # Best moments with timestamps
    moms = s.get("best_moments", [])
    if moms:
        _section_label(f"Best Moments — {len(moms)} highlighted")
        for i, m in enumerate(moms):
            event_row(m["timestamp"], m["description"], f"cmom_{i}", m.get("seconds", 0))

    # Coaching analysis — split into paragraphs, not a wall of text
    analysis = s.get("coaching_analysis", "")
    if analysis:
        _section_label("Coach's Analysis")
        paragraphs = [p.strip() for p in analysis.split("\n") if p.strip()]
        for para in paragraphs:
            st.markdown(
                f'<p style="font-size:13px;color:#CCCCCC;line-height:1.7;margin-bottom:10px;">'
                f'{para}</p>',
                unsafe_allow_html=True)


# ── ask the engineer ──────────────────────────────────────────────────────────

QUICK_QUESTIONS = [
    "Where am I losing the most time?",
    "How's my braking technique?",
    "What should I focus on next session?",
    "Where are my strongest moments?",
]

def ask_section(structured: dict) -> None:
    st.markdown('<hr style="border-color:#22222E;margin:14px 0 10px;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        'color:#55555F;margin-bottom:8px;">Ask the Engineer</div>',
        unsafe_allow_html=True)

    q_cols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        if q_cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state["eng_question"] = q
            st.session_state.pop("eng_answer", None)
            st.rerun()

    question = st.text_input("", placeholder="Ask anything about this session…",
                              value=st.session_state.get("eng_question", ""),
                              key="eng_input", label_visibility="collapsed")

    send_col, _ = st.columns([0.2, 0.8])
    with send_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Ask →", disabled=not question.strip(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if send and question.strip():
        with st.spinner("Analysing…"):
            try:
                from coaching.report import ask_engineer
                st.session_state["eng_answer"] = ask_engineer(question.strip(), structured)
                st.session_state["eng_question"] = question.strip()
            except Exception as e:
                st.session_state["eng_answer"] = f"Error: {e}"
        st.rerun()

    answer = st.session_state.get("eng_answer")
    if answer:
        st.markdown(
            f'<div style="background:#16161D;border:1px solid #22222E;border-radius:8px;'
            f'padding:12px 14px;margin-top:8px;font-size:13px;color:#fff;line-height:1.6;">'
            f'{answer}</div>',
            unsafe_allow_html=True)


# ── video player with timeline markers ───────────────────────────────────────

def video_player_with_markers(sport: str, structured: dict) -> None:
    import json as _json
    import streamlit.components.v1 as components

    errors  = structured.get("errors", [])
    moments = structured.get("best_moments", [])

    markers = []
    for e in errors:
        markers.append({"s": e.get("seconds", 0), "t": "error",
                        "d": e.get("description", "")[:80]})
    for m in moments:
        markers.append({"s": m.get("seconds", 0), "t": "moment",
                        "d": m.get("description", "")[:80]})

    markers_json = _json.dumps(markers)

    html = f"""
<style>
  body{{margin:0;background:#08080C;}}
  #wrap{{position:relative;width:100%;font-family:Inter,sans-serif;}}
  #vid{{width:100%;border-radius:10px 10px 0 0;display:block;background:#000;max-height:340px;}}
  #tl-outer{{background:#16161D;border:1px solid #22222E;border-top:none;
             border-radius:0 0 10px 10px;padding:10px 12px 8px;position:relative;}}
  #tl-track{{position:relative;height:4px;background:#2E2E3E;border-radius:2px;margin:0 0 6px;cursor:pointer;}}
  #tl-fill{{position:absolute;left:0;top:0;height:100%;background:#00FF87;border-radius:2px;width:0%;transition:width .1s linear;}}
  #tl-head{{position:absolute;top:-4px;width:12px;height:12px;background:#00FF87;border-radius:50%;
            transform:translateX(-50%);left:0%;transition:left .1s linear;pointer-events:none;}}
  .mk{{position:absolute;top:50%;transform:translate(-50%,-50%);
       width:10px;height:10px;border-radius:50%;cursor:pointer;z-index:2;transition:transform .15s;}}
  .mk:hover{{transform:translate(-50%,-50%) scale(1.6);}}
  .mk.error{{background:#FF3B3B;box-shadow:0 0 6px rgba(255,59,59,.6);}}
  .mk.moment{{background:#00FF87;box-shadow:0 0 6px rgba(0,255,135,.5);}}
  #tooltip{{position:fixed;background:#1A1A24;border:1px solid #2E2E3E;border-radius:6px;
            padding:6px 10px;font-size:11px;color:#fff;max-width:220px;line-height:1.4;
            pointer-events:none;display:none;z-index:9999;}}
  #tl-legend{{display:flex;gap:14px;}}
  .leg{{display:flex;align-items:center;gap:5px;font-size:10px;color:#9090A0;}}
  .leg-dot{{width:8px;height:8px;border-radius:50%;}}
</style>
<div id="wrap">
  <video id="vid" controls></video>
  <div id="tl-outer">
    <div id="tl-track">
      <div id="tl-fill"></div>
      <div id="tl-head"></div>
    </div>
    <div id="tl-legend">
      <div class="leg"><div class="leg-dot" style="background:#FF3B3B;"></div>Errors</div>
      <div class="leg"><div class="leg-dot" style="background:#00FF87;"></div>Best Moments</div>
    </div>
  </div>
</div>
<div id="tooltip"></div>
<script>
const vid = document.getElementById('vid');
vid.src = window.location.origin + '/app/static/{sport}_video.mp4';
const track = document.getElementById('tl-track');
const fill  = document.getElementById('tl-fill');
const head  = document.getElementById('tl-head');
const tip   = document.getElementById('tooltip');
const markers = {markers_json};

function pct(s){{ return (s / (vid.duration||1)) * 100; }}

function buildMarkers(){{
  markers.forEach(function(m){{
    const dot = document.createElement('div');
    dot.className = 'mk ' + (m.t==='error'?'error':'moment');
    dot.style.left = pct(m.s) + '%';
    dot.addEventListener('click', function(e){{
      e.stopPropagation();
      vid.currentTime = m.s;
      vid.play();
    }});
    dot.addEventListener('mouseenter', function(e){{
      tip.textContent = m.d;
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top  = (e.clientY - 30) + 'px';
    }});
    dot.addEventListener('mousemove', function(e){{
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top  = (e.clientY - 30) + 'px';
    }});
    dot.addEventListener('mouseleave', function(){{ tip.style.display='none'; }});
    track.appendChild(dot);
  }});
}}

vid.addEventListener('loadedmetadata', buildMarkers);
vid.addEventListener('timeupdate', function(){{
  const p = (vid.currentTime / (vid.duration||1)) * 100;
  fill.style.width = p + '%';
  head.style.left  = p + '%';
}});

track.addEventListener('click', function(e){{
  const rect = track.getBoundingClientRect();
  const p = (e.clientX - rect.left) / rect.width;
  vid.currentTime = p * (vid.duration||0);
  vid.play();
}});
</script>
"""
    components.html(html, height=420)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;padding:4px 0 14px;">'
        '<span style="font-size:22px;font-weight:800;letter-spacing:-.03em;">'
        'PitLane<span style="color:#00FF87;">AI</span></span>'
        '<span style="font-size:12px;color:#9090A0;">Vision-only AI race engineer · No sensors · No hardware · Just video</span>'
        '</div>', unsafe_allow_html=True)

    # sport selector (persists across reruns)
    if "sport" not in st.session_state:
        st.session_state["sport"] = "karting"

    s1, s2, s3 = st.columns([1, 1, 1])
    for col, label in zip([s1, s2, s3], ["Karting", "Biking", "Cycling"]):
        is_active = st.session_state["sport"] == label.lower()
        css_class = "sport-btn-active" if is_active else "sport-btn"
        col.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if col.button(label, key=f"sport_{label}", use_container_width=True):
            st.session_state["sport"] = label.lower()
            st.session_state.pop("eng_answer", None)
            st.session_state.pop("eng_question", None)
            st.rerun()
        col.markdown('</div>', unsafe_allow_html=True)

    sport = st.session_state["sport"]

    # controls
    c1, c3 = st.columns([3.5, 0.55])
    with c1:
        uploaded = st.file_uploader("", type=["mp4","mov","avi"], label_visibility="collapsed")
    with c3:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        go = st.button("▶  Analyze", disabled=uploaded is None, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if go and uploaded:
        os.makedirs(CV_DIR, exist_ok=True)
        os.makedirs(COACHING_OUT, exist_ok=True)
        tmp_path = os.path.join(CV_DIR, "upload_tmp.mp4")
        open(tmp_path, "wb").write(uploaded.read())
        try:
            with st.spinner("Gemini is watching your footage — finding errors and best moments…"):
                run_coaching_pipeline(sport, tmp_path)
            for k in ("eng_answer", "eng_question"):
                st.session_state.pop(k, None)
            st.success("Analysis complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
        return

    structured = load_structured(sport)

    if not structured:
        st.markdown(
            '<div style="margin-top:80px;text-align:center;">'
            '<div style="font-size:48px;margin-bottom:14px;">🏎</div>'
            '<div style="font-size:15px;font-weight:600;color:#9090A0;">'
            'Upload a race video and click Analyze.</div></div>',
            unsafe_allow_html=True)
        return

    st.markdown('<hr style="border-color:#22222E;margin:10px 0;">', unsafe_allow_html=True)

    left, right = st.columns([11, 9])

    # ── LEFT: video player with timeline markers ─────────────────────────────
    with left:
        vpath = _video_path(sport)
        if os.path.exists(vpath):
            video_player_with_markers(sport, structured)
        else:
            st.markdown(
                '<div style="background:#16161D;border:1px solid #22222E;border-radius:12px;'
                'height:300px;display:flex;align-items:center;justify-content:center;'
                'color:#55555F;font-size:13px;">No video uploaded yet</div>',
                unsafe_allow_html=True)


    # ── RIGHT: tabs + ask the engineer ───────────────────────────────────────
    with right:
        errs = structured.get("errors", [])
        moms = structured.get("best_moments", [])

        tab_e, tab_m, tab_c = st.tabs([
            f"⚠️  Errors ({len(errs)})",
            f"⚡  Best Moments ({len(moms)})",
            "📋  Coaching",
        ])

        with tab_e:
            st.markdown(
                '<p style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
                'color:#55555F;margin-bottom:10px;padding-top:12px;">Errors detected in this session</p>',
                unsafe_allow_html=True)
            if errs:
                for i, e in enumerate(errs):
                    event_row(e["timestamp"], e["description"], f"err_{i}", e.get("seconds", 0))
            else:
                st.markdown('<p style="color:#9090A0;font-size:13px;">No errors detected.</p>', unsafe_allow_html=True)

        with tab_m:
            st.markdown(
                '<p style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
                'color:#55555F;margin-bottom:10px;padding-top:12px;">Best moments from this session</p>',
                unsafe_allow_html=True)
            if moms:
                for i, m in enumerate(moms):
                    event_row(m["timestamp"], m["description"], f"mom_{i}", m.get("seconds", 0))
            else:
                st.markdown('<p style="color:#9090A0;font-size:13px;">No highlights detected.</p>', unsafe_allow_html=True)

        with tab_c:
            coaching_tab(structured)

        ask_section(structured)


if __name__ == "__main__":
    main()
