"""T-ChatKBQA interactive demo.

This is an offline presentation/demo surface. It does not run the full
LLM + Freebase pipeline live. Instead it combines:
  - real temporal-signal detection from src.agent
  - recorded trial artifacts from the locked v1 experiment
  - static architecture/ablation summaries for presentation

The full system requires a much heavier environment: model checkpoints,
Freebase/Virtuoso backend, retrieval assets, and external services.
"""

import streamlit as st
import sys
import os
import time
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import detect_temporal_signals

# — Page config ———
st.set_page_config(
    page_title="T-ChatKBQA",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# — Custom CSS ———
st.markdown("""
<style>
    :root {
        --bg: #edf4fb;
        --card: #ffffff;
        --ink: #102033;
        --muted: #506176;
        --line: #c8d7e6;
        --accent: #0f766e;
        --accent-soft: #d9f3ef;
        --ok: #166534;
        --warn: #b45309;
        --bad: #b91c1c;
        --blue-soft: #dbeafe;
        --sand: #e8f0f8;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, #d9ecff 0, rgba(217,236,255,0) 32%),
            linear-gradient(180deg, #f7fbff 0%, #eaf2f9 100%);
        color: var(--ink);
    }
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp span,
    .stApp .stMarkdown,
    .stApp .stCaption,
    .stApp .stAlert,
    .stApp .stExpander,
    .stApp .stMetric,
    .stApp .stTabs,
    .stApp .stDataFrame,
    .stApp .stTable {
        color: var(--ink);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbff 0%, #edf4fb 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--ink);
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stAlertContainer"] *,
    div[data-testid="stMetric"] *,
    div[data-testid="stExpander"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stTextArea"] > div,
    div[data-testid="stExpander"] {
        color: var(--ink);
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #b9cde0 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: var(--muted);
    }
    .stApp code {
        background: #d9f3ef;
        color: #0f5132;
        padding: 0.12rem 0.35rem;
        border-radius: 0.35rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%);
        color: #f8fcff;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0;
        color: #dcedff;
        font-size: 1rem;
    }
    .notice {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0 20px 0;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .status-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
    }
    .status-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .status-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 4px;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .agent-step {
        padding: 13px 16px;
        border-radius: 12px;
        margin: 8px 0;
        font-family: 'SF Mono', 'Courier New', monospace;
        font-size: 14px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .step-detect { background: #e8f1fb; border-left: 4px solid #2563eb; }
    .step-route  { background: #fdf1df; border-left: 4px solid #ea580c; }
    .step-exec   { background: #f8e8ec; border-left: 4px solid #be185d; }
    .step-return { background: #e9f5ec; border-left: 4px solid #15803d; }
    .eyebrow {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .prompt-card {
        background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        margin: 8px 0 18px 0;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }
    .prompt-hint {
        color: var(--muted);
        font-size: 0.93rem;
        margin-top: -2px;
        margin-bottom: 10px;
    }
    .chip-row {
        margin: 4px 0 10px 0;
        color: var(--muted);
        font-size: 0.9rem;
    }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #b9cde0 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border: 1px solid #0f766e !important;
        box-shadow: 0 0 0 1px #0f766e !important;
    }
    div[data-testid="stCodeBlock"] {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
    }
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCode"] pre {
        color: #e5eef8 !important;
        background: #0f172a !important;
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: white;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #115e59;
        background: #115e59;
        color: white;
    }
    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stTable"] table {
        color: var(--ink) !important;
    }
</style>
""", unsafe_allow_html=True)


# — Pre-computed curated examples ———
CURATED = [
    {
        "question": "Who was the US president before Obama?",
        "type": "before",
        "signals": ["before"],
        "pred_sexpr": "(ARGMIN (JOIN (R government.government_position_held) m.02mjmr) government.government_position_held.from)",
        "hallucinated_rel": "government.government_position_held",
        "grounded_rel": "/government/government_position_held",
        "correct_rel": "/government/politician/government_positions_held",
        "gold_answer": ["George W. Bush"],
        "error": "Relation wrong domain — model used generic 'position_held' instead of 'politician/positions_held'",
    },
    {
        "question": "What was the earliest album associated with The Rolling Stones?",
        "type": "first",
        "signals": ["earliest"],
        "pred_sexpr": "(ARGMIN (JOIN (R music.artist.album) m.07mvp) music.artist.album)",
        "hallucinated_rel": None,
        "grounded_rel": "/music/artist/album",
        "correct_rel": "/music/artist/album",
        "gold_answer": ["The Rolling Stones (EP)"],
        "error": None,
    },
    {
        "question": "which dawkins book to read first?",
        "type": "first",
        "signals": ["first"],
        "pred_sexpr": "(ARGMIN (JOIN (R book.author.works_written) m.05w2x0) book.author.works_written)",
        "hallucinated_rel": None,
        "grounded_rel": "/book/author/works_written",
        "correct_rel": "/book/author/works_written",
        "gold_answer": ["The Selfish Gene"],
        "error": None,
    },
    {
        "question": "who is the leader of japan right now?",
        "type": "explicit_time",
        "signals": ["right now"],
        "pred_sexpr": "(JOIN (R government.head_of_state.country.leader) m.0bq_h)",
        "hallucinated_rel": "government.head_of_state.country.leader",
        "grounded_rel": None,
        "correct_rel": "/government/politician/government_positions_held",
        "gold_answer": ["Shinzō Abe"],
        "error": "Complete hallucination — 'head_of_state.country.leader' not in Freebase",
    },
    {
        "question": "who was the first jedi master?",
        "type": "first",
        "signals": ["first"],
        "pred_sexpr": "(ARGMIN (JOIN (R film.director.jedi_master) m.star_wars.film.director.jedi_master) film.director.jedi_master)",
        "hallucinated_rel": "film.director.jedi_master",
        "grounded_rel": None,
        "correct_rel": None,
        "gold_answer": [],
        "error": "Fictional question — 'jedi_master' relation doesn't exist in Freebase. Model should answer 'no data'",
    },
    {
        "question": "what team is reggie bush on 2011?",
        "type": "during",
        "signals": ["2011"],
        "pred_sexpr": "(ARGMIN (JOIN (R sports.pro_athlete.team) m.010l89) sports.pro_athlete.team)",
        "hallucinated_rel": "sports.pro_athlete.team",
        "grounded_rel": "/sports/pro_athlete/teams",
        "correct_rel": "/sports/pro_athlete/teams",
        "gold_answer": ["Miami Dolphins"],
        "error": "Relation almost correct (singular 'team' vs plural 'teams') — fixed by fuzzy matching",
    },
]


@st.cache_resource(show_spinner=False)
def get_live_agent_and_status():
    config_path = os.environ.get("TKBQA_CONFIG", "configs/inference.yaml")
    if not os.path.exists(config_path):
        return None, {
            "ready": False,
            "mode": "offline",
            "reason": f"Missing config: {config_path}",
        }

    try:
        from src.pipeline import TemporalKBQAPipeline
        from src.agent import TemporalQuestionAgent

        pipeline = TemporalKBQAPipeline.from_config(config_path)
        runtime = pipeline.runtime_status()
        ready = runtime["model_exists"]
        if runtime.get("adapter_only_checkpoint") and not runtime.get("base_model_exists"):
            ready = False
        reason_parts = []
        if not runtime["model_exists"]:
            reason_parts.append("model checkpoint not found")
        if runtime.get("adapter_only_checkpoint") and not runtime.get("base_model_exists"):
            reason_parts.append("base LLaMA model missing for LoRA adapter")
        if not runtime.get("entity_list_exists"):
            reason_parts.append("entity list file missing")
        if not runtime.get("surface_map_exists"):
            reason_parts.append("surface map file missing")
        if runtime.get("surface_index_error"):
            reason_parts.append("surface index unavailable")
        if runtime.get("llm_error"):
            reason_parts.append("LLM load error")
        return (
            TemporalQuestionAgent(pipeline),
            {
                "ready": ready,
                "mode": "live" if ready else "degraded-live",
                "reason": ", ".join(reason_parts) if reason_parts else "live pipeline available",
                "runtime": runtime,
            },
        )
    except Exception as e:
        return None, {
            "ready": False,
            "mode": "offline",
            "reason": str(e),
        }


def render_sidebar():
    with st.sidebar:
        st.markdown("## T-ChatKBQA")
        st.caption("Offline demo for the final project submission")
        st.markdown("---")
        st.markdown("### Read This Demo As")
        st.caption("1. Real temporal-signal detection")
        st.caption("2. Recorded artifacts from the locked v1 run")
        st.caption("3. Honest diagnosis of where the pipeline breaks")
        st.markdown("---")
        st.markdown("### Locked V1 Snapshot")
        st.metric("Valid S-expressions", "87.6%", delta="1,174 / 1,340")
        st.metric("Best relation grounding", "88.9%", delta="with fuzzy grounding")
        st.metric("Answer-level F1", "0.0", delta="current blocker")
        st.markdown("---")
        st.caption("Based on ChatKBQA (ACL 2024)")
        st.caption("Primary benchmark: TempQuestions")


def render_overview():
    _, live_status = get_live_agent_and_status()
    st.markdown(
        """
        <div class="hero">
            <h1>T-ChatKBQA</h1>
            <p>Temporal KBQA built on top of ChatKBQA. This demo is strongest as an engineering walkthrough and ablation viewer, not as a live answer-quality showcase.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="notice">
            <div class="eyebrow">What This Page Is Showing</div>
            <div>
                The app mixes one live component (<code>detect_temporal_signals()</code>) with recorded outputs from the locked v1 experiment.
                It is designed to explain the system, the benchmark result, and the current failure boundary clearly during submission or demo.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if live_status["ready"]:
        st.success(
            "Live mode is available in this environment. The walkthrough tab can call the real agent pipeline instead of only showing recorded artifacts."
        )
    else:
        st.warning(
            "This Streamlit app is currently running in offline/degraded mode. "
            f"Live pipeline status: {live_status['reason']}"
        )
    st.warning(
        "This Streamlit app is a lightweight presentation layer, not the full live inference stack. "
        "A complete end-to-end run requires several heavyweight components that are intentionally not started inside this demo."
    )
    with st.expander("What is required to run the full system end-to-end?", expanded=False):
        st.markdown(
            """
            To run the real Temporal ChatKBQA pipeline instead of this offline demo, the environment must provide:

            - a fine-tuned LLaMA-2 checkpoint and the matching inference config
            - a running Freebase backend, typically Virtuoso with SPARQL on `localhost:8890/sparql` and ODBC on port `13001`
            - entity-linking and retrieval assets, including ELQ and FACC1-derived lookup resources
            - enough RAM / disk / GPU to serve the model and KB services reliably
            - dataset artifacts, label maps, and preprocessing outputs aligned with the selected config

            In other words, the Streamlit page is intentionally not pretending to be a one-command production deployment.
            It is a transparent demo surface for presentation: live signal detection, recorded trial outputs, architecture,
            and error analysis. The API and CLI are the real execution surfaces once the full environment is installed.
            """
        )
    st.markdown(
        """
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">Syntax generation</div>
                <div class="status-value">87.6%</div>
                <div class="status-note">Valid S-expressions from the v1 run</div>
            </div>
            <div class="status-card">
                <div class="status-label">Relation grounding</div>
                <div class="status-value">88.9%</div>
                <div class="status-note">Only after fuzzy grounding is added</div>
            </div>
            <div class="status-card">
                <div class="status-label">Entity grounding</div>
                <div class="status-value">5.8%</div>
                <div class="status-note">Main downstream bottleneck</div>
            </div>
            <div class="status-card">
                <div class="status-label">Answer-level benchmark</div>
                <div class="status-value">0.0</div>
                <div class="status-note">F1 / Hits@1 / Accuracy in locked v1</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_demo():
    live_agent, live_status = get_live_agent_and_status()
    st.header("Agent Walkthrough")
    st.caption("This tab uses the real agent pipeline when the environment is ready; otherwise it falls back to recorded trial artifacts.")
    if live_status["ready"]:
        st.success("Live agent path enabled. Generation, routing, and execution will use the configured model/backend.")
    else:
        st.info(
            "Live agent path is not fully ready in this environment. "
            "Temporal-signal detection runs live, while generation/grounding/execution fall back to recorded artifacts. "
            f"Reason: {live_status['reason']}"
        )

    if "live_question_input" not in st.session_state:
        st.session_state["live_question_input"] = ""
    if "submitted_question" not in st.session_state:
        st.session_state["submitted_question"] = ""
    if "example_select" not in st.session_state:
        st.session_state["example_select"] = "(keep current text)"

    def _load_question(sample: str) -> None:
        st.session_state["live_question_input"] = sample

    def _apply_selected_example() -> None:
        selected = st.session_state.get("example_select", "(keep current text)")
        if selected != "(keep current text)":
            st.session_state["live_question_input"] = selected

    st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
    st.markdown("**Ask a temporal question**")
    st.markdown(
        '<div class="prompt-hint">Use a date, ordering word, or temporal phrase such as <code>before</code>, <code>after</code>, <code>first</code>, <code>last</code>, or a year.</div>',
        unsafe_allow_html=True,
    )

    example_cols = st.columns(3)
    quick_examples = [
        "Who was the US president before Obama?",
        "What was the earliest album associated with The Rolling Stones?",
        "what team is reggie bush on 2011?",
    ]
    for idx, sample in enumerate(quick_examples):
        if example_cols[idx].button(sample, key=f"quick_example_{idx}", use_container_width=True):
            _load_question(sample)

    st.markdown('<div class="chip-row">Or pick a curated example from the list below.</div>', unsafe_allow_html=True)
    st.selectbox(
        "Curated examples",
        ["(keep current text)"] + [c["question"] for c in CURATED],
        key="example_select",
        on_change=_apply_selected_example,
    )

    st.text_area(
        "Question",
        placeholder="e.g. Who was CEO of Apple before Tim Cook?",
        key="live_question_input",
        height=100,
        label_visibility="collapsed",
    )
    question = st.session_state.get("live_question_input", "")

    action_cols = st.columns([0.22, 0.22, 0.56])
    analyze_clicked = action_cols[0].button("Analyze Question", use_container_width=True)
    clear_clicked = action_cols[1].button("Clear", use_container_width=True)
    if clear_clicked:
        st.session_state["live_question_input"] = ""
        st.session_state["submitted_question"] = ""
        st.session_state["example_select"] = "(keep current text)"
        st.rerun()
    if analyze_clicked:
        st.session_state["submitted_question"] = question.strip()

    st.markdown('</div>', unsafe_allow_html=True)

    question = st.session_state.get("submitted_question", "").strip()

    if not question:
        st.info("Enter a question, choose one of the examples, then click `Analyze Question`.")
        return

    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Reasoning trace**")

        with st.spinner("Running signal detection..."):
            time.sleep(0.2)
        signals = detect_temporal_signals(question)
        is_temporal = len(signals) > 0

        st.markdown(
            f'<div class="agent-step step-detect">'
            f'<b>STEP 1 — DETECT</b><br>'
            f'Question scanned for temporal cues.<br>'
            f'Signals found: <code>{signals if signals else "none"}</code><br>'
            f'Classification: <b>{"temporal" if is_temporal else "non-temporal"}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.1)

        route = "temporal" if is_temporal else "standard"
        st.markdown(
            f'<div class="agent-step step-route">'
            f'<b>STEP 2 — ROUTE</b><br>'
            f'Chosen pipeline: <b>{route}</b><br>'
            f'{"Expected operators: ARGMIN / ARGMAX / TC when available" if is_temporal else "Expected operator family: JOIN-oriented baseline flow"}'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.1)

        live_result = None
        if live_status["ready"] and live_agent is not None:
            try:
                with st.spinner("Running live agent pipeline..."):
                    live_result = live_agent.run(question)
                st.markdown(
                    f'<div class="agent-step step-exec">'
                    f'<b>STEP 3 — EXECUTE</b><br>'
                    f'Live pipeline executed with the configured model and backend.<br>'
                    f'Answers found: <b>{len(live_result.get("answer", []))}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="agent-step step-return">'
                    f'<b>STEP 4 — RETURN</b><br>'
                    f'Return mode for this demo: <b>live agent result</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Live execution failed: {e}")
        if live_result is None:
            st.markdown(
                f'<div class="agent-step step-exec">'
                f'<b>STEP 3 — EXECUTE</b><br>'
                f'In the real system this stage would call the generator, grounding logic, ELQ/FACC1 retrieval, and KB execution over Virtuoso.<br>'
                f'In this offline/degraded demo we only surface a recorded trial artifact if the question matches one of the curated examples.'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="agent-step step-return">'
                f'<b>STEP 4 — RETURN</b><br>'
                f'Return mode for this demo: <b>{"recorded trial case" if any(c["question"].lower() == question.lower() for c in CURATED) else "reasoning-only walkthrough"}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    matched = next((c for c in CURATED if c["question"].lower() == question.lower()), None)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Result panel**")
        if live_result is not None:
            answers = live_result.get("answer", [])
            if answers:
                st.success(f"Live answer(s): {answers}")
            else:
                st.warning("Live pipeline returned no answer.")
            if live_result.get("sparql_used"):
                st.caption("SPARQL executed by the live pipeline")
                st.code(live_result["sparql_used"], language="sparql")
            if live_result.get("temporal_constraint"):
                st.caption(f"Temporal constraint: {live_result['temporal_constraint']}")
            st.markdown("**Reasoning steps**")
            for step in live_result.get("reasoning_steps", []):
                st.caption(step)
        elif matched:
            st.caption("This block is pulled from a curated example in the locked v1 analysis set.")
            st.code(matched["pred_sexpr"], language="clojure")
            if matched["hallucinated_rel"]:
                st.error(f"Hallucinated relation: `{matched['hallucinated_rel']}`")
            else:
                st.success("Relation path is plausible in this example.")

            gold = matched["gold_answer"] if matched["gold_answer"] else ["no reliable KB answer"]
            st.markdown(f"**Gold / expected answer**: `{gold}`")
            if matched.get("grounded_rel"):
                st.caption(f"Grounded relation after retrieval step: `{matched['grounded_rel']}`")
            if matched.get("correct_rel"):
                st.caption(f"Reference relation for comparison: `{matched['correct_rel']}`")

            if matched.get("error"):
                st.warning(matched["error"])
            else:
                st.info("This is one of the cleaner cases in the recorded run.")
        else:
            st.info("No recorded artifact for this exact question. Pick a curated example to show a real trial case.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_ablation_explorer():
    st.header("Ablation Study Explorer")
    st.caption("The useful story of v1 is not final accuracy. It is the decomposition of where the pipeline fails.")

    # — Summary metrics row ———
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valid syntax", "87.6%", "generator learns structure")
    col2.metric("Relation grounding", "88.9%", "only with fuzzy grounding")
    col3.metric("Entity grounding", "5.8%", "current bottleneck")
    col4.metric("Best answered rate", "2.6%", "with golden entity")

    st.markdown("---")

    # — Comparison table ———
    st.subheader("Before vs After: Each Ablation Step")

    ablation_data = [
        {"Stage": "1. Generate-only (no retrieval)",
         "Relation Match": "0.5%", "Entity Match": "0%",
         "Answered": "0.4%", "F1": "0.00",
         "Insight": "Both entity & relation hallucinated → pipeline broken"},
        {"Stage": "2. +Fuzzy Relation Grounding",
         "Relation Match": "88.9%", "Entity Match": "5.8%",
         "Answered": "0.7%", "F1": "0.00",
         "Insight": "Relation fixed (+178x). Entity now the bottleneck."},
        {"Stage": "3. +Golden Entity (upper bound)",
         "Relation Match": "60.2%", "Entity Match": "100%",
         "Answered": "2.6%", "F1": "0.00",
         "Insight": "Entity fixed. KB coverage (22 rels) too small for TempQuestions."},
    ]

    st.dataframe(ablation_data, use_container_width=True, hide_index=True)

    # — Visual bottleneck shift ———
    st.markdown("---")
    st.subheader("Bottleneck Shift Visualization")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Trial 1: Raw")
        st.progress(0.005, text="Relation: 0.5%")
        st.progress(0.0, text="Entity: 0%")
        st.progress(0.004, text="Answered: 0.4%")
        st.caption("Generate-only baseline")
    with col2:
        st.markdown("### Trial 2: +Relation")
        st.progress(0.889, text="Relation: 88.9% ✅")
        st.progress(0.058, text="Entity: 5.8%")
        st.progress(0.007, text="Answered: 0.7%")
        st.caption("Relations improve sharply, entities do not")
    with col3:
        st.markdown("### Trial 3: +Gold Entity")
        st.progress(0.602, text="Relation: 60.2%")
        st.progress(1.0, text="Entity: 100% ✅")
        st.progress(0.026, text="Answered: 2.6%")
        st.caption("Coverage and relevance remain limiting")

    # — Key takeaways ———
    st.markdown("---")
    st.subheader("What V1 Proves")
    st.markdown("""
    1. **The generator learns syntax**: 87.6% valid S-expressions is a real signal.
    2. **Retrieval matters**: relation grounding changes executability dramatically.
    3. **Entity grounding is harder than relation grounding** in the current setup.
    4. **Coverage is too narrow**: 22 training relations are not enough for TempQuestions.
    5. **This is an engineering-valid v1, not a solved QA system**.
    """)


def render_error_gallery():
    st.header("Error Gallery: What The Model Actually Generated")
    st.caption("Recorded outputs from the locked run, grouped by failure mode for discussion.")

    error_tabs = st.tabs([
        "Hallucinated Relations",
        "Correct Predictions",
        "Invalid Syntax",
        "Temporal Failures",
    ])

    with error_tabs[0]:
        st.warning("**Dominant error family:** model invents non-existent or malformed Freebase relations")
        hallucinations = [
            ("who was the first jedi master?",
             "film.director.jedi_master",
             "Relation invented from Star Wars context — should return 'no data'"),
            ("who does david beckham play for in 2012?",
             "sports.sports_team_roster.athlete",
             "Close! 'sports_team_roster' exists but 'athlete' field name is wrong"),
            ("who is the leader of japan right now?",
             "government.head_of_state.country.leader",
             "Hierarchical hallucination — model chains plausible government subdomains"),
            ("how many people were at the 2006 fifa world cup",
             "sports.sports_team.tournaments.fifa_world_cup_tournament",
             "Overly specific — 'tournaments' path doesn't exist in Freebase schema"),
            ("who was london mayor before boris johnson?",
             "government.government_position_held.politician.who",
             "Appends 'who' as if it's a Freebase field — language model leakage"),
        ]
        for q, rel, note in hallucinations:
            with st.expander(f"Q: {q}"):
                st.markdown(f"**Hallucinated relation:** `{rel}`")
                st.caption(f"**Why:** {note}")

    with error_tabs[1]:
        st.success("**Useful counterexamples:** a few predictions are correct or nearly correct")
        for c in CURATED:
            if c.get("error") is None:
                with st.expander(f"Q: {c['question']}"):
                    st.code(c["pred_sexpr"], language="clojure")
                    st.success(f"Gold answer: {c['gold_answer']}")

    with error_tabs[2]:
        st.error("**Syntax failures:** malformed or truncated S-expressions")
        invalid_examples = [
            ("who is the current governor of arizona 2010?",
             "(ARGMIN (JOIN (R government.government_position.government_appointment.government_term.government_term.government_position.government_appointment.jOIN (M ...",
             "Recursive loop — model gets stuck repeating 'government_term.government_position...'"),
            ("who was the first jedi master? (beam 3)",
             '("',
             "Truncated — generation stopped mid-token"),
        ]
        for q, sexpr, note in invalid_examples:
            with st.expander(f"Q: {q}"):
                st.code(sexpr[:200], language="clojure")
                st.caption(f"**Why:** {note}")

    with error_tabs[3]:
        st.info("**Temporal-constraint gap:** TC operator is effectively absent in the current run")
        st.markdown("""
        **Reason:** Training data from Zenodo lacks date literals.
        Model only learned JOIN / ARGMIN / ARGMAX, never TC (temporal constraint).

        **Impact:** Questions with explicit dates ("in 2010", "during 2005") cannot be answered.

        **Fix (v2):** Source date literals from raw Freebase RDF dump or Wikidata cross-reference.
        """)
        tc_examples = [
            ("what happened to justin bieber 2012?", "Generated ARGMIN instead of TC"),
            ("which movie did steve hanft direct in 2004?", "Should use TC with year constraint"),
            ("who is the current governor of arizona 2010?", "Year constraint ignored"),
        ]
        for q, note in tc_examples:
            st.markdown(f"- **Q:** {q} → *{note}*")


def render_architecture():
    st.header("System Architecture")

    # — Pipeline visualization using columns ———
    st.subheader("End-to-End Pipeline (with real example)")
    st.caption("Trace for a representative temporal question. This panel explains the intended end-to-end system, not a live execution trace.")

    steps = [
        ("INPUT", "Natural-language question\nwith a temporal cue", "#e3f2fd"),
        ("AGENT", "Detect cue and route\ninto temporal or standard flow", "#fff3e0"),
        ("LLM", "LLaMA-2-7B + LoRA\ngenerates logical form", "#fce4ec"),
        ("GROUND", "Ground relation/entity\nagainst KB-supported space", "#f3e5f5"),
        ("EXECUTE", "Convert to SPARQL and\nquery KB backend", "#e8f5e9"),
        ("RETURN", "Answer, provenance,\nand failure reason", "#e0f2f1"),
    ]

    cols = st.columns(len(steps))
    for i, (title, desc, color) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f'<div style="background:{color};padding:12px;border-radius:8px;'
                f'text-align:center;min-height:120px;font-size:13px;">'
                f'<b>{title}</b><br><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown('<p style="text-align:center;margin:0;">→</p>', unsafe_allow_html=True)

    st.markdown("---")

    # — Tech stack ———
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temporal Operators")
        st.markdown("""
        | Operator | Semantics | SPARQL Pattern |
        |----------|-----------|---------------|
        | `TC` | Time Constraint range | `FILTER(?date >= from && ?date <= to)` |
        | `ARGMAX` | Most recent / last | `ORDER BY DESC(?date) LIMIT 1` |
        | `ARGMIN` | Earliest / first | `ORDER BY ASC(?date) LIMIT 1` |
        | `gt/lt` | After / Before | `FILTER(?date > X)` |
        """)

    with col2:
        st.subheader("Deployment")
        st.markdown("""
        | Method | Command |
        |--------|---------|
        | REST API | `uvicorn src.api:app --port 8000` |
        | CLI | `python -m src.cli --question "..."` |
        | Docker | `docker run -p 8000:8000 tchatkbqa` |
        | Streamlit | `streamlit run src/streamlit_app.py` |
        """)
        st.caption("This Streamlit app is a presentation/demo layer. The API/CLI are the real inference surfaces once the full model + KB environment is installed.")
        st.markdown(
            """
            **Important deployment note**

            The commands above are not equivalent in setup cost. `streamlit run src/streamlit_app.py` starts this lightweight demo page immediately,
            but a true end-to-end API/CLI run additionally depends on:

            - the temporal checkpoint
            - Freebase/Virtuoso services
            - ELQ entity linker
            - retrieval data such as FACC1 aliases and label maps
            - matching runtime dependencies and configuration files
            """
        )

    st.subheader("Key Modules")
    st.markdown("""
    | Module | Lines | Role |
    |--------|-------|------|
    | `src/agent.py` | 175 | Agentic routing: detect → route → execute → refine |
    | `src/pipeline.py` | ~300 | End-to-end inference (temporal + standard) |
    | `src/api.py` | ~80 | FastAPI with lazy LLM loading |
    | `executor/sparql_executor.py` | ~400 | KB queries via ODBC/SPARQL |
    | `executor/logic_form_util.py` | ~300 | S-expression → SPARQL conversion |
    | `parse_sparql_tempquestions.py` | ~200 | TempQuestions SPARQL → S-expr parser |
    | `scripts/eval_zenodo.py` | ~280 | Zenodo triple-based answer evaluation |
    | `scripts/grounded_eval.py` | ~270 | Fuzzy-match grounded evaluation |
    | `src/streamlit_app.py` | ~350 | This demo |
    """)


def main():
    render_sidebar()

    render_overview()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Walkthrough",
        "Ablation",
        "Error Gallery",
        "Architecture",
    ])

    with tab1:
        render_live_demo()
    with tab2:
        render_ablation_explorer()
    with tab3:
        render_error_gallery()
    with tab4:
        render_architecture()

    st.divider()
    st.caption(
        "T-ChatKBQA v1.0. LLaMA-2-7B + LoRA, 2,061 Zenodo-derived training examples, 22 relations, 268 TempQuestions test questions in the locked run. This Streamlit app is the lightweight demo surface; full inference requires the external model and KB stack."
    )


if __name__ == "__main__":
    main()
