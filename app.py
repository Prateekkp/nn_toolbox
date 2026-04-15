from pathlib import Path

import streamlit as st
from src.learner_pages.perceptron_ui import perceptron_page
from src.assets.documents.perceptron import perceptron_docs_page
from src.assets.documents.forward_propagation import forward_propagation_docs_page
from src.assets.documents.back_propagation import back_propagation_docs_page
from src.assets.documents.mnp import mnp_docs_page
from src.learner_pages.forward_propagation import forward_propagation_page
from src.learner_pages.backward_propagation import backward_propagation_page
from src.learner_pages.mlp import mlp_page
from src.ai_playground_pages.ask_ai import explore_data_page
from src.ai_playground_pages.personality_decoder import personality_decoder_page
from src.ai_playground_pages.palm import palm_reader_page
from src.application_pages.rnn.rnn_landing import rnn_application_page

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="NeuroLens",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Sidebar: Navigation
# ---------------------------
st.sidebar.title("Navigation")


def _reset_palm_state():
    for key in ["palm_phase", "palm_result", "palm_image_bytes", "palm_annotated_img", "palm_line_count"]:
        st.session_state.pop(key, None)


def _reset_personality_state():
    for key in [
        "pd_phase",
        "pd_name",
        "pd_name_lines",
        "pd_question_mode",
        "pd_age",
        "pd_profession",
        "pd_dynamic_step",
        "pd_dynamic_answers",
        "pd_current_question",
    ]:
        st.session_state.pop(key, None)


def _activate_ai_playground():
    st.session_state["learner_nav"] = None
    st.session_state["application_nav"] = None
    st.session_state["doc_nav"] = None


def _activate_learner():
    st.session_state["ai_playground_nav"] = None
    st.session_state["application_nav"] = None
    st.session_state["doc_nav"] = None


def _activate_application():
    st.session_state["ai_playground_nav"] = None
    st.session_state["learner_nav"] = None
    st.session_state["doc_nav"] = None


def _activate_docs():
    st.session_state["ai_playground_nav"] = None
    st.session_state["learner_nav"] = None
    st.session_state["application_nav"] = None

# Home button — acts as a simple toggle
if st.sidebar.button("Home", use_container_width=True):
    st.session_state["learner_nav"] = None
    st.session_state["application_nav"] = None
    st.session_state["ai_playground_nav"] = None
    st.session_state["doc_nav"] = None
    st.session_state["_page"] = "Home"
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("**AI Playground**")
st.sidebar.caption("Use AI to quickly explore uploaded datasets.")
ai_playground_page = st.sidebar.selectbox(
    "AI Playground",
    ["Explore Data", "Personality Decoder", "Palm Reader"],
    index=None,
    placeholder="Pick a tool...",
    key="ai_playground_nav",
    label_visibility="collapsed",
    on_change=_activate_ai_playground,
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Learner**")
st.sidebar.caption("Step through core neural network concepts.")
learner_page = st.sidebar.selectbox(
    "Learner",
    [
        "Perceptron",
        "Forward Propagation",
        "Backward Propagation",
        "Multi-Layer Perceptron (MLP)"
    ],
    index=None,
    placeholder="Pick a concept...",
    key="learner_nav",
    label_visibility="collapsed",
    on_change=_activate_learner,
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Application**")
st.sidebar.caption("See neural network concepts applied in real projects.")
application_page = st.sidebar.selectbox(
    "Application",
    ["OpenCV", "RNN"],
    index=None,
    placeholder="Pick an application...",
    key="application_nav",
    label_visibility="collapsed",
    on_change=_activate_application,
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Documentation**")
st.sidebar.caption("Reference guides for each module.")
doc_page = st.sidebar.selectbox(
    "Select Documentation",
    [
        "Perceptron",
        "Forward Propagation",
        "Backward Propagation",
        "Multi-Layer Perceptron (MLP)"
    ],
    index=None,
    placeholder="Pick a topic...",
    key="doc_nav",
    label_visibility="collapsed",
    on_change=_activate_docs,
)

# ---------------------------
# Page Resolution
# ---------------------------
# Priority: ai_playground > learner > application > doc > home
if ai_playground_page is not None:
    page = ai_playground_page
elif learner_page is not None:
    page = learner_page
elif application_page is not None:
    page = application_page
elif doc_page is not None:
    page = f"Docs - {doc_page}"
else:
    page = "Home"

previous_page = st.session_state.get("_active_page")
if previous_page != page:
    if page == "Palm Reader":
        _reset_palm_state()
    elif page == "Personality Decoder":
        _reset_personality_state()
    st.session_state["_active_page"] = page

# ---------------------------
# Main Content Routing
# ---------------------------
if page == "Home":

    # ---------------------------
    # Header Section (HOME ONLY)
    # ---------------------------
    st.markdown(
        """
        <h1 style="text-align:center;">NeuroLens</h1>
        <p style="text-align:center; font-size:16px;">
        Build, tune, and understand neural networks from scratch — one concept at a time.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    image_path = Path(__file__).parent / "src" / "assets" / "image" / "nn_image.jpg"
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        st.image(str(image_path), width=1200)

    st.subheader("About This Toolbox")

    st.markdown("""
    This application is an **educational neural network simulator** designed to help
    students understand how neural networks work internally.

    ### What you can do here:
    - Learn neural networks by **building them from scratch**
    - Manually tune learning parameters
    - Upload CSV datasets and experiment
    - Visualize training and learning behavior

    ### How to use:
    1. Select a module from the sidebar  
    2. Upload a dataset (if required)  
    3. Configure parameters  
    4. Train and observe results  
    """)

    st.info("Use the sidebar to navigate through different neural network concepts.")

elif page == "Perceptron":
    perceptron_page()

elif page == "Forward Propagation":
    forward_propagation_page()

elif page == "Backward Propagation":
    backward_propagation_page()

elif page == "Multi-Layer Perceptron (MLP)":
    mlp_page()

elif page == "OpenCV":
    try:
        from src.application_pages.open_cv.open_cv_landing import open_cv_landing_page
        open_cv_landing_page()
    except Exception as e:
        st.warning("OpenCV pages are available on local environment only.")
        st.info("Please run the app locally to use OpenCV features.")
        st.caption(f"Import detail: {e}")

elif page == "RNN":
    rnn_application_page()

elif page == "Explore Data":
    explore_data_page()

elif page == "Personality Decoder":
    personality_decoder_page()

elif page == "Palm Reader":
    palm_reader_page()

elif page == "Docs - Perceptron":
    perceptron_docs_page()

elif page == "Docs - Forward Propagation":
    forward_propagation_docs_page()

elif page == "Docs - Backward Propagation":
    back_propagation_docs_page()

elif page == "Docs - Multi-Layer Perceptron (MLP)":
    mnp_docs_page()

# ---------------------------
# Footer (Global, Minimal)
# ---------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px;">
    Neural Network Learning Toolbox | Built for educational purposes
    </p>
    """,
    unsafe_allow_html=True
)